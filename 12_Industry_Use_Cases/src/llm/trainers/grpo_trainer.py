"""GRPO Trainer with real reward function support."""

from pathlib import Path
from typing import Callable, Optional

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


class GRPOTrainingError(Exception):
    """Raised when GRPO training cannot proceed due to configuration issues."""

    pass


REWARD_TEMPLATE_ERRORS = {
    "no_template": """
GRPO requires a reward function to optimize against.

You must provide one of:
1. A built-in template (math, code_sandbox, format_check)
2. A custom reward function

GRPO (Group Relative Policy Optimization) works by:
- Generating multiple responses to the same prompt
- Scoring each with a reward function
- Learning to maximize rewards relative to other responses

Without a real reward signal, the model learns nothing.
""",
    "math_no_ground_truth": """
You selected 'math' template but did not provide ground truth answers.

The math reward function checks if the model's answer matches a known
correct answer. Your dataset must include ground truth.

Required format:
{"prompt": [...], "ground_truth": "4"}

Or for multiple choice:
{"prompt": [...], "ground_truth": "A", "options": ["A", "B", "C"]}
""",
    "format_check_no_pattern": """
You selected 'format_check' template but did not provide a pattern.

The format_check reward function validates that the response contains
a specific regex pattern. Your dataset must include a 'pattern' field.

Required format:
{"prompt": [...], "pattern": r"\\d+"}

Common patterns:
- Numbers: r"\\d+"
- JSON: r"\\{.*\\}"
- Code blocks: r"```.*```"
- Step-by-step: r"Step \\d+:.*"

GRPO (Group Relative Policy Optimization) uses this reward signal to
teach the model to follow specific formatting requirements.
""",
    "invalid_template_name": """
Invalid reward template name.

Available templates: math, code_sandbox, format_check
""",
    "custom_code_no_function": """
Custom reward code must define a function named 'reward_func'.

Your code should look like:

def reward_func(completions, **kwargs):
    # Your reward logic here
    return [1.0 if "correct" in c.lower() else 0.0 for c in completions]

The function must:
- Accept 'completions' (list of strings) and '**kwargs'
- Return a list of floats (one reward per completion)
""",
}


# Pre-built reward function templates


def create_math_reward_func(
    ground_truth_field: str = "ground_truth",
) -> Callable[[list[str], ...], list[float]]:
    """Create a reward function for math problems.

    Args:
        ground_truth_field: Field name in dataset containing correct answer

    Returns:
        Reward function that returns 1.0 for correct, 0.0 for incorrect
    """
    import re

    def reward_func(completions: list[str], **kwargs) -> list[float]:
        rewards = []

        for i, completion in enumerate(completions):
            reward = 0.0

            # Handle completion as string or list
            if isinstance(completion, list):
                completion_text = " ".join([msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in completion])
            else:
                completion_text = str(completion)
            
            completion_lower = completion_text.lower().strip()

            # Get ground truth for this sample if available
            ground_truths = kwargs.get(ground_truth_field)
            if ground_truths and i < len(ground_truths):
                ground_truth = str(ground_truths[i]).lower().strip()

                # Check various answer formats
                if ground_truth in completion_lower:
                    reward = 1.0
                # Check for boxed format: \boxed{answer}
                elif "boxed" in completion_lower:
                    match = re.search(r"\\boxed\{([^}]+)\}", completion_text, re.IGNORECASE)
                    if match and match.group(1).strip().lower() == ground_truth:
                        reward = 1.0

            rewards.append(reward)

        return rewards

    return reward_func


def create_format_check_reward_func(
    required_pattern: str,
) -> Callable[[list[str], ...], list[float]]:
    """Create a reward function that checks for required format.

    Args:
        required_pattern: Regex pattern that must be present in response

    Returns:
        Reward function returning 1.0 if pattern found, 0.0 otherwise
    """
    import re

    # Use DOTALL flag so . matches newlines (for multiline patterns)
    pattern = re.compile(required_pattern, re.DOTALL)

    def reward_func(completions: list[str], **kwargs) -> list[float]:
        return [1.0 if pattern.search(c) else 0.0 for c in completions]

    return reward_func


# Template registry
REWARD_TEMPLATES = {
    "math": {
        "name": "Math Answer Check",
        "description": "Checks if model output contains correct answer",
        "required_fields": ["ground_truth"],
        "create_func": create_math_reward_func,
    },
    "code_sandbox": {
        "name": "Code Execution",
        "description": "Executes code and checks for correct output (not yet implemented)",
        "required_fields": ["test_cases"],
        # Implementation would require code execution sandbox
    },
    "format_check": {
        "name": "Format Validation",
        "description": "Validates response matches required pattern (regex)",
        "required_fields": ["pattern"],
        "create_func": create_format_check_reward_func,
    },
}


def validate_grpo_config(
    dataset: list[dict], reward_template: str | None, custom_reward_code: str | None
) -> tuple[bool, str]:
    """Validate GRPO configuration before training.

    Args:
        dataset: Training dataset
        reward_template: Name of template to use (or None)
        custom_reward_code: Custom Python code for reward function

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not reward_template and not custom_reward_code:
        return False, REWARD_TEMPLATE_ERRORS["no_template"]

    if reward_template:
        if reward_template not in REWARD_TEMPLATES:
            available = list(REWARD_TEMPLATES.keys())
            return (
                False,
                f"{REWARD_TEMPLATE_ERRORS['invalid_template_name']}\n\nAvailable: {available}",
            )

        template = REWARD_TEMPLATES[reward_template]

        # Check required fields in dataset
        if template.get("required_fields"):
            missing_fields = []
            for field in template["required_fields"]:
                # Check first example if dataset exists
                if not dataset or field not in dataset[0]:
                    missing_fields.append(field)

            if missing_fields:
                error_key = f"{reward_template}_no_{missing_fields[0]}"
                if reward_template == "format_check":
                    error_key = f"{reward_template}_no_pattern"

                return False, (
                    f"Template '{reward_template}' requires dataset fields: {missing_fields}\n\n"
                    f"{REWARD_TEMPLATE_ERRORS.get(error_key, '')}"
                )

    # Validate custom reward code syntax if provided
    if custom_reward_code:
        try:
            compile(custom_reward_code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error in custom reward code:\n{str(e)}"

    return True, ""


def train_grpo(
    dataset,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs",
    epochs: int = 1,
    reward_template: Optional[str] = None,
    custom_reward_code: Optional[str] = None,
    reward_func: Callable | None = None,
    use_quantization: bool = True,
) -> str:
    """
    Train a model using Group Relative Policy Optimization (GRPO) with real reward functions.

    GRPO is an online RL method suitable for tasks with verifiable rewards
    such as mathematical reasoning and code execution.

    Args:
        dataset: Dataset in GRPO format with 'prompt' field containing
                 conversation history (list of dicts with 'role' and 'content').
        base_model: HuggingFace model identifier for the base model.
        output_dir: Directory to save the trained model and tokenizer.
        epochs: Number of training epochs.
        reward_template: Name of built-in template (math, code_sandbox, format_check).
                         Mutually exclusive with custom_reward_code and reward_func.
        custom_reward_code: Python code defining custom reward function.
                            Mutually exclusive with reward_template and reward_func.
        reward_func: Callable that takes completions and returns rewards.
                     If None but no template/custom_code is provided, an error is raised.
        use_quantization: Whether to use 4-bit quantization. Set to False if
                          bitsandbytes is not available or incompatible.

    Returns:
        Path to the output directory containing the trained model.

    Raises:
        GRPOTrainingError: If configuration is invalid or training fails.
    """
    # Validate configuration
    is_valid, error_message = validate_grpo_config(
        dataset=dataset, reward_template=reward_template, custom_reward_code=custom_reward_code
    )

    if not is_valid:
        raise GRPOTrainingError(error_message)

    # Create reward function
    if reward_template:
        template = REWARD_TEMPLATES[reward_template]

        # Extract parameters from dataset
        if reward_template == "math":
            ground_truth_field = "ground_truth"
            reward_func = create_math_reward_func(ground_truth_field)
        elif reward_template == "format_check":
            # Get pattern from first dataset item
            if not dataset or "pattern" not in dataset[0]:
                raise GRPOTrainingError(f"{REWARD_TEMPLATE_ERRORS['format_check_no_pattern']}")
            pattern = dataset[0].get("pattern", r".*")
            reward_func = create_format_check_reward_func(pattern)
        else:
            raise GRPOTrainingError(f"Template '{reward_template}' is not yet implemented")

    elif custom_reward_code:
        # Execute user-provided reward function code
        local_ns = {}
        try:
            exec(custom_reward_code, {}, local_ns)
        except Exception as e:
            raise GRPOTrainingError(f"Failed to execute custom reward code: {str(e)}")

        if "reward_func" not in local_ns:
            raise GRPOTrainingError(REWARD_TEMPLATE_ERRORS["custom_code_no_function"])

        reward_func = local_ns["reward_func"]

    elif reward_func is None:
        raise GRPOTrainingError(REWARD_TEMPLATE_ERRORS["no_template"])

    # Now proceed with training using the reward function
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_kwargs = {"device_map": "auto"}
    
    if use_quantization:
        try:
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        except (ImportError, RuntimeError) as e:
            print(f"Warning: bitsandbytes not available ({e}), using float16 instead")
            use_quantization = False
    
    if not use_quantization:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_strategy="no",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)
