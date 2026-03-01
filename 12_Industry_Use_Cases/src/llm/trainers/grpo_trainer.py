from pathlib import Path
from typing import Callable

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


def _dummy_reward_func(completions: list[str], **kwargs) -> list[float]:
    return [0.5 for _ in completions]


def train_grpo(
    dataset,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs",
    epochs: int = 1,
    reward_func: Callable | None = None,
) -> str:
    """
    Train a model using Group Relative Policy Optimization (GRPO) with TRL.

    GRPO is an online RL method suitable for tasks with verifiable rewards
    such as mathematical reasoning and code execution.

    Args:
        dataset: Dataset in GRPO format with 'prompt' field containing
                 conversation history (list of dicts with 'role' and 'content').
        base_model: HuggingFace model identifier for the base model.
        output_dir: Directory to save the trained model and tokenizer.
        epochs: Number of training epochs.
        reward_func: Callable that takes completions and returns rewards.
                     If None, uses a placeholder returning constant 0.5.

    Returns:
        Path to the output directory containing the trained model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if reward_func is None:
        reward_func = _dummy_reward_func

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
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
        reward_func=reward_func,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)
