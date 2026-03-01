# Agentic AutoML Platform - Implementation Plan

**Generated:** February 2026  
**Goal:** Achieve 95+ scores on all code review criteria

---

## Executive Summary

This plan addresses gaps identified in the code review to bring all components from their current scores (80 overall) to 95+. The plan prioritizes simplicity and clarity so any developer can pick up where we leave off.

### Current Scores

| Goal | Current Score |
|------|---------------|
| 1. File Uploads (CSV/TXT/PDF) | 85/100 |
| 2. Auto-detect Task Types | 90/100 |
| 3. Train Models (ML/LLM) | 80/100 |
| 4. Downloadable + Inference | 65/100 |
| **OVERALL** | **80/100** |

### Target Scores

All goals at 95+ for an overall score of **96+/100**

---

## Architecture Decisions

These decisions were made based on project constraints and user requirements:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Metaflow metadata storage | Local file-based (`.metaflow` directory) | Simplest setup, no infrastructure needed |
| GRPO reward functions | Both template dropdown + custom code editor | Flexibility for users while providing common options |
| LLM playground chat UI | Full OpenAI-style with parameters | Richer UX with streaming, system prompts, temperature controls |
| NVIDIA container version | `nvcr.io/nvidia/pytorch:26.01-py3` | Latest with Blackwell GPU support |

---

## Critical Priority Items

### 1. Docker NVIDIA PyTorch Base Image (CRITICAL)

**Why this matters:** The current Dockerfile uses `python:3.11-slim` which has NO GPU support. This completely breaks LLM training on NVIDIA hardware.

**File to modify:** `docker/Dockerfile`

**Current (WRONG):**
```dockerfile
FROM python:3.11-slim AS builder
```

**Target (CORRECT):**
```dockerfile
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS builder
```

**Full context:** The `nvcr.io/nvidia/pytorch:26.01-py3` container:
- Comes with CUDA 12.x, cuDNN, PyTorch pre-installed
- Supports NVIDIA Blackwell, Hopper, Ada Lovelace architectures
- Is the official NVIDIA recommendation for ML workloads

**Other changes needed in Dockerfile:**
1. The NVIDIA image already has Python 3.x, so remove any Python version installation
2. UV may or may not be present - test and add if needed:
   ```dockerfile
   RUN which uv || pip install uv
   ```
3. Keep the multi-stage build pattern but verify all dependencies work with the NVIDIA image

**File to modify:** `docker/entrypoint.sh`

Add GPU validation:
```bash
# Check for Blackwell GPU (or any NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader
else
    echo "WARNING: No NVIDIA GPU detected. LLM training will run on CPU (very slow)."
fi
```

**Trade-offs:**
- Pro: Full GPU acceleration for LLM training
- Con: Container size larger (~10GB vs ~1GB)
- Con: Requires `nvidia-container-toolkit` on host for GPU passthrough

---

### 2. LLM Inference Implementation (CRITICAL)

**Why this matters:** Currently the inference playground only supports sklearn model prediction. Users cannot test their fine-tuned LLMs at all.

**New file to create:** `src/llm/inference_server.py`

**Implementation:**

```python
"""LLM Inference Server - OpenAI-compatible API client."""

from typing import Generator, Optional
import openai


class LLMInferenceServer:
    """Client for OpenAI-compatible LLM inference endpoints."""
    
    def __init__(
        self,
        base_url: str = "http://192.168.1.79:8080/v1",
        api_key: str = "not-needed"
    ):
        """Initialize the inference server.
        
        Args:
            base_url: Base URL for the OpenAI-compatible API
            api_key: API key (often not needed for local endpoints)
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "minimax-m2.5-mlx@8bit",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate a response from the LLM.
        
        Args:
            prompt: User message
            system_prompt: System instruction (optional)
            model: Model name to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        if stream:
            return self._stream_response(**kwargs)
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def _stream_response(self, **kwargs) -> Generator[str, None, None]:
        """Stream response from the API."""
        kwargs["stream"] = True
        response = self.client.chat.completions.create(**kwargs)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Global instance for use in Gradio app
_inference_server: Optional[LLMInferenceServer] = None


def get_inference_server() -> LLMInferenceServer:
    """Get or create the global inference server instance."""
    global _inference_server
    if _inference_server is None:
        # Load from config or use defaults
        from src.config import settings
        _inference_server = LLMInferenceServer(
            base_url=settings.LLM_INFERENCE_URL,
            api_key=settings.LLM_INFERENCE_KEY
        )
    return _inference_server


def generate_response(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "minimax-m2.5-mlx@8bit",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0
) -> str:
    """Generate a non-streaming response."""
    server = get_inference_server()
    return server.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False
    )


def generate_streaming(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "minimax-m2.5-mlx@8bit",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0
) -> Generator[str, None, None]:
    """Generate a streaming response."""
    server = get_inference_server()
    return server.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True
    )
```

**Configuration to add in `src/config.py`:**

```python
# LLM Inference settings
LLM_INFERENCE_URL: str = "http://192.168.1.79:8080/v1"
LLM_INFERENCE_KEY: str = "not-needed"  # Change if your endpoint requires auth
```

**Dependencies to add in `pyproject.toml`:**

```toml
dependencies = [
    # ... existing dependencies ...
    "openai>=1.0",  # For OpenAI-compatible API client
]
```

**Trade-offs:**
- Pro: Works with any OpenAI-compatible endpoint (local or remote)
- Pro: Supports streaming for better UX
- Con: Requires external API endpoint to be running
- Con: No built-in LoRA adapter merging (would need separate implementation)

---

### 3. Metaflow Integration for All Training (CRITICAL)

**Why this matters:** Currently training happens directly in app.py via `train_ml_model_direct()`. This bypasses Metaflow entirely, losing:
- Experiment tracking
- Artifact versioning  
- Visual cards/reports
- Checkpointing for resume

**Two approaches:**

#### Approach A: Run Metaflow as subprocess (Easier, less integrated)

In `src/app.py`, replace direct training with:

```python
import subprocess
import json


def train_with_metaflow(data_path: str, target_column: str) -> dict:
    """Run Metaflow training flow and return results."""
    
    # Run the flow
    result = subprocess.run(
        [
            "python", "-m", "src.flows.ml_training_flow",
            "run",
            "--data_path", data_path,
            "--target_column", target_column
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Metaflow training failed: {result.stderr}")
    
    # Parse output for results
    # Metaflow outputs artifact values to stdout in JSON format with --json-output
    # Or we can query the metadata store directly
    
    return {
        "status": "completed",
        # Extract model_path, metrics from Metaflow artifacts
    }
```

#### Approach B: Use Metaflow Python API (More integrated, recommended)

Create a helper module `src/flows/runner.py`:

```python
"""Metaflow flow execution helpers."""

import os
import time
from typing import Optional

# Set local metadata store before importing Metaflow flows
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"


def run_ml_training_flow(
    data_path: str,
    target_column: str,
    wait_for_completion: bool = True
) -> str:
    """Run ML training flow via Metaflow Python API.
    
    Args:
        data_path: Path to CSV file
        target_column: Target column name
        wait_for_completion: If True, block until flow completes
        
    Returns:
        Flow run ID (e.g., "MLTrainingFlow/123")
        
    Raises:
        RuntimeError: If flow fails
    """
    from metaflow import Flow, Run
    
    # Import the flow class - must be done after setting metadata
    from src.flows.ml_training_flow import MLTrainingFlow
    
    # Create and run the flow
    flow = MLTrainingFlow(
        data_path=data_path,
        target_column=target_column
    )
    
    # Run the flow - this executes all steps locally
    run = flow.run()
    
    if wait_for_completion:
        # In local mode, run() blocks until complete
        # Check for failures
        if not flow.successful:
            raise RuntimeError(f"Flow failed: {flow.stderr}")
    
    # Return the run ID
    return f"MLTrainingFlow/{run.id}"


def get_flow_artifacts(run_id: str) -> dict:
    """Get artifacts from a completed flow run.
    
    Args:
        run_id: Flow run ID (e.g., "MLTrainingFlow/123")
        
    Returns:
        Dictionary of artifacts from the run
    """
    flow_name, run_number = run_id.split("/")
    
    run = Flow(flow_name)[int(run_number)]
    
    return {
        "model_path": run.data.model_path,
        "task_type": run.data.task_type,
        "metrics": run.data.metrics,
    }


def poll_flow_status(run_id: str) -> dict:
    """Poll a running flow for status and progress.
    
    Args:
        run_id: Flow run ID
        
    Returns:
        Status dictionary with state, progress percentage, current step
    """
    flow_name, run_number = run_id.split("/")
    
    try:
        run = Flow(flow_name)[int(run_number)]
    except KeyError:
        return {"state": "not_found", "progress": 0.0}
    
    if run.finished:
        return {
            "state": "completed" if run.successful else "failed",
            "progress": 1.0,
            "current_step": "finished"
        }
    
    # Get current step from task
    current_task = run.task("end")  # Check if we've reached end
    
    return {
        "state": "running",
        "progress": 0.5,  # Would need more sophisticated tracking
        "current_step": "training"  # Could inspect run tasks
    }
```

**Enhancing the flow with Metaflow cards:**

Modify `src/flows/ml_training_flow.py`:

```python
from metaflow import card, current
from metaflow.cards import Markdown, Table, Image


class MLTrainingFlow(FlowSpec):
    """Metaflow pipeline for automated ML model training."""
    
    data_path = Parameter("data_path", required=True)
    target_column = Parameter("target_column", required=True)

    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        # ... existing code ...
        
        # Add a card showing data summary
        self.card = Markdown(f"""
        ## Data Loaded
        
        - Rows: {len(self.df)}
        - Columns: {len(self.df.columns)}
        - File: {self.data_path}
        
        ### Column Types
        | Column | Type |
        |--------|------|
        {chr(10).join(f"| {col} | {dtype} |" for col, dtype in self.df.dtypes.items())}
        """)
        
        self.next(self.validate_data)

    # ... other steps ...

    @card(type="html")  # This creates an automatic card
    @step
    def evaluate(self):
        """Evaluate model and create result card."""
        
        # ... existing evaluation code ...
        
        # Create a beautiful results table
        if self.task_type == "classification":
            metrics_table = Table(
                ["Metric", "Value"],
                data=[
                    ["Accuracy", f"{self.metrics.get('accuracy', 0):.2%}"],
                    ["F1 (Macro)", f"{self.metrics.get('f1_macro', 0):.4f}"],
                    ["Precision", f"{self.metrics.get('precision_macro', 0):.4f}"],
                    ["Recall", f"{self.metrics.get('recall_macro', 0):.4f}"],
                ]
            )
        else:
            metrics_table = Table(
                ["Metric", "Value"],
                data=[
                    ["RMSE", f"{self.metrics.get('rmse', 0):.4f}"],
                    ["MAE", f"{self.metrics.get('mae', 0):.4f}"],
                    ["R²", f"{self.metrics.get('r2', 0):.4f}"],
                ]
            )
        
        # Add to card
        current.card.append(metrics_table)
        
        self.next(self.save_model)

    @step
    def save_model(self):
        # ... existing code ...
        
        self.next(self.end)

    @step
    def end(self):
        print(f"Training complete!")
        print(f"Task type: {self.task_type}")
        print(f"Model saved to: {self.model_path}")
```

**New file needed:** `src/flows/llm_training_flow.py`

This should mirror the structure of ml_training_flow.py but for LLM training. See the implementation in `todo.md` lines 795-956 as reference.

**Refactoring app.py:**

Replace the direct training call:
```python
# OLD CODE (bypasses Metaflow):
training_result = train_ml_model_direct(filepath, target_column)

# NEW CODE (uses Metaflow):
from src.flows.runner import run_ml_training_flow, get_flow_artifacts

# Submit job and start flow
run_id = run_ml_training_flow(data_path=filepath, target_column=target)

# Get results from Metaflow artifacts
artifacts = get_flow_artifacts(run_id)
model_path = artifacts["model_path"]
metrics = artifacts["metrics"]
```

**Trade-offs:**
- Pro: Full experiment tracking and artifact versioning
- Pro: Beautiful visual cards for training results
- Pro: Checkpointing support for long-running jobs
- Con: More complex setup and debugging
- Con: Local execution doesn't parallelize like cloud Metaflow

---

### 4. Real DPO Preference Data Handling (CRITICAL)

**Why this matters:** Currently `_generate_rejected_response()` creates fake "worse" answers. This produces meaningless DPO training because the rejected responses aren't real human preferences.

**File to modify:** `src/llm/dataset_converter.py`

**Current problem code (REMOVE):**
```python
def _generate_rejected_response(good_answer: str) -> str:
    """Generate a deliberately worse response for DPO training."""
    words = good_answer.split()
    if len(words) > 10:
        truncated = " ".join(words[:5])
        return f"{truncated}... [incomplete response]"
    elif len(words) > 3:
        return words[0]
    else:
        return "I don't know."
```

**New validation code to ADD:**

```python
DPO_VALIDATION_ERRORS = {
    "missing_rejected": """
Your DPO dataset is missing 'rejected' responses.

DPO (Direct Preference Optimization) requires preference pairs that reflect 
actual human judgments about which response is better. You cannot generate 
rejected responses automatically - they must come from real preference data.

REQUIRED FORMAT (JSONL):
{"prompt": [{"role": "user", "content": "What is 2+2?"}], 
 "chosen": [{"role": "assistant", "content": "The answer is 4."}], 
 "rejected": [{"role": "assistant", "content": "I don't know."}]}

HOW TO CREATE DPO DATA:
1. Human annotation: Have humans rate/compare model outputs
2. LLM-as-judge: Use a stronger model to evaluate responses  
3. Existing datasets: Download from HuggingFace (e.g., Anthropic HH-RLHF)
4. Custom collection: Collect prompts + 2+ responses with human rankings

See: https://huggingface.co/docs/trl/dpo_trainer#dataset-format
""",
    
    "missing_chosen": """
Your DPO dataset is missing 'chosen' responses.

Each example needs both a 'chosen' (preferred) and 'rejected' (disfavored) 
response. The chosen response should be the one a human would prefer.
""",
    
    "invalid_format": """
Your DPO file is not in the expected format.

Expected: JSONL (JSON Lines) with one JSON object per line
Each object must have: 'prompt', 'chosen', 'rejected'

Example (one line):
{"prompt": [{"role": "user", "content": "Hello!"}], 
 "chosen": [{"role": "assistant", "content": "Hi there!"}], 
 "rejected": [{"role": "assistant", "content": "yo"}]}
""",
    
    "empty_file": """
Your DPO file is empty or contains no valid examples.

DPO training requires at least several hundred preference pairs for 
meaningful results. Consider:
- Using an existing DPO dataset from HuggingFace
- Collecting more preference data
""",
}


def validate_dpo_format(examples: list[dict]) -> tuple[bool, str]:
    """Validate DPO dataset format.
    
    Args:
        examples: List of potential DPO examples
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Error messages are verbose and actionable.
    """
    if not examples:
        return False, DPO_VALIDATION_ERRORS["empty_file"]
    
    for i, example in enumerate(examples):
        # Check required fields
        if "prompt" not in example:
            return False, f"Example {i}: Missing 'prompt' field.\n\n{DPO_VALIDATION_ERRORS['missing_chosen']}"
        
        if "chosen" not in example:
            return False, f"Example {i}: Missing 'chosen' field.\n\n{DPO_VALIDATION_ERRORS['missing_chosen']}"
        
        if "rejected" not in example:
            return False, f"Example {i}: Missing 'rejected' field.\n\n{DPO_VALIDATION_ERRORS['missing_rejected']}"
        
        # Validate that chosen != rejected (common mistake)
        if example.get("chosen") == example.get("rejected"):
            return False, f"Example {i}: 'chosen' and 'rejected' are identical. They must be different responses."
    
    return True, ""


def convert_to_dpo_format(filepath: str) -> list[dict]:
    """Convert a file to DPO training format with validation.
    
    Args:
        filepath: Path to the input TXT/JSONL file
        
    Returns:
        List of training examples in DPO format
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = path.suffix.lower()
    
    # Support both TXT (legacy) and JSONL (preferred)
    if suffix == ".jsonl":
        examples = _parse_jsonl_file(filepath)
    elif suffix == ".txt":
        # Legacy format - warn that this is limited
        import warnings
        warnings.warn(
            "TXT format for DPO is deprecated. Please use JSONL format "
            "for proper preference data. See documentation for details.",
            DeprecationWarning
        )
        examples = _convert_txt_to_dpo_legacy(filepath)
    else:
        raise ValueError(
            f"Unsupported file type for DPO: {suffix}. "
            "Use .jsonl (recommended) or .txt."
        )
    
    # Validate the dataset
    is_valid, error_message = validate_dpo_format(examples)
    if not is_valid:
        raise ValueError(error_message)
    
    return examples


def _parse_jsonl_file(filepath: str) -> list[dict]:
    """Parse a JSONL file into a list of dicts."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {e}\n"
                    "Each line must be a valid JSON object."
                )
    return examples


def _convert_txt_to_dpo_legacy(filepath: str) -> list[dict]:
    """Legacy TXT to DPO conversion - DEPRECATED.
    
    This creates synthetic rejected responses which are NOT suitable 
    for real DPO training. Included for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "Converting TXT to DPO format creates SYNTHETIC rejected responses. "
        "This is NOT suitable for meaningful DPO training.",
        UserWarning
    )
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    examples = []
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        if not (question and answer):
            continue
        
        # Generate synthetic rejected - but warn!
        words = answer.split()
        if len(words) > 10:
            rejected = " ".join(words[:3]) + "..."
        else:
            rejected = "I don't know."
        
        examples.append({
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": answer}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    
    return examples
```

**Update app.py to use new validation:**

```python
# In train_llm_model function, for DPO:
if training_method == "DPO":
    from src.llm.dataset_converter import convert_to_dpo_format
    
    try:
        dataset = convert_to_dpo_format(filepath)
    except ValueError as e:
        return format_error_html(str(e)), None
```

**Trade-offs:**
- Pro: Real DPO training actually works (meaningful preference learning)
- Pro: Verbose errors guide users to correct solution
- Con: Users MUST have real preference data (can't fake it)
- Con: Backward compatibility with old TXT format is limited

---

### 5. Real GRPO Reward Functions (CRITICAL)

**Why this matters:** The `_dummy_reward_func` returns constant 0.5 for all responses, which provides zero learning signal.

**File to modify:** `src/llm/trainers/grpo_trainer.py`

**Current problem code (REMOVE):**
```python
def _dummy_reward_func(completions: list[str], **kwargs) -> list[float]:
    return [0.5 for _ in completions]
```

**New implementation:**

```python
"""GRPO Trainer with real reward function support."""

from pathlib import Path
from typing import Callable, Optional
import json


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
    
    "invalid_template_name": """
Invalid reward template name.

Available templates: math, code_sandbox, format_check
""",
}


# Pre-built reward function templates

def create_math_reward_func(
    ground_truth_field: str = "ground_truth"
) -> Callable[[list[dict], list[str]], list[float]]:
    """Create a reward function for math problems.
    
    Args:
        ground_truth_field: Field name in dataset containing correct answer
        
    Returns:
        Reward function that returns 1.0 for correct, 0.0 for incorrect
    """
    def reward_func(
        completions: list[str],
        prompts: list[dict] | None = None,
        dataset: list[dict] | None = None,
        **kwargs
    ) -> list[float]:
        rewards = []
        
        for i, completion in enumerate(completions):
            reward = 0.0
            
            # Try to extract answer from completion
            completion_lower = completion.lower().strip()
            
            if dataset and i < len(dataset):
                ground_truth = str(dataset[i].get(ground_truth_field, "")).lower().strip()
                
                # Check various answer formats
                if ground_truth in completion_lower:
                    reward = 1.0
                # Check for boxed format: \boxed{answer}
                elif "boxed" in completion_lower:
                    import re
                    match = re.search(r'\\boxed\{([^}]+)\}', completion, re.IGNORECASE)
                    if match and match.group(1).strip().lower() == ground_truth:
                        reward = 1.0
            
            rewards.append(reward)
        
        return rewards
    
    return reward_func


def create_format_check_reward_func(
    required_pattern: str
) -> Callable[[list[str], ...], list[float]]:
    """Create a reward function that checks for required format.
    
    Args:
        required_pattern: Regex pattern that must be present in response
        
    Returns:
        Reward function returning 1.0 if pattern found, 0.0 otherwise
    """
    import re
    
    pattern = re.compile(required_pattern)
    
    def reward_func(
        completions: list[str],
        **kwargs
    ) -> list[float]:
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
        "description": "Executes code and checks for correct output",
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
    dataset: list[dict],
    reward_template: str | None,
    custom_reward_code: str | None
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
            return False, f"{REWARD_TEMPLATE_ERRORS['invalid_template_name']}\n\nAvailable: {list(REWARD_TEMPLATES.keys())}"
        
        template = REWARD_TEMPLATES[reward_template]
        
        # Check required fields in dataset
        if template.get("required_fields"):
            missing_fields = []
            for field in template["required_fields"]:
                # Check first example
                if not dataset or not any(field in ex for ex in dataset):
                    missing_fields.append(field)
            
            if missing_fields:
                return False, f"Template '{reward_template}' requires dataset fields: {missing_fields}\n\n{REWARD_TEMPLATE_ERRORS.get(f'{reward_template}_no_ground_truth', '')}"
    
    return True, ""


def train_grpo(
    dataset: list[dict],
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs",
    epochs: int = 1,
    reward_template: Optional[str] = None,
    custom_reward_code: Optional[str] = None,
) -> str:
    """Train a model using GRPO with real reward functions.
    
    Args:
        dataset: Dataset in GRPO format
        base_model: HuggingFace model identifier
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        reward_template: Name of built-in template (math, code_sandbox, format_check)
        custom_reward_code: Python code defining custom reward function
        
    Returns:
        Path to output directory
    """
    # Validate configuration
    is_valid, error_message = validate_grpo_config(
        dataset=dataset,
        reward_template=reward_template,
        custom_reward_code=custom_reward_code
    )
    
    if not is_valid:
        raise GRPOTrainingError(error_message)
    
    # Create reward function
    if reward_template:
        template = REWARD_TEMPLATES[reward_template]
        
        # Extract parameters from dataset
        if reward_template == "math":
            ground_truth_field = dataset[0].get("ground_truth_col", "ground_truth")
            reward_func = create_math_reward_func(ground_truth_field)
        elif reward_template == "format_check":
            pattern = dataset[0].get("pattern", r".*")
            reward_func = create_format_check_reward_func(pattern)
        else:
            raise GRPOTrainingError(f"Template '{reward_template}' not yet implemented")
    
    elif custom_reward_code:
        # Execute user-provided reward function code
        local_ns = {}
        exec(custom_reward_code, {}, local_ns)
        
        if "reward_func" not in local_ns:
            raise GRPOTrainingError(
                "Custom reward code must define a function named 'reward_func'"
            )
        
        reward_func = local_ns["reward_func"]
    
    else:
        raise GRPOTrainingError(REWARD_TEMPLATE_ERRORS["no_template"])
    
    # Now proceed with training using the reward function
    # (Rest of existing train_grpo implementation, but with real reward_func)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ... model loading and training code ...
    
    return str(output_path)
```

**UI integration in app.py:**

```python
# Add to LLM training tab:
with gr.Accordion("GRPO Reward Configuration", open=False):
    reward_template = gr.Dropdown(
        label="Reward Template",
        choices=["math", "code_sandbox", "format_check"],
        value=None,
    )
    
    custom_reward_code = gr.Code(
        label="Custom Reward Function (Python)",
        language="python",
        value='def reward_func(completions, **kwargs):\n    # Your reward logic here\n    return [1.0 if "correct" in c.lower() else 0.0 for c in completions]',
    )

# On train button click, pass reward config to trainer
```

**Trade-offs:**
- Pro: Real learning signal for GRPO training
- Pro: Flexible templates + custom code
- Con: Requires user to understand reward function design
- Con: Code sandbox for execution is security concern (need isolation)

---

## Medium Priority Items

### 6. Excel/TSV File Support

**Why this matters:** CSV is common but enterprise data often comes as Excel (.xlsx) or TSV (tab-separated).

**Files to modify:**
- `src/ml/data_validator.py`
- `src/app.py`

**Dependencies to add in pyproject.toml:**
```toml
dependencies = [
    # ... existing ...
    "openpyxl>=3.1",  # For Excel support
]
```

**Implementation in data_validator.py:**

```python
import pandas as pd


def detect_file_format(filepath: str) -> str:
    """Detect file format from extension.
    
    Returns:
        'csv', 'tsv', 'xlsx', or raises ValueError
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    
    format_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "tsv",  # TXT often means TSV
        ".xlsx": "excel",
        ".xls": "excel",
    }
    
    if suffix not in format_map:
        raise ValueError(
            f"Unsupported file type: {suffix}\n"
            "Supported types: .csv, .tsv, .xlsx, .xls"
        )
    
    return format_map[suffix]


def read_data_file(filepath: str) -> pd.DataFrame:
    """Read data file into pandas DataFrame.
    
    Supports CSV, TSV, and Excel formats.
    """
    fmt = detect_file_format(filepath)
    
    if fmt == "csv":
        return pd.read_csv(filepath)
    elif fmt == "tsv":
        # Auto-detect delimiter for TSV
        return pd.read_csv(filepath, sep="\t")
    elif fmt == "excel":
        return pd.read_excel(filepath)
    
    raise ValueError(f"Unknown format: {fmt}")


def validate_csv(
    filepath: str,
    target_column: Optional[str] = None
) -> dict:
    """Validate data file (CSV, TSV, or Excel).
    
    This function now accepts any supported format and validates
    the structure appropriately.
    """
    path = Path(filepath)
    
    if not path.exists():
        return {
            "valid": False,
            "message": f"File not found: {filepath}"
        }
    
    try:
        df = read_data_file(filepath)
    except Exception as e:
        return {
            "valid": False,
            "message": f"Failed to read file: {e}"
        }
    
    if df.empty:
        return {
            "valid": False,
            "message": "File is empty or contains no data"
        }
    
    # Existing validation logic remains the same...
    columns = list(df.columns)
    
    if target_column and target_column not in columns:
        return {
            "valid": False,
            "message": f"Target column '{target_column}' not found in file. "
                       f"Available columns: {columns}"
        }
    
    return {
        "valid": True,
        "message": f"File validated successfully ({len(df)} rows, {len(columns)} columns)",
        "columns": columns,
        "row_count": len(df),
    }
```

**Update app.py file filters:**
```python
# In create_tabular_ml_tab:
csv_file = gr.File(
    label="Upload Data File",
    file_types=[".csv", ".tsv", ".xlsx", ".xls"],  # Expanded
    type="filepath",
)
```

**Trade-offs:**
- Pro: Supports common enterprise formats
- Con: openpyxl adds dependency
- Con: Excel files may have multiple sheets (need UI for sheet selection)

---

### 7. Enhanced Metaflow Cards and Checkpointing

**Add to ml_training_flow.py:**

```python
from metaflow import checkpoint


@checkpoint
@step
def train_model(self):
    """Train model with checkpoint support."""
    # If this step is interrupted, it can resume from here
    result = train_ensemble(self.x_train, self.y_train, self.task_type)
    self.model = result["model"]
    self.estimators = result["estimators"]
    self.next(self.evaluate)
```

**Card for data visualization:**

```python
@card(type="html")
@step
def preprocess(self):
    # ... existing code ...
    
    # Add visualization to card
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    self.df.hist(ax=ax, bins=20)
    
    current.card.append(Image.from_matplotlib(fig))
    
    self.next(self.train_model)
```

---

## Implementation Order

### Step 1: Infrastructure (Start here)
1. Update `docker/Dockerfile` to NVIDIA PyTorch 26.01
2. Add `openai`, `openpyxl` to dependencies in pyproject.toml
3. Update entrypoint.sh with GPU checks

### Step 2: Data Validation (Quick win)
1. Add Excel/TSV support to data_validator.py
2. Update app.py file filters

### Step 3: LLM Inference (High impact)
1. Create `src/llm/inference_server.py`
2. Add config settings
3. Update Inference Playground in app.py with LLM chat UI

### Step 4: Metaflow Integration (Core feature)
1. Create `src/flows/runner.py` helper module
2. Enhance ml_training_flow.py with cards
3. Create llm_training_flow.py
4. Refactor app.py to use flows

### 5. DPO/GRPO Fixes (Ensure quality)
1. Update dataset_converter.py with real validation
2. Remove dummy reward functions from grpo_trainer.py
3. Add GRPO UI in app.py

### 6. Testing
1. Update test files for new functionality
2. Run full test suite

---

## File Changes Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `src/llm/inference_server.py` | OpenAI-compatible LLM inference client |
| `src/flows/runner.py` | Metaflow flow execution helpers |
| `src/flows/llm_training_flow.py` | LLM training Metaflow pipeline |

### Files to Modify

| File | Changes |
|------|---------|
| `docker/Dockerfile` | NVIDIA PyTorch base image |
| `docker/entrypoint.sh` | GPU validation |
| `pyproject.toml` | Add openai, openpyxl dependencies |
| `src/config.py` | LLM inference URL settings |
| `src/app.py` | Metaflow integration, new playground UI |
| `src/ml/data_validator.py` | Excel/TSV support |
| `src/llm/dataset_converter.py` | Real DPO validation |
| `src/llm/trainers/grpo_trainer.py` | Remove dummy, add templates |
| `src/flows/ml_training_flow.py` | Add cards |

### Files to Delete (after verification)

| File | Reason |
|------|--------|
| `src/llm/dataset_converter.py` (old `_generate_rejected_response`) | Replaced with validation |
| `src/llm/trainers/grpo_trainer.py` (old `_dummy_reward_func`) | Replaced with templates |

---

## Testing Commands

After implementing changes:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_ml_pipeline.py -v
uv run pytest tests/test_llm_pipeline.py -v

# Lint check
uv run ruff check src/

# Format check  
uv run ruff format --check src/
```

---

## Troubleshooting

### Docker won't build with NVIDIA image
- Ensure `nvidia-container-toolkit` is installed on host
- Test: `docker run --gpus all nvidia/cuda:12.0 nvidia-smi`

### Metaflow cards not showing
- Run `metaflow card view` in the flow directory
- For local metadata, check `.metaflow/` directory

### LLM inference fails to connect
- Verify endpoint is running: `curl http://192.168.1.79:8080/v1/models`
- Check firewall settings

### DPO validation fails
- Ensure your JSONL file has proper structure (one JSON object per line)
- Validate with: `python -c "import json; [json.loads(l) for l in open('file.jsonl')]"`

---

## References

- NVIDIA PyTorch Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Metaflow Cards: https://docs.metaflow.org/api/cards
- TRL DPO Trainer: https://huggingface.co/docs/trl/dpo_trainer
- TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
- OpenAI Python SDK: https://github.com/openai/openai-python

---

## Estimated Effort

| Task | Hours |
|------|-------|
| Docker infrastructure | 1-2 |
| Excel/TSV support | 0.5-1 |
| LLM inference server | 2-3 |
| Metaflow integration | 4-6 |
| DPO/GRPO fixes | 2-3 |
| Testing | 1-2 |
| **Total** | **10.5-17 hours** |

---

*End of Implementation Plan*