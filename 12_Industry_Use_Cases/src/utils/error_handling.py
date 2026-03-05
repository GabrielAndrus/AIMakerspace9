"""Comprehensive Error Handling for ML/LLM Training.

Provides user-friendly error messages and recovery strategies
for common training failures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorCategory(Enum):
    HARDWARE = "hardware"
    DATA = "data"
    MODEL = "model"
    TRAINING = "training"
    PIPELINE = "pipeline"
    NETWORK = "network"
    CONFIGURATION = "configuration"


class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ActionableStep:
    """A specific action the user can take to resolve an error."""

    description: str
    action_type: str

    command: Optional[str] = None
    config_key: Optional[str] = None
    config_value: Optional[Any] = None

    def format(self) -> str:
        lines = [f"  • {self.description}"]

        if self.command:
            lines.append(f"    → Run: `{self.command}`")
        elif self.config_key and self.config_value is not None:
            lines.append(f"    → Set: {self.config_key} = {self.config_value}")

        return "\n".join(lines)


@dataclass
class UserFriendlyError(Exception):
    """An error message optimized for user comprehension and action."""

    error_code: str
    category: ErrorCategory
    severity: ErrorSeverity

    title: str
    summary_template: str

    context: dict = field(default_factory=dict)
    actionable_steps: list[ActionableStep] = field(default_factory=list)

    documentation_url: Optional[str] = None

    def __post_init__(self):
        try:
            self.summary = self.summary_template.format(**self.context)
        except KeyError:
            self.summary = self.summary_template

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"[{self.severity.value.upper()}] {self.title}",
            "=" * 60,
            "",
            self.summary,
            "",
        ]

        if self.actionable_steps:
            lines.append("How to fix:")
            for step in self.actionable_steps:
                lines.append(step.format())

        if self.documentation_url:
            lines.append(f"\nLearn more: {self.documentation_url}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "summary": self.summary,
            "actionable_steps": [
                {"description": s.description, "type": s.action_type} for s in self.actionable_steps
            ],
        }


ERROR_REGISTRY = {
    "cuda_oom": UserFriendlyError(
        error_code="cuda_oom",
        category=ErrorCategory.HARDWARE,
        severity=ErrorSeverity.ERROR,
        title="GPU Out of Memory",
        summary_template=(
            "Your GPU ran out of memory while training. "
            "Attempted to allocate {attempted_mb} MB but only {free_mb} MB available."
        ),
        context={},
        actionable_steps=[
            ActionableStep(
                description="Reduce batch size by half",
                action_type="config_change",
                config_key="training.batch_size",
                config_value="{suggested_batch}",
            ),
            ActionableStep(
                description="Enable gradient accumulation (simulates larger batch)",
                action_type="config_change",
                config_key="training.gradient_accumulation_steps",
                config_value=4,
            ),
            ActionableStep(
                description="Enable mixed precision (FP16) - uses 50% less memory",
                action_type="config_change",
                config_key="training.fp16",
                config_value=True,
            ),
        ],
        documentation_url="docs/troubleshooting/memory.md",
    ),
    "dataset_not_found": UserFriendlyError(
        error_code="dataset_not_found",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.CRITICAL,
        title="Dataset Not Found",
        summary_template="The training dataset file could not be found at: {dataset_path}",
        context={},
        actionable_steps=[
            ActionableStep(
                description="Verify file exists",
                action_type="command",
                command="ls -la {dataset_path}",
            ),
            ActionableStep(
                description="Check for typos in file path",
                action_type="file_edit",
            ),
        ],
    ),
    "invalid_csv_format": UserFriendlyError(
        error_code="invalid_csv_format",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        title="Invalid CSV Format",
        summary_template="The uploaded file is not a valid CSV: {issue}",
        context={},
        actionable_steps=[
            ActionableStep(
                description="Validate CSV structure with pandas",
                action_type="command",
                command="python -c \"import pandas as pd; print(pd.read_csv('{dataset_path}').head())\"",
            ),
            ActionableStep(
                description="Check encoding (try UTF-8 or Latin-1)",
                action_type="file_edit",
            ),
        ],
    ),
    "missing_target_column": UserFriendlyError(
        error_code="missing_target_column",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        title="Target Column Not Found",
        summary_template=(
            "The specified target column '{target_column}' does not exist. "
            "Available columns: {available_columns}"
        ),
        context={},
        actionable_steps=[
            ActionableStep(
                description="View file headers to see available columns",
                action_type="command",
                command="head -1 {dataset_path}",
            ),
        ],
    ),
    "dpo_missing_rejected": UserFriendlyError(
        error_code="dpo_missing_rejected",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        title="DPO Dataset Missing 'rejected' Responses",
        summary_template=(
            "Your DPO dataset is missing 'rejected' responses. "
            "DPO requires preference pairs that reflect actual human judgments."
        ),
        context={},
        actionable_steps=[
            ActionableStep(
                description="Use existing DPO dataset from Hugging Face",
                action_type="command",
                command="search 'dpo' on huggingface.co/datasets",
            ),
        ],
        documentation_url="https://huggingface.co/docs/trl/dpo_trainer#dataset-format",
    ),
    "grpo_no_reward": UserFriendlyError(
        error_code="grpo_no_reward",
        category=ErrorCategory.TRAINING,
        severity=ErrorSeverity.ERROR,
        title="GRPO Missing Reward Function",
        summary_template=(
            "GRPO requires a reward function to evaluate model outputs. "
            "Without one, the model has no learning signal."
        ),
        context={},
        actionable_steps=[
            ActionableStep(
                description="Select a built-in reward template",
                action_type="config_change",
                config_key="grpo.reward_template",
                config_value="{available_templates}",
            ),
            ActionableStep(
                description="Provide custom reward function code",
                action_type="file_edit",
            ),
        ],
    ),
    "model_download_failed": UserFriendlyError(
        error_code="model_download_failed",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        title="Could Not Download Model",
        summary_template=(
            "Failed to download '{model_name}' from Hugging Face Hub. "
            "Network issues or access restrictions may be the cause."
        ),
        context={},
        actionable_steps=[
            ActionableStep(
                description="Check network connectivity",
                action_type="command",
                command="curl -I https://huggingface.co",
            ),
            ActionableStep(
                description="Use a mirror/proxy (for restricted regions)",
                action_type="config_change",
                config_key="HF_ENDPOINT",
                config_value="https://hf-mirror.com",
            ),
        ],
    ),
}


def create_user_error(error_code: str, **context) -> UserFriendlyError:
    """Create a user-friendly error from registry with context substitution."""

    if error_code not in ERROR_REGISTRY:
        return UserFriendlyError(
            error_code="unknown",
            category=ErrorCategory.TRAINING,
            severity=ErrorSeverity.ERROR,
            title="Training Error",
            summary_template="An unexpected error occurred: {details}",
            context={"details": error_code},
        )

    template = ERROR_REGISTRY[error_code]

    def substitute(value, ctx):
        if isinstance(value, str):
            try:
                return value.format(**ctx)
            except KeyError:
                return value
        elif isinstance(value, list):
            return [substitute(item, ctx) for item in value]
        elif isinstance(value, ActionableStep):
            return ActionableStep(
                description=substitute(value.description, ctx),
                action_type=value.action_type,
                command=substitute(value.command or "", ctx) if value.command else None,
                config_key=value.config_key,
                config_value=substitute(str(value.config_value or ""), ctx)
                if value.config_value
                else None,
            )
        return value

    error = UserFriendlyError(
        error_code=template.error_code,
        category=template.category,
        severity=template.severity,
        title=template.title,
        summary_template=template.summary_template,
        context={**template.context, **context},
        actionable_steps=[substitute(s, context) for s in template.actionable_steps],
        documentation_url=template.documentation_url,
    )

    return error


def classify_exception(error: Exception) -> str:
    """Map Python exceptions to error codes."""

    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    if "cuda" in error_str and ("memory" in error_str or "oom" in error_type):
        return "cuda_oom"

    if "filenotfound" in error_type or "no such file" in error_str:
        return "dataset_not_found"

    if "connection" in error_str or "timeout" in error_type:
        return "model_download_failed"

    if "key" in error_str and ("not found" in error_str or "not in" in error_str):
        return "missing_target_column"

    if "parsing" in error_str or "tokenize" in error_str:
        return "invalid_csv_format"

    if "rejected" in error_str and ("missing" in error_str or "required" in error_str):
        return "dpo_missing_rejected"

    if "reward" in error_str and ("missing" in error_str or "required" in error_str):
        return "grpo_no_reward"

    return "unknown_error"


def format_exception_for_user(error: Exception, context: Optional[dict] = None) -> str:
    """Format any exception as a user-friendly message."""

    error_code = classify_exception(error)
    ctx = context or {}
    ctx.setdefault("error_details", str(error))

    user_error = create_user_error(error_code, **ctx)
    return str(user_error)


def handle_cuda_oom_error(
    original_error: Exception,
    current_batch_size: int,
    gpu_memory_gb: float,
) -> UserFriendlyError:
    """Special handler for CUDA OOM with smart recovery suggestions."""

    suggested_batch = max(1, current_batch_size // 2)
    attempted_mb = (
        str(original_error).split()[-1] if hasattr(original_error, "__str__") else "unknown"
    )

    return create_user_error(
        "cuda_oom",
        attempted_mb=attempted_mb,
        free_mb=f"{gpu_memory_gb * 1024:.0f}",
        suggested_batch=suggested_batch,
    )


AUTOML_ERRORS = {
    "missing_target": """
The target column '{column}' contains {count} missing values ({percent:.1f}%).

To fix this issue:
1. Open your CSV file
2. Either remove rows with missing target values, or
3. Fill missing values with an appropriate default

Example pandas code:
    df = df.dropna(subset=['{column}'])
""",
    "invalid_types": """
Column '{column}' contains inconsistent data types.

Found {count} values that don't match the expected type '{expected_type}'.
First problematic value: '{sample_value}' at row {row_number}

To fix this issue:
1. Open your CSV file and inspect column '{column}'
2. Ensure all values are of type '{expected_type}'
3. Convert or remove incompatible values

Example pandas code:
    df['{column}'] = pd.to_numeric(df['{column}'], errors='coerce')
    df = df.dropna(subset=['{column}'])
""",
    "file_not_found": """
File not found: '{filepath}'

The specified file does not exist at the given path.

To fix this issue:
1. Verify the file path is correct
2. Check if the file has been moved or renamed
3. Ensure you have read permissions for the directory

Example:
    import os
    print(os.path.exists('{filepath}'))  # Should return True
""",
    "invalid_csv": """
CSV parsing error: {error_message}

The file '{filepath}' could not be parsed as a valid CSV.

To fix this issue:
1. Open the file in a text editor to inspect its structure
2. Check for mismatched quotes, delimiters, or line endings
3. Ensure the file is saved with proper CSV formatting

Common issues:
- Mixed quote styles (use consistent single or double quotes)
- Incorrect delimiter (expected comma, found: {delimiter})
- Broken rows with different column counts
""",
    "missing_features": """
No feature columns found in the dataset.

After excluding the target column '{target_column}', no valid feature columns remain.
This typically occurs when:
- The dataset only contains the target column
- All non-target columns are empty or invalid

To fix this issue:
1. Verify your CSV contains multiple columns
2. Ensure feature columns have valid data types (numeric or categorical)
3. Check for empty or completely null columns

Example pandas code:
    print(df.columns.tolist())  # List all available columns
    print(df.drop(columns=['{target_column}']).dtypes)  # Check feature types
""",
    "too_few_samples": """
Insufficient training data: {count} samples found.

Training requires at least {minimum} samples for reliable results.
Your dataset has {count} rows, which is below the minimum threshold.

To fix this issue:
1. Collect more data for your dataset
2. Reduce model complexity if limited data is unavoidable
3. Consider using cross-validation with fewer folds

Current dataset size: {count} rows
Minimum required: {minimum} rows
Recommended for robust training: {recommended} rows

Example calculation:
    import math
    n_samples = len(df)
    min_required = max(100, int(len(df.columns) * 10))
""",
    "encoding_error": """
File encoding error: {error_message}

The file '{filepath}' could not be read with the detected encoding.

Detected encoding: {detected_encoding}
Expected encoding: utf-8

To fix this issue:
1. Open the file in a text editor and save with UTF-8 encoding
2. Or specify the correct encoding when reading

Example pandas code:
    df = pd.read_csv('{filepath}', encoding='latin-1')  # Try alternative encodings
    df = pd.read_csv('{filepath}', encoding='utf-8-sig')  # For UTF-8 with BOM
    df = pd.read_csv('{filepath}', encoding='cp1252')  # Windows-1252
""",
}


def format_error(error_key: str, **kwargs) -> str:
    """
    Format an error message from AUTOML_ERRORS with provided parameters.

    Args:
        error_key: Key identifying the error type in AUTOML_ERRORS dictionary
        **kwargs: Parameters to substitute into the error message template

    Returns:
        Formatted error message string with placeholders replaced by values

    Raises:
        KeyError: If error_key is not found in AUTOML_ERRORS
    """
    template = AUTOML_ERRORS[error_key]
    return template.format(**kwargs)
