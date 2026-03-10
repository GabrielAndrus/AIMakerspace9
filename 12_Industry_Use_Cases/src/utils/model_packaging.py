"""Model Serialization and Packaging with Full Metadata.

Provides standardized saving/loading for sklearn ensembles and LoRA adapters,
ensuring all metadata needed for inference and reproducibility is preserved.
"""

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import sklearn


@dataclass
class SklearnModelMetadata:
    """Metadata for a saved sklearn model."""

    model_name: str
    version: str
    created_at: str

    sklearn_version: str
    numpy_version: str
    python_version: str

    task_type: str
    target_column: str
    feature_columns: list[str]

    metrics: dict[str, float] = field(default_factory=dict)

    pipeline_steps: list[dict] = field(default_factory=list)

    categorical_columns: list[str] = field(default_factory=list)

    file_hash: str = ""
    file_size_mb: float = 0.0

    description: str = ""


@dataclass
class LoRAModelMetadata:
    """Metadata for a saved LoRA adapter."""

    model_name: str
    version: str
    created_at: str

    base_model: str
    base_model_revision: Optional[str] = None

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=list)

    training_method: str = "sft"
    num_epochs: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    dataset_size: Optional[int] = None

    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None

    framework_versions: dict[str, str] = field(default_factory=dict)

    file_hash: str = ""


def save_sklearn_model_package(
    pipeline,
    output_dir: str,
    model_name: str,
    version: str = "1.0.0",
    task_type: str = "classification",
    target_column: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    metrics: Optional[dict] = None,
    description: str = "",
) -> str:
    """Save sklearn pipeline as a versioned, validated package.

    Creates directory structure:
    output_dir/
    └── model_name/
        └── version/
            ├── model.joblib
            ├── metadata.json
            └── requirements.txt

    Args:
        pipeline: Trained sklearn Pipeline or model
        output_dir: Base directory for saving
        model_name: Name for the model
        version: Version string (e.g., "1.0.0")
        task_type: "classification" or "regression"
        target_column: Name of the target column
        feature_columns: List of feature column names
        metrics: Dict of metric name -> value
        description: Human-readable description

    Returns:
        Path to saved model directory
    """

    save_path = Path(output_dir) / model_name / version
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / "model.joblib"
    joblib.dump(pipeline, model_file, compress=3)

    file_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()[:16]
    file_size_mb = model_file.stat().st_size / (1024 * 1024)

    pipeline_steps = []
    for name, step in getattr(pipeline, "steps", []):
        params = {}
        for key, val in step.get_params().items():
            if isinstance(val, (int, float, str, bool, type(None))):
                params[key] = val
        pipeline_steps.append(
            {
                "name": name,
                "type": type(step).__name__,
                "module": type(step).__module__,
                "params": params,
            }
        )

    metadata = SklearnModelMetadata(
        model_name=model_name,
        version=version,
        created_at=datetime.utcnow().isoformat(),
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        python_version=sys.version.split()[0],
        task_type=task_type,
        target_column=target_column,
        feature_columns=feature_columns or [],
        categorical_columns=categorical_columns or [],
        metrics=metrics or {},
        pipeline_steps=pipeline_steps,
        file_hash=file_hash,
        file_size_mb=file_size_mb,
        description=description,
    )

    (save_path / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))

    requirements = f"""# Model: {model_name} v{version}
# Generated: {metadata.created_at}

scikit-learn=={sklearn.__version__}
numpy=={np.__version__}
joblib>=1.3.0
pandas>=2.0.0
"""
    (save_path / "requirements.txt").write_text(requirements)

    return str(save_path)


def load_sklearn_model_package(
    model_dir: str,
    version: Optional[str] = None,
) -> tuple[Any, SklearnModelMetadata]:
    """Load sklearn model with metadata and integrity verification.

    Args:
        model_dir: Path to model directory (containing version subdirs)
        version: Specific version to load (latest if None)

    Returns:
        Tuple of (pipeline, metadata)

    Raises:
        FileNotFoundError: If model or metadata not found
        ValueError: If hash verification fails
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if version:
        load_path = model_path / version
    else:
        versions = sorted(
            [d.name for d in model_path.iterdir() if d.is_dir()],
            key=lambda v: [int(x) for x in v.split(".")],
        )
        if not versions:
            raise ValueError(f"No versions found in {model_dir}")
        load_path = model_path / versions[-1]

    metadata_file = load_path / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Missing metadata.json in {load_path}\n"
            "This model was saved without packaging. "
            "Use save_sklearn_model_package() for full metadata."
        )

    metadata = SklearnModelMetadata(**json.loads(metadata_file.read_text()))

    model_file = load_path / "model.joblib"
    current_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()[:16]

    if metadata.file_hash and current_hash != metadata.file_hash:
        raise ValueError(
            f"Model file hash mismatch!\n"
            f"Expected: {metadata.file_hash}\n"
            f"Actual: {current_hash}\n\n"
            "The model file may be corrupted or modified. Do not use this model."
        )

    pipeline = joblib.load(model_file)

    return pipeline, metadata


def save_lora_adapter_package(
    model,
    tokenizer,
    output_dir: str,
    model_name: str,
    version: str = "1.0.0",
    base_model: str = "",
    lora_config=None,
    training_args: Optional[dict] = None,
    metrics: Optional[dict] = None,
    adapter_path: Optional[str] = None,
) -> str:
    """Save LoRA adapter with complete metadata for inference.

    Creates directory structure:
    output_dir/
    └── model_name/
        └── version/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── metadata.json

    Returns:
        Path to saved adapter directory
    """
    import torch
    import transformers
    import peft
    import shutil

    save_path = Path(output_dir) / model_name / version
    save_path.mkdir(parents=True, exist_ok=True)

    if adapter_path and model is None:
        src_path = Path(adapter_path)
        for file in src_path.glob("*"):
            if file.is_file() and not file.name.startswith("."):
                shutil.copy2(file, save_path / file.name)
    else:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    adapter_file = save_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = save_path / "adapter_model.bin"

    file_hash = ""
    if adapter_file.exists():
        file_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest()[:16]

    metadata = LoRAModelMetadata(
        model_name=model_name,
        version=version,
        created_at=datetime.utcnow().isoformat(),
        base_model=base_model,
        lora_rank=getattr(lora_config, "r", 8) if lora_config else 8,
        lora_alpha=getattr(lora_config, "lora_alpha", 16) if lora_config else 16,
        lora_dropout=getattr(lora_config, "lora_dropout", 0.1) if lora_config else 0.1,
        target_modules=list(lora_config.target_modules)
        if lora_config and hasattr(lora_config, "target_modules")
        else [],
        training_method=training_args.get("method", "sft") if training_args else "sft",
        num_epochs=training_args.get("num_epochs") if training_args else None,
        learning_rate=training_args.get("learning_rate") if training_args else None,
        batch_size=training_args.get("batch_size") if training_args else None,
        train_loss=metrics.get("train_loss") if metrics else None,
        eval_loss=metrics.get("eval_loss") if metrics else None,
        framework_versions={
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        file_hash=file_hash,
    )

    (save_path / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))

    return str(save_path)


def validate_model_package(model_dir: str) -> dict:
    """Validate a model package for completeness.

    Returns:
        {
            "valid": bool,
            "model_type": "sklearn" or "lora",
            "issues": list[str],
            "warnings": list[str],
            "metadata": dict | None,
        }
    """
    path = Path(model_dir)

    result = {
        "valid": True,
        "model_type": "auto",
        "issues": [],
        "warnings": [],
        "metadata": None,
    }

    if not path.exists():
        result["valid"] = False
        result["issues"].append(f"Directory not found: {model_dir}")
        return result

    if (path / "adapter_config.json").exists():
        result["model_type"] = "lora"
    elif (path / "model.joblib").exists() or (path / "model.pkl").exists():
        result["model_type"] = "sklearn"
    else:
        result["valid"] = False
        result["issues"].append(
            "Could not detect model type. "
            "Expected adapter_config.json (LoRA) or model.joblib (sklearn)"
        )
        return result

    if result["model_type"] == "sklearn":
        required = ["model.joblib", "metadata.json"]

        for f in required:
            if not (path / f).exists():
                result["valid"] = False
                result["issues"].append(f"Missing: {f}")

        if (path / "metadata.json").exists():
            try:
                metadata = json.loads((path / "metadata.json").read_text())
                result["metadata"] = metadata

                if (path / "model.joblib").exists() and metadata.get("file_hash"):
                    actual_hash = hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()[
                        :16
                    ]

                    if actual_hash != metadata.get("file_hash"):
                        result["valid"] = False
                        result["issues"].append("File hash mismatch - model may be corrupted")

                current_sklearn = sklearn.__version__
                saved_sklearn = metadata.get("sklearn_version", "")

                if saved_sklearn and saved_sklearn.split(".")[0] != current_sklearn.split(".")[0]:
                    result["warnings"].append(
                        f"sklearn version mismatch: saved={saved_sklearn}, current={current_sklearn}"
                    )
            except json.JSONDecodeError:
                result["warnings"].append("Could not parse metadata.json")

    elif result["model_type"] == "lora":
        if (
            not (path / "adapter_model.safetensors").exists()
            and not (path / "adapter_model.bin").exists()
        ):
            result["valid"] = False
            result["issues"].append("Missing adapter weights file")

        if not (path / "adapter_config.json").exists():
            result["valid"] = False
            result["issues"].append("Missing adapter_config.json")

    return result


def create_downloadable_zip(model_dir: str, output_path: Optional[str] = None) -> str:
    """Create a downloadable zip of the model package."""
    import shutil

    model_path = Path(model_dir)

    if not output_path:
        output_path = model_path.name

    # make_archive appends .zip automatically
    archive_path = shutil.make_archive(output_path, "zip", model_path.parent, model_path.name)

    return archive_path
