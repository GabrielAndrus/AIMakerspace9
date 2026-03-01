import zipfile
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, filepath: str) -> str:
    """
    Save an sklearn ensemble model to disk using joblib.

    Args:
        model: Trained sklearn model (e.g., VotingClassifier, VotingRegressor)
        filepath: Path where the model should be saved

    Returns:
        Absolute path to the saved model file

    Raises:
        ValueError: If filepath is empty or None
        OSError: If the directory cannot be created or written to
    """
    if not filepath:
        raise ValueError("filepath cannot be empty")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    return str(path.resolve())


def load_model(filepath: str) -> Any:
    """
    Load a saved sklearn model from disk.

    Args:
        filepath: Path to the saved model file

    Returns:
        The loaded sklearn model

    Raises:
        FileNotFoundError: If the model file does not exist
        ValueError: If filepath is empty or None
    """
    if not filepath:
        raise ValueError("filepath cannot be empty")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    return joblib.load(path)


def save_lora_adapter(model, output_dir: str) -> str:
    """
    Save a LoRA adapter to disk using safetensors format.

    Args:
        model: PEFT model with LoRA adapters to save
        output_dir: Directory where the adapter should be saved

    Returns:
        Absolute path to the saved adapter directory

    Raises:
        ValueError: If output_dir is empty or None
        OSError: If the directory cannot be created or written to
    """
    if not output_dir:
        raise ValueError("output_dir cannot be empty")

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    return str(path.resolve())


def create_model_archive(model_path: str, output_path: str) -> str:
    """
    Create a zip archive of a model directory for download.

    Args:
        model_path: Path to the model directory to archive
        output_path: Path where the zip archive should be created

    Returns:
        Absolute path to the created zip archive

    Raises:
        FileNotFoundError: If model_path does not exist
        ValueError: If any path is empty or None
        OSError: If the archive cannot be created
    """
    if not model_path:
        raise ValueError("model_path cannot be empty")
    if not output_path:
        raise ValueError("output_path cannot be empty")

    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source)
                archive.write(file_path, arcname)

    return str(output.resolve())
