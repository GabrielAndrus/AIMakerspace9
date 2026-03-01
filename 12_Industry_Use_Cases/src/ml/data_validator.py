from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.error_handling import format_error


def validate_csv(filepath: str, target_column: Optional[str] = None) -> dict:
    """
    Validate a CSV file for ML training.

    Args:
        filepath: Path to the CSV file
        target_column: Optional target column name to validate

    Returns:
        Dictionary with keys:
            - 'valid' (bool): Whether validation passed
            - 'message' (str): Success or error message
            - 'columns' (list): List of column names if valid, empty otherwise

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If CSV cannot be parsed or target validation fails
    """
    path = Path(filepath)

    if not path.exists():
        return {
            "valid": False,
            "message": format_error("file_not_found", filepath=filepath),
            "columns": [],
        }

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {
            "valid": False,
            "message": format_error(
                "invalid_csv",
                error_message=str(e),
                filepath=filepath,
                delimiter="unknown",
            ),
            "columns": [],
        }

    if target_column is not None:
        if target_column not in df.columns:
            return {
                "valid": False,
                "message": f"Target column '{target_column}' not found in CSV. Available columns: {df.columns.tolist()}",
                "columns": [],
            }

        missing_count = df[target_column].isna().sum()
        if missing_count > 0:
            total_count = len(df)
            missing_pct = (missing_count / total_count) * 100
            return {
                "valid": False,
                "message": format_error(
                    "missing_target",
                    column=target_column,
                    count=missing_count,
                    percent=missing_pct,
                ),
                "columns": [],
            }

    return {
        "valid": True,
        "message": f"Successfully validated CSV with {len(df)} rows and {len(df.columns)} columns.",
        "columns": df.columns.tolist(),
    }


def detect_task_type(y: pd.Series) -> str:
    """
    Detect whether the target variable is for classification or regression.

    Args:
        y: Target variable as a pandas Series

    Returns:
        'classification' or 'regression'
    """
    if y.dtype == "object":
        return "classification"

    n_unique = y.nunique()
    n_total = len(y)

    if n_unique < 10:
        return "classification"

    unique_ratio = n_unique / n_total
    if unique_ratio < 0.05:
        return "classification"

    return "regression"


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Categorize DataFrame columns by data type for ML processing.

    Args:
        df: Input DataFrame to analyze

    Returns:
        Dictionary mapping column names to types:
            - 'numeric': Integer or float columns
            - 'categorical': Object/string columns with <= 50 unique values
            - 'text': Object/string columns with > 50 unique values
    """
    column_types = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif df[col].dtype == "object":
            n_unique = df[col].nunique()
            if n_unique <= 50:
                column_types[col] = "categorical"
            else:
                column_types[col] = "text"
        else:
            column_types[col] = "categorical"

    return column_types
