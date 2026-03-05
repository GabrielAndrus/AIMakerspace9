import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any


class DatasetProfiler:
    """Analyze uploaded datasets and generate natural language descriptions."""

    def __init__(self):
        pass

    def profile(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Profile a dataset and return statistics."""

        y = df[target_column]

        profile = {
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,
            "target_column": target_column,
            "problem_type": self._detect_problem_type(y),
            "class_distribution": self._get_class_distribution(y),
            "feature_types": self._analyze_feature_types(df),
            "missing_values": self._check_missing_values(df),
            "numeric_summary": self._get_numeric_summary(df),
            "categorical_summary": self._get_categorical_summary(df),
        }

        return profile

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect if classification or regression."""
        if y.dtype == "object" or y.dtype.name == "category":
            unique_vals = y.nunique()
            if unique_vals <= 10:
                return "classification"
        if pd.api.types.is_numeric_dtype(y):
            return "regression"
        return "classification"

    def _get_class_distribution(self, y: pd.Series) -> dict:
        """Get class distribution for classification."""
        if y.dtype == "object" or y.dtype.name == "category":
            dist = y.value_counts()
            return {
                "counts": dist.to_dict(),
                "imbalance_ratio": dist.max() / dist.min() if len(dist) > 1 else 1.0,
            }
        return {}

    def _analyze_feature_types(self, df: pd.DataFrame) -> dict:
        """Analyze feature types."""
        result = {"numeric": [], "categorical": [], "other": []}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                result["numeric"].append(col)
            elif df[col].dtype == "object":
                result["categorical"].append(col)
            else:
                result["other"].append(col)
        return result

    def _check_missing_values(self, dict) -> dict:
        """Check for missing values."""
        return {col: int(df[col].isna().sum()) for col in df.columns}

    def _get_numeric_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics for numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        return df[numeric_cols].describe().to_dict()

    def _get_categorical_summary(self, df: pd.DataFrame) -> dict:
        """Get summary for categorical features."""
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) == 0:
            return {}
        return {col: df[col].nunique() for col in cat_cols}

    def generate_query(self, profile: dict) -> str:
        """Generate a natural language query from the profile."""

        query_parts = []

        n_rows = profile.get("n_rows", 0)
        n_features = profile.get("n_features", 0)
        problem_type = profile.get("problem_type", "classification")

        query_parts.append(
            f"Dataset with {n_rows} rows and {n_features} features for {problem_type}"
        )

        if problem_type == "classification":
            class_dist = profile.get("class_distribution", {})
            if "imbalance_ratio" in class_dist and class_dist["imbalance_ratio"] > 3:
                query_parts.append(
                    f"highly imbalanced classes (ratio: {class_dist['imbalance_ratio']:.1f})"
                )

        feat_types = profile.get("feature_types", {})
        if feat_types.get("categorical"):
            query_parts.append(f"with {len(feat_types['categorical'])} categorical features")

        missing = profile.get("missing_values", {})
        total_missing = sum(missing.values())
        if total_missing > 0:
            query_parts.append(f"has {total_missing} missing values")

        return ". ".join(query_parts) + "."
