"""Tests for data validation module."""

import pytest
import pandas as pd
from pathlib import Path


class TestDetectTaskType:
    """Tests for task type detection."""

    def test_classification_string_target(self):
        from src.ml.data_validator import detect_task_type

        y = pd.Series(["A", "B", "A", "C"] * 25)
        assert detect_task_type(y) == "classification"

    def test_classification_few_unique_values(self):
        from src.ml.data_validator import detect_task_type

        y = pd.Series([0, 1, 0, 1] * 25)
        assert detect_task_type(y) == "classification"

    def test_regression_many_unique_values(self):
        from src.ml.data_validator import detect_task_type

        y = pd.Series([i * 0.5 for i in range(100)])
        assert detect_task_type(y) == "regression"


class TestValidateCSV:
    """Tests for CSV validation."""

    def test_valid_csv(self, sample_csv_classification):
        from src.ml.data_validator import validate_csv

        result = validate_csv(sample_csv_classification)

        assert result["valid"] is True
        assert "columns" in result
        assert len(result["columns"]) == 4

    def test_missing_target_column(self, sample_csv_classification):
        from src.ml.data_validator import validate_csv

        result = validate_csv(sample_csv_classification, target_column="nonexistent")

        assert result["valid"] is False
        assert "not found" in result["message"].lower()

    def test_file_not_found(self):
        from src.ml.data_validator import validate_csv

        result = validate_csv("/nonexistent/path/file.csv")

        assert result["valid"] is False
        assert "not found" in result["message"].lower()


class TestExcelAndTSVSupport:
    """Tests for Excel and TSV file support."""

    def test_detect_file_format_csv(self, sample_csv_classification):
        from src.ml.data_validator import detect_file_format

        fmt = detect_file_format(sample_csv_classification)
        assert fmt == "csv"

    def test_read_tsv_file(self, tmp_path):
        from src.ml.data_validator import read_data_file
        import pandas as pd

        tsv_path = Path(tmp_path) / "test.tsv"
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.to_csv(tsv_path, sep="\t", index=False)

        result = read_data_file(str(tsv_path))

        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]
