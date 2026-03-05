from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sklearn.ensemble import VotingClassifier, VotingRegressor

from src.ml.auto_ensemble import create_ensemble
from src.ml.data_validator import detect_task_type, validate_csv
from src.job_queue.job_manager import JobManager


@pytest.fixture
def temp_db(tmp_path: Path) -> Generator[str, None, None]:
    db_path = tmp_path / "test_jobs.db"
    yield str(db_path)


@pytest.fixture
def temp_csv(tmp_path: Path) -> Generator[str, None, None]:
    csv_content = "feature_a,feature_b,target\n1.0,2.0,class_a\n3.0,4.0,class_b\n5.0,6.0,class_a\n"
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(csv_content)
    yield str(csv_path)


@pytest.fixture
def classification_target() -> pd.Series:
    return pd.Series(["cat", "dog", "cat", "bird", "dog"])


@pytest.fixture
def regression_target() -> pd.Series:
    return pd.Series(
        [
            1.5,
            2.7,
            3.2,
            4.8,
            5.1,
            6.9,
            7.3,
            8.2,
            9.5,
            10.1,
            11.4,
            12.8,
            13.6,
            14.9,
            15.2,
            16.7,
            17.3,
            18.8,
            19.1,
            20.5,
        ]
    )


def test_detect_classification(classification_target: pd.Series) -> None:
    """
    Test that string/object target is detected as classification task.

    Args:
        classification_target: Series with categorical string values
    """
    result = detect_task_type(classification_target)
    assert result == "classification"


def test_detect_regression(regression_target: pd.Series) -> None:
    """
    Test that continuous numeric target with many unique values is detected as regression.

    Args:
        regression_target: Series with continuous numeric values
    """
    result = detect_task_type(regression_target)
    assert result == "regression"


def test_validate_csv_success(temp_csv: str) -> None:
    """
    Test that valid CSV file passes validation.

    Args:
        temp_csv: Path to temporary valid CSV file
    """
    result = validate_csv(temp_csv)
    assert result["valid"] is True
    assert "columns" in result
    assert len(result["columns"]) == 3


def test_validate_missing_file(tmp_path: Path) -> None:
    """
    Test that missing file returns appropriate error message.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    missing_file = str(tmp_path / "nonexistent.csv")
    result = validate_csv(missing_file)
    assert result["valid"] is False
    assert "not found" in result["message"].lower() or "does not exist" in result["message"].lower()
    assert result["columns"] == []


def test_create_classification_ensemble() -> None:
    """
    Test that classification task type returns VotingClassifier.
    """
    model = create_ensemble("classification")
    assert isinstance(model, VotingClassifier)
    assert len(model.estimators) == 3


def test_create_regression_ensemble() -> None:
    """
    Test that regression task type returns VotingRegressor.
    """
    model = create_ensemble("regression")
    assert isinstance(model, VotingRegressor)
    assert len(model.estimators) == 3


def test_submit_job(temp_db: str) -> None:
    """
    Test that submitting a job returns a valid job ID.

    Args:
        temp_db: Path to temporary database file
    """
    manager = JobManager(db_path=temp_db)
    job_id = manager.submit_job("ml_training", {"data_path": "/tmp/data.csv"})
    assert job_id is not None
    assert len(job_id) == 36


def test_get_job(temp_db: str) -> None:
    """
    Test that a submitted job can be retrieved.

    Args:
        temp_db: Path to temporary database file
    """
    manager = JobManager(db_path=temp_db)
    job_id = manager.submit_job("ml_training", {"data_path": "/tmp/data.csv"})
    job = manager.get_job(job_id)
    assert job is not None
    assert job.id == job_id
    assert job.job_type == "ml_training"
