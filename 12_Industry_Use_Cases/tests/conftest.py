"""Shared pytest fixtures for Agentic AutoML Platform."""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_classification(test_data_dir):
    """Sample classification CSV for testing."""
    import pandas as pd

    csv_path = test_data_dir / "sample_classification.csv"

    if not csv_path.exists():
        df = pd.DataFrame(
            {
                "feature_1": [1.5, 2.3, 0.8, 3.1, 1.9] * 20,
                "feature_2": [2.1, 1.9, 3.2, 0.5, 2.8] * 20,
                "feature_3": [0.7, 1.1, 0.3, 2.2, 0.9] * 20,
                "target": ["A", "B", "A", "C", "B"] * 20,
            }
        )
        df.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def sample_csv_regression(test_data_dir):
    """Sample regression CSV for testing."""
    import pandas as pd

    csv_path = test_data_dir / "sample_regression.csv"

    if not csv_path.exists():
        df = pd.DataFrame(
            {
                "feature_1": [1.5, 2.3, 0.8, 3.1, 1.9] * 20,
                "feature_2": [2.1, 1.9, 3.2, 0.5, 2.8] * 20,
                "target": [10.5, 15.2, 8.1, 22.3, 12.7] * 20,
            }
        )
        df.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_gpu_available():
    """Fixture that can be used to skip tests if no GPU."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def sklearn_classification_model(sample_csv_classification):
    """Pre-trained classification model for testing."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(sample_csv_classification)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)),
        ],
        voting="soft",
    )

    ensemble.fit(X_train, y_train)
    return ensemble, list(X.columns)


@pytest.fixture
def mock_job_manager():
    """Mock job manager for testing without SQLite."""
    from unittest.mock import MagicMock

    manager = MagicMock()
    manager.submit_job.return_value = "test-job-123"
    manager.get_job.return_value = MagicMock(
        id="test-job-123",
        status=MagicMock(value="completed"),
        progress=1.0,
        result_path="/tmp/model.joblib",
    )
    return manager
