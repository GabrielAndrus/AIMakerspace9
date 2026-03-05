"""Tests for model packaging with metadata."""

import pytest
import json
from pathlib import Path


class TestSklearnModelPackaging:
    """Tests for sklearn model serialization."""

    def test_save_creates_directory_structure(self, sklearn_classification_model, temp_output_dir):
        from src.utils.model_packaging import save_sklearn_model_package

        model, features = sklearn_classification_model

        saved_path = save_sklearn_model_package(
            pipeline=model,
            output_dir=temp_output_dir,
            model_name="test_classifier",
            version="1.0.0",
            task_type="classification",
            target_column="target",
            feature_columns=features,
            metrics={"accuracy": 0.85},
        )

        assert Path(saved_path).exists()
        assert (Path(saved_path) / "model.joblib").exists()
        assert (Path(saved_path) / "metadata.json").exists()
        assert (Path(saved_path) / "requirements.txt").exists()

    def test_metadata_contains_versions(self, sklearn_classification_model, temp_output_dir):
        from src.utils.model_packaging import save_sklearn_model_package
        import sklearn
        import numpy

        model, features = sklearn_classification_model

        saved_path = save_sklearn_model_package(
            pipeline=model,
            output_dir=temp_output_dir,
            model_name="test_classifier",
            version="1.0.0",
            task_type="classification",
            target_column="target",
            feature_columns=features,
        )

        metadata = json.loads((Path(saved_path) / "metadata.json").read_text())

        assert metadata["sklearn_version"] == sklearn.__version__
        assert metadata["numpy_version"] == numpy.__version__

    def test_metadata_contains_metrics(self, sklearn_classification_model, temp_output_dir):
        from src.utils.model_packaging import save_sklearn_model_package

        model, features = sklearn_classification_model

        saved_path = save_sklearn_model_package(
            pipeline=model,
            output_dir=temp_output_dir,
            model_name="test_classifier",
            version="1.0.0",
            task_type="classification",
            target_column="target",
            feature_columns=features,
            metrics={"accuracy": 0.85, "f1": 0.82},
        )

        metadata = json.loads((Path(saved_path) / "metadata.json").read_text())

        assert metadata["metrics"]["accuracy"] == 0.85
        assert metadata["metrics"]["f1"] == 0.82

    def test_load_model_with_metadata(self, sklearn_classification_model, temp_output_dir):
        from src.utils.model_packaging import save_sklearn_model_package, load_sklearn_model_package

        model, features = sklearn_classification_model

        saved_path = save_sklearn_model_package(
            pipeline=model,
            output_dir=temp_output_dir,
            model_name="test_classifier",
            version="1.0.0",
            task_type="classification",
            target_column="target",
            feature_columns=features,
        )

        loaded_model, metadata = load_sklearn_model_package(saved_path)

        assert hasattr(loaded_model, "predict")
        assert metadata.task_type == "classification"
        assert metadata.target_column == "target"

    def test_hash_verification_failure(self, sklearn_classification_model, temp_output_dir):
        from src.utils.model_packaging import save_sklearn_model_package, load_sklearn_model_package

        model, features = sklearn_classification_model

        saved_path = save_sklearn_model_package(
            pipeline=model,
            output_dir=temp_output_dir,
            model_name="test_classifier",
            version="1.0.0",
            task_type="classification",
            target_column="target",
            feature_columns=features,
        )

        (Path(saved_path) / "model.joblib").write_bytes(b"corrupted")

        with pytest.raises(ValueError, match="hash mismatch"):
            load_sklearn_model_package(saved_path)


class TestValidateModelPackage:
    """Tests for package validation."""

    def test_validate_valid_sklearn_package(self, sklearn_classification_model, temp_output_dir):
        from src.utils.model_packaging import save_sklearn_model_package, validate_model_package

        model, features = sklearn_classification_model

        saved_path = save_sklearn_model_package(
            pipeline=model,
            output_dir=temp_output_dir,
            model_name="test_classifier",
            version="1.0.0",
            task_type="classification",
            target_column="target",
            feature_columns=features,
        )

        result = validate_model_package(saved_path)

        assert result["valid"] is True
        assert result["model_type"] == "sklearn"

    def test_validate_missing_metadata(self, temp_output_dir):
        from src.utils.model_packaging import validate_model_package
        import joblib

        pkg_dir = Path(temp_output_dir) / "incomplete" / "1.0.0"
        pkg_dir.mkdir(parents=True)

        joblib.dump({"model": "test"}, pkg_dir / "model.joblib")

        result = validate_model_package(str(pkg_dir))

        assert result["valid"] is False
        assert any("metadata" in issue.lower() for issue in result["issues"])
