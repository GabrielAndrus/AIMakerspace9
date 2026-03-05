"""Tests for Metaflow runner module.

This module tests the functionality of running flows programmatically,
retrieving artifacts, and polling flow status.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# Set environment variable before any imports
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"


class TestMLTrainingFlow:
    """Test ML training flow via Metaflow."""

    def test_run_ml_training_flow_simple(self):
        """Test running ML training flow with simple data."""
        from src.flows.runner import get_flow_artifacts, run_ml_training_flow

        # Create a simple test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "feature1": [1, 2, 3, 4, 5],
                    "feature2": [10, 20, 30, 40, 50],
                    "target": [0, 1, 0, 1, 0],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Run the flow
            run_id = run_ml_training_flow(
                data_path=csv_path,
                target_column="target",
                wait_for_completion=True,
            )

            # Verify run_id format
            assert run_id is not None
            assert "MLTrainingFlow" in run_id

            # Get artifacts
            artifacts = get_flow_artifacts(run_id)

            # Verify required artifacts
            assert "model_path" in artifacts
            assert "task_type" in artifacts
            assert "metrics" in artifacts
            assert artifacts["model_path"] is not None

        finally:
            # Cleanup
            os.unlink(csv_path)

    def test_run_ml_training_flow_regression(self):
        """Test ML training flow with regression data."""
        from src.flows.runner import get_flow_artifacts, run_ml_training_flow

        # Create regression test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "x1": [1, 2, 3, 4, 5],
                    "x2": [10, 20, 30, 40, 50],
                    "y": [100, 200, 300, 400, 500],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Run the flow
            run_id = run_ml_training_flow(
                data_path=csv_path,
                target_column="y",
                wait_for_completion=True,
            )

            # Get artifacts
            artifacts = get_flow_artifacts(run_id)

            # Verify regression task type
            assert "task_type" in artifacts
            assert artifacts["task_type"] == "regression"

        finally:
            os.unlink(csv_path)

    def test_get_flow_artifacts_invalid_run_id(self):
        """Test getting artifacts with invalid run ID."""
        from src.flows.runner import get_flow_artifacts

        # Test with completely invalid format
        with pytest.raises(ValueError):
            get_flow_artifacts("invalid")

    def test_poll_flow_status(self):
        """Test polling flow status."""
        from src.flows.runner import poll_flow_status, run_ml_training_flow

        # Create test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "feature1": [1, 2, 3],
                    "target": [0, 1, 0],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Run flow
            run_id = run_ml_training_flow(
                data_path=csv_path,
                target_column="target",
            )

            # Poll status
            status = poll_flow_status(run_id)

            # Verify status structure
            assert "state" in status
            assert "progress" in status
            assert "current_step" in status
            assert 0.0 <= status["progress"] <= 1.0

        finally:
            os.unlink(csv_path)

    def test_poll_flow_status_not_found(self):
        """Test polling status for non-existent run."""
        from src.flows.runner import poll_flow_status

        status = poll_flow_status("MLTrainingFlow/99999")
        assert status["state"] == "not_found"


class TestLLMTrainingFlow:
    """Test LLM training flow via Metaflow."""

    def test_run_llm_training_flow_sft(self):
        """Test running LLM training flow with SFT method."""
        from src.flows.runner import get_flow_artifacts, run_llm_training_flow

        # Create a simple test TXT file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a sample text for training.\n")
            f.write("Another line of training data.\n")
            txt_path = f.name

        try:
            # Run the flow
            run_id = run_llm_training_flow(
                data_path=txt_path,
                training_method="SFT",
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                epochs=1,  # Use minimal epochs for testing
                learning_rate=2e-4,
                wait_for_completion=False,  # Don't wait for long training
            )

            # Verify run_id format
            assert run_id is not None
            assert "LLMTrainingFlow" in run_id

        finally:
            os.unlink(txt_path)

    def test_run_llm_training_flow_invalid_method(self):
        """Test LLM training flow with invalid method."""
        from src.flows.runner import run_llm_training_flow

        # Create test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Sample text\n")
            txt_path = f.name

        try:
            # This should fail due to invalid method
            run_id = run_llm_training_flow(
                data_path=txt_path,
                training_method="INVALID_METHOD",
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                epochs=1,
                wait_for_completion=False,
            )

        finally:
            os.unlink(txt_path)


class TestFlowCards:
    """Test Metaflow card retrieval."""

    def test_get_flow_cards(self):
        """Test retrieving cards from a flow run."""
        from src.flows.runner import get_flow_cards, run_ml_training_flow

        # Create test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "feature1": [1, 2, 3],
                    "target": [0, 1, 0],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Run flow
            run_id = run_ml_training_flow(
                data_path=csv_path,
                target_column="target",
            )

            # Get cards
            cards = get_flow_cards(run_id)

            # Verify cards structure
            assert isinstance(cards, list)
            for card in cards:
                assert "type" in card
                assert "step" in card

        finally:
            os.unlink(csv_path)


class TestEnvironmentSetup:
    """Test environment configuration."""

    def test_metadata_environment_variable(self):
        """Verify METAFLOW_DEFAULT_METADATA is set correctly."""
        assert os.environ.get("METAFLOW_DEFAULT_METADATA") == "local"

    def test_runner_module_imports(self):
        """Test that runner module can be imported."""
        from src.flows import runner

        # Verify key functions exist
        assert hasattr(runner, "run_ml_training_flow")
        assert hasattr(runner, "run_llm_training_flow")
        assert hasattr(runner, "get_flow_artifacts")
        assert hasattr(runner, "poll_flow_status")
        assert hasattr(runner, "get_flow_cards")


class TestIntegration:
    """Integration tests for the complete Metaflow workflow."""

    def test_end_to_end_ml_training(self):
        """Test complete ML training workflow through Metaflow."""
        from src.flows.runner import get_flow_artifacts, run_ml_training_flow

        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "age": [25, 30, 35, 40, 45],
                    "income": [50000, 60000, 70000, 80000, 90000],
                    "risk": ["low", "medium", "low", "high", "medium"],
                }
            )
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Run training
            run_id = run_ml_training_flow(
                data_path=csv_path,
                target_column="risk",
            )

            # Verify flow completed
            status = poll_flow_status(run_id)
            assert status["state"] in ["completed", "running"]

            # Get results
            artifacts = get_flow_artifacts(run_id)
            assert all(k in artifacts for k in ["model_path", "task_type", "metrics"])

        finally:
            os.unlink(csv_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
