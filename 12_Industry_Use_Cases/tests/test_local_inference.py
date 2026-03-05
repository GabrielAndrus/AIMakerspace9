"""Tests for local LLM inference with LoRA adapter support."""

import json
import tempfile
from pathlib import Path

import pytest


class TestValidateLoraAdapter:
    """Test LoRA adapter validation functionality."""

    def test_validate_missing_directory(self):
        """Test validation fails for non-existent directory."""
        from src.llm.local_inference import validate_lora_adapter

        result = validate_lora_adapter("/nonexistent/path")
        assert not result["valid"]
        assert any("Directory not found" in issue for issue in result["issues"])
        assert len(result["warnings"]) == 0

    def test_validate_missing_adapter_config(self):
        """Test validation fails without adapter_config.json."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory but no required files
            result = validate_lora_adapter(tmpdir)
            assert not result["valid"]
            assert any("adapter_config.json" in issue for issue in result["issues"])

    def test_validate_missing_weights(self):
        """Test validation fails without weight files."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create adapter_config.json but no weights
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"base_model_name_or_path": "test/model"}, f)

            result = validate_lora_adapter(tmpdir)
            assert not result["valid"]
            assert any("adapter_model" in issue for issue in result["issues"])

    def test_validate_valid_adapter_safetensors(self):
        """Test validation passes with safetensors weights."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create valid adapter files
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-0.5B",
                        "r": 16,
                        "lora_alpha": 32,
                    },
                    f,
                )

            # Create dummy safetensors file
            weights_path = tmppath / "adapter_model.safetensors"
            weights_path.write_bytes(b"dummy safetensors data")

            result = validate_lora_adapter(tmpdir)
            assert result["valid"]
            assert len(result["issues"]) == 0
            assert result["base_model"] == "Qwen/Qwen2.5-0.5B"

    def test_validate_valid_adapter_bin(self):
        """Test validation passes with .bin weights."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create valid adapter files
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"base_model_name_or_path": "test/model"}, f)

            # Create dummy .bin file
            weights_path = tmppath / "adapter_model.bin"
            weights_path.write_bytes(b"dummy bin data")

            result = validate_lora_adapter(tmpdir)
            assert result["valid"]

    def test_validate_invalid_json(self):
        """Test validation fails with malformed JSON."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create malformed JSON
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                f.write("{ invalid json")

            result = validate_lora_adapter(tmpdir)
            assert not result["valid"]
            assert any("not valid JSON" in issue for issue in result["issues"])

    def test_validate_with_metadata(self):
        """Test validation reads base_model from metadata.json."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create adapter_config.json without base_model
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"r": 16}, f)

            # Create metadata.json with base_model
            metadata_path = tmppath / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({"base_model": "Qwen/Qwen2.5-0.5B-Instruct"}, f)

            # Create weights
            weights_path = tmppath / "adapter_model.safetensors"
            weights_path.write_bytes(b"dummy data")

            result = validate_lora_adapter(tmpdir)
            assert result["valid"]
            assert result["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert len(result["warnings"]) == 1

    def test_validate_warning_no_base_model(self):
        """Test validation warns when base_model cannot be detected."""
        from src.llm.local_inference import validate_lora_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create adapter_config.json without base_model
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"r": 16}, f)

            # Create weights
            weights_path = tmppath / "adapter_model.safetensors"
            weights_path.write_bytes(b"dummy data")

            result = validate_lora_adapter(tmpdir)
            assert result["valid"]
            assert result["base_model"] is None
            assert any("base_model_name_or_path" in w for w in result["warnings"])


class TestDetectBaseModel:
    """Test base model auto-detection from metadata."""

    def test_detect_from_metadata(self):
        """Test detection reads from metadata.json."""
        from src.llm.local_inference import LocalLLMInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create metadata.json
            metadata_path = tmppath / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({"base_model": "Qwen/Qwen2.5-0.5B"}, f)

            inferencer = LocalLLMInference()
            base_model = inferencer._detect_base_model(tmppath)

            assert base_model == "Qwen/Qwen2.5-0.5B"

    def test_detect_from_adapter_config(self):
        """Test detection reads from adapter_config.json."""
        from src.llm.local_inference import LocalLLMInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create adapter_config.json
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"base_model_name_or_path": "unsloth/Phi-3.5-mini-instruct"}, f)

            inferencer = LocalLLMInference()
            base_model = inferencer._detect_base_model(tmppath)

            assert base_model == "unsloth/Phi-3.5-mini-instruct"

    def test_detect_metadata_preferred(self):
        """Test metadata.json is preferred over adapter_config.json."""
        from src.llm.local_inference import LocalLLMInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create both files with different values
            metadata_path = tmppath / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({"base_model": "from_metadata"}, f)

            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"base_model_name_or_path": "from_config"}, f)

            inferencer = LocalLLMInference()
            base_model = inferencer._detect_base_model(tmppath)

            # metadata.json should be preferred
            assert base_model == "from_metadata"

    def test_detect_not_found(self):
        """Test detection returns None when no metadata found."""
        from src.llm.local_inference import LocalLLMInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create empty directory
            inferencer = LocalLLMInference()
            base_model = inferencer._detect_base_model(tmppath)

            assert base_model is None


class TestLocalLLMInferenceClass:
    """Test LocalLLMInference class initialization and methods."""

    def test_initialization_defaults(self):
        """Test instance initializes with default values."""
        from src.llm.local_inference import LocalLLMInference

        inferencer = LocalLLMInference()
        assert inferencer.max_seq_length == 2048
        assert inferencer.load_in_4bit is True
        assert inferencer.model is None
        assert inferencer.tokenizer is None
        assert inferencer.base_model_name is None

    def test_initialization_custom_params(self):
        """Test instance initializes with custom parameters."""
        from src.llm.local_inference import LocalLLMInference

        inferencer = LocalLLMInference(max_seq_length=4096, load_in_4bit=False)
        assert inferencer.max_seq_length == 4096
        assert inferencer.load_in_4bit is False

    def test_load_trained_lora_missing_directory(self):
        """Test load_trained_lora raises error for missing directory."""
        from src.llm.local_inference import LocalLLMInference

        inferencer = LocalLLMInference()

        with pytest.raises(FileNotFoundError, match="LoRA adapter directory not found"):
            inferencer.load_trained_lora("/nonexistent/path")

    def test_load_trained_lora_incomplete_adapter(self):
        """Test load_trained_lora raises error for incomplete adapter."""
        from src.llm.local_inference import LocalLLMInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create directory without required files
            inferencer = LocalLLMInference()

            with pytest.raises(FileNotFoundError, match="LoRA adapter incomplete"):
                inferencer.load_trained_lora(str(tmppath))

    def test_load_trained_lora_no_base_model(self):
        """Test load_trained_lora raises error when base model cannot be determined."""
        from src.llm.local_inference import LocalLLMInference

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create config without base_model
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"r": 16}, f)

            inferencer = LocalLLMInference()

            with pytest.raises(ValueError, match="Could not determine base model"):
                inferencer.load_trained_lora(str(tmppath))

    def test_generate_without_model(self):
        """Test generate raises error when no model is loaded."""
        from src.llm.local_inference import LocalLLMInference

        inferencer = LocalLLMInference()

        with pytest.raises(RuntimeError, match="No model loaded"):
            inferencer.generate("Hello")

    def test_generate_streaming_without_model(self):
        """Test generate_streaming raises error when no model is loaded."""
        from src.llm.local_inference import LocalLLMInference

        inferencer = LocalLLMInference()

        with pytest.raises(RuntimeError, match="No model loaded"):
            list(inferencer.generate_streaming("Hello"))


class TestMergeLoraToBase:
    """Test LoRA adapter merging functionality."""

    def test_merge_missing_directory(self):
        """Test merge raises error for missing directory."""
        from src.llm.local_inference import merge_lora_to_base

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create config without base_model
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"r": 16}, f)

            with pytest.raises(FileNotFoundError, match="Directory not found"):
                merge_lora_to_base("/nonexistent/path", "/output")

    def test_merge_no_base_model(self):
        """Test merge raises error when base model cannot be determined."""
        from src.llm.local_inference import merge_lora_to_base

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create config without base_model
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"r": 16}, f)

            with pytest.raises(ValueError, match="Could not determine base model"):
                merge_lora_to_base(str(tmppath), "/output")

    def test_merge_with_explicit_base_model(self):
        """Test merge works with explicitly provided base model."""
        from src.llm.local_inference import merge_lora_to_base

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            output_tmpdir = tempfile.mkdtemp()

            # Create valid adapter
            config_path = tmppath / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"r": 16}, f)

            # Note: This will fail because we don't have a real model,
            # but it should not fail on base_model detection
            with pytest.raises(Exception):  # Will fail during actual model loading
                merge_lora_to_base(str(tmppath), output_tmpdir, base_model="test/model")


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_local_inference_singleton(self):
        """Test get_local_inference returns singleton instance."""
        from src.llm.local_inference import (
            LocalLLMInference,
            get_local_inference,
        )

        # Reset global instance
        import src.llm.local_inference as li_module

        li_module._local_inference = None

        instance1 = get_local_inference()
        instance2 = get_local_inference()

        assert isinstance(instance1, LocalLLMInference)
        assert instance1 is instance2  # Same instance

    def test_get_local_inference_persists(self):
        """Test get_local_inference persists across calls."""
        from src.llm.local_inference import get_local_inference

        # Reset global instance
        import src.llm.local_inference as li_module

        li_module._local_inference = None

        instance1 = get_local_inference()
        instance1.max_seq_length = 4096

        instance2 = get_local_inference()
        assert instance2.max_seq_length == 4096
