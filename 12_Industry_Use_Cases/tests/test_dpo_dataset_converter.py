"""Tests for DPO dataset conversion and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from src.llm.dataset_converter import (
    DPO_VALIDATION_ERRORS,
    _convert_txt_to_dpo_legacy,
    _parse_jsonl_file,
    convert_to_dpo_format,
    validate_dpo_format,
)


class TestDPOValidationErrors:
    """Test that all DPO validation error messages are defined and helpful."""

    def test_all_error_keys_exist(self):
        """Verify all expected error keys are present."""
        required_keys = [
            "missing_rejected",
            "missing_chosen",
            "missing_prompt",
            "invalid_format",
            "empty_file",
            "identical_chosen_rejected",
            "unsupported_format",
        ]
        for key in required_keys:
            assert key in DPO_VALIDATION_ERRORS, f"Missing error key: {key}"

    def test_error_messages_are_actionable(self):
        """Verify error messages contain actionable guidance."""
        for key, message in DPO_VALIDATION_ERRORS.items():
            assert len(message) > 100, f"Error message for '{key}' is too short"
            # Accept HTML structure with headers or lists
            assert "<h3>" in message or "<ul>" in message or "*" in message, (
                f"Error message for '{key}' lacks structure"
            )


class TestValidateDPOFormat:
    """Test DPO format validation."""

    def test_empty_dataset_fails(self):
        """Empty datasets should fail validation."""
        is_valid, error = validate_dpo_format([])
        assert not is_valid
        assert "empty" in error.lower()

    def test_missing_prompt_fails(self):
        """Examples without 'prompt' field should fail."""
        examples = [
            {
                "chosen": [{"role": "assistant", "content": "Hi"}],
                "rejected": [{"role": "assistant", "content": "No"}],
            }
        ]
        is_valid, error = validate_dpo_format(examples)
        assert not is_valid
        assert "Missing 'prompt' field" in error

    def test_missing_chosen_fails(self):
        """Examples without 'chosen' field should fail."""
        examples = [
            {
                "prompt": [{"role": "user", "content": "Hello"}],
                "rejected": [{"role": "assistant", "content": "No"}],
            }
        ]
        is_valid, error = validate_dpo_format(examples)
        assert not is_valid
        assert "Missing 'chosen' field" in error

    def test_missing_rejected_fails(self):
        """Examples without 'rejected' field should fail."""
        examples = [
            {
                "prompt": [{"role": "user", "content": "Hello"}],
                "chosen": [{"role": "assistant", "content": "Hi"}],
            }
        ]
        is_valid, error = validate_dpo_format(examples)
        assert not is_valid
        assert "Missing 'rejected' field" in error

    def test_identical_chosen_rejected_fails(self):
        """Examples with identical chosen and rejected should fail."""
        response = [{"role": "assistant", "content": "Same answer"}]
        examples = [
            {
                "prompt": [{"role": "user", "content": "Hello"}],
                "chosen": response,
                "rejected": response,
            }
        ]
        is_valid, error = validate_dpo_format(examples)
        assert not is_valid
        assert "identical" in error.lower()

    def test_valid_dpo_example_passes(self):
        """Valid DPO examples should pass validation."""
        examples = [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "chosen": [{"role": "assistant", "content": "The answer is 4."}],
                "rejected": [{"role": "assistant", "content": "I don't know."}],
            }
        ]
        is_valid, error = validate_dpo_format(examples)
        assert is_valid
        assert error == ""

    def test_multiple_valid_examples_pass(self):
        """Multiple valid DPO examples should pass validation."""
        examples = [
            {
                "prompt": [{"role": "user", "content": f"Question {i}?"}],
                "chosen": [{"role": "assistant", "content": f"Good answer {i}"}],
                "rejected": [{"role": "assistant", "content": f"Bad answer {i}"}],
            }
            for i in range(5)
        ]
        is_valid, error = validate_dpo_format(examples)
        assert is_valid
        assert error == ""

    def test_error_includes_example_number(self):
        """Validation errors should include the example number for debugging."""
        examples = [
            {
                "prompt": [{"role": "user", "content": "Valid"}],
                "chosen": [{"role": "assistant", "content": "Good"}],
                "rejected": [{"role": "assistant", "content": "Bad"}],
            },
            {"prompt": [{"role": "user", "content": "Invalid"}], "chosen": []},  # Missing rejected
        ]
        is_valid, error = validate_dpo_format(examples)
        assert not is_valid
        assert "Example 1" in error  # Second example (0-indexed)


class TestParseJSONLFile:
    """Test JSONL file parsing."""

    def test_parse_valid_jsonl(self, tmp_path):
        """Parse a valid JSONL file."""
        jsonl_file = tmp_path / "valid.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(
                '{"prompt": [{"role": "user", "content": "Q1"}], "chosen": [{"role": "assistant", "content": "A1"}], "rejected": [{"role": "assistant", "content": "B1"}]}\n'
            )
            f.write(
                '{"prompt": [{"role": "user", "content": "Q2"}], "chosen": [{"role": "assistant", "content": "A2"}], "rejected": [{"role": "assistant", "content": "B2"}]}\n'
            )

        examples = _parse_jsonl_file(str(jsonl_file))
        assert len(examples) == 2
        assert examples[0]["prompt"][0]["content"] == "Q1"
        assert examples[1]["chosen"][0]["content"] == "A2"

    def test_parse_jsonl_with_empty_lines(self, tmp_path):
        """JSONL files with empty lines should be handled gracefully."""
        jsonl_file = tmp_path / "with_empty.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(
                '{"prompt": [{"role": "user", "content": "Q1"}], "chosen": [{"role": "assistant", "content": "A1"}], "rejected": [{"role": "assistant", "content": "B1"}]}\n'
            )
            f.write("\n")  # Empty line
            f.write(
                '{"prompt": [{"role": "user", "content": "Q2"}], "chosen": [{"role": "assistant", "content": "A2"}], "rejected": [{"role": "assistant", "content": "B2"}]}\n'
            )

        examples = _parse_jsonl_file(str(jsonl_file))
        assert len(examples) == 2

    def test_parse_invalid_json_fails(self, tmp_path):
        """Invalid JSON should raise a detailed error."""
        jsonl_file = tmp_path / "invalid.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(
                '{"prompt": [{"role": "user", "content": "Q1"}], "chosen": [{"role": "assistant", "content": "A1"}], "rejected": [{"role": "assistant", "content": "B1"}]}\n'
            )
            f.write("this is not json\n")

        with pytest.raises(ValueError) as exc_info:
            _parse_jsonl_file(str(jsonl_file))

        error_msg = str(exc_info.value)
        assert "Invalid JSON on line 2" in error_msg

    def test_parse_empty_jsonl(self, tmp_path):
        """Empty JSONL files should return empty list."""
        jsonl_file = tmp_path / "empty.jsonl"
        with open(jsonl_file, "w") as f:
            pass  # Empty file

        examples = _parse_jsonl_file(str(jsonl_file))
        assert len(examples) == 0


class TestConvertToDPOFormat:
    """Test DPO format conversion."""

    def test_convert_jsonl_file(self, tmp_path):
        """Convert a valid JSONL file to DPO format."""
        jsonl_file = tmp_path / "dpo.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(
                '{"prompt": [{"role": "user", "content": "What is 2+2?"}], "chosen": [{"role": "assistant", "content": "4"}], "rejected": [{"role": "assistant", "content": "5"}]}\n'
            )

        examples = convert_to_dpo_format(str(jsonl_file))
        assert len(examples) == 1
        assert examples[0]["prompt"][0]["content"] == "What is 2+2?"
        assert examples[0]["chosen"][0]["content"] == "4"
        assert examples[0]["rejected"][0]["content"] == "5"

    def test_convert_jsonl_with_validation_error(self, tmp_path):
        """JSONL files with validation errors should raise ValueError."""
        jsonl_file = tmp_path / "invalid.jsonl"
        # Missing 'rejected' field
        with open(jsonl_file, "w") as f:
            f.write(
                '{"prompt": [{"role": "user", "content": "Q"}], "chosen": [{"role": "assistant", "content": "A"}]}\n'
            )

        with pytest.raises(ValueError) as exc_info:
            convert_to_dpo_format(str(jsonl_file))

        error_msg = str(exc_info.value)
        assert "Missing 'rejected' field" in error_msg

    def test_convert_txt_file_with_deprecation_warning(self, tmp_path):
        """TXT files should work but emit deprecation warnings."""
        txt_file = tmp_path / "qa_pairs.txt"
        with open(txt_file, "w") as f:
            f.write("Q: What is 2+2?\nA: The answer is 4.\n")

        with pytest.warns(DeprecationWarning):
            examples = convert_to_dpo_format(str(txt_file))

        # Should still parse, but with synthetic rejected
        assert len(examples) == 1

    def test_convert_unsupported_format(self, tmp_path):
        """Unsupported file formats should raise ValueError."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w") as f:
            f.write("col1,col2\nval1,val2\n")

        with pytest.raises(ValueError) as exc_info:
            convert_to_dpo_format(str(csv_file))

        error_msg = str(exc_info.value)
        assert "Unsupported file format" in error_msg

    def test_convert_nonexistent_file(self):
        """Non-existent files should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            convert_to_dpo_format("/nonexistent/path/file.jsonl")


class TestConvertTxtToDPOLegacy:
    """Test legacy TXT to DPO conversion."""

    def test_convert_valid_qa_pairs(self, tmp_path):
        """Convert valid Q:/A: format to DPO."""
        txt_file = tmp_path / "qa.txt"
        with open(txt_file, "w") as f:
            f.write("Q: What is the capital of France?\nA: Paris\n")
            f.write("Q: What is 2+2?\nA: 4\n")

        with pytest.warns(UserWarning, match="SYNTHETIC"):
            examples = _convert_txt_to_dpo_legacy(str(txt_file))

        assert len(examples) == 2
        # Check that synthetic rejected responses were created
        for ex in examples:
            assert "prompt" in ex
            assert "chosen" in ex
            assert "rejected" in ex

    def test_convert_empty_qa_pairs(self, tmp_path):
        """Empty Q:/A: pairs should be skipped."""
        txt_file = tmp_path / "empty.txt"
        with open(txt_file, "w") as f:
            f.write("Q: \nA: \n")

        with pytest.warns(UserWarning):
            examples = _convert_txt_to_dpo_legacy(str(txt_file))

        assert len(examples) == 0

    def test_convert_warns_about_synthetic_data(self, tmp_path):
        """TXT conversion should warn about synthetic rejected responses."""
        txt_file = tmp_path / "warning.txt"
        with open(txt_file, "w") as f:
            f.write("Q: Test?\nA: Answer\n")

        with pytest.warns(UserWarning, match="SYNTHETIC"):
            _convert_txt_to_dpo_legacy(str(txt_file))


class TestIntegration:
    """Integration tests for the full DPO conversion pipeline."""

    def test_full_pipeline_valid_jsonl(self, tmp_path):
        """Test the full pipeline with a valid JSONL file."""
        # Create test data
        jsonl_file = tmp_path / "test.jsonl"
        test_data = [
            {
                "prompt": [{"role": "user", "content": "What is Python?"}],
                "chosen": [{"role": "assistant", "content": "Python is a programming language."}],
                "rejected": [{"role": "assistant", "content": "I don't know what Python is."}],
            },
            {
                "prompt": [{"role": "user", "content": "What is 5+3?"}],
                "chosen": [{"role": "assistant", "content": "The answer is 8."}],
                "rejected": [{"role": "assistant", "content": "I think it's 7."}],
            },
        ]

        with open(jsonl_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Convert and validate
        examples = convert_to_dpo_format(str(jsonl_file))
        is_valid, _ = validate_dpo_format(examples)

        assert len(examples) == 2
        assert is_valid

    def test_full_pipeline_catches_data_issues(self, tmp_path):
        """Test that the pipeline catches data quality issues."""
        # Create test data with identical chosen/rejected
        jsonl_file = tmp_path / "bad.jsonl"
        test_data = [
            {
                "prompt": [{"role": "user", "content": "Test"}],
                "chosen": [{"role": "assistant", "content": "Same answer"}],
                "rejected": [{"role": "assistant", "content": "Same answer"}],  # Identical!
            }
        ]

        with open(jsonl_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Should fail validation
        with pytest.raises(ValueError) as exc_info:
            convert_to_dpo_format(str(jsonl_file))

        error_msg = str(exc_info.value)
        assert "identical" in error_msg.lower()
