"""Tests for structured error handling."""

import pytest


class TestErrorRegistry:
    """Tests for error registry and creation."""

    def test_create_registered_error(self):
        from src.utils.error_handling import create_user_error, ErrorCategory, ErrorSeverity

        error = create_user_error(
            "cuda_oom",
            attempted_mb="1024",
            free_mb="512",
            suggested_batch=8,
        )

        assert error.error_code == "cuda_oom"
        assert error.category == ErrorCategory.HARDWARE
        assert error.severity == ErrorSeverity.ERROR
        assert "GPU" in error.title

    def test_create_unknown_error(self):
        from src.utils.error_handling import create_user_error

        error = create_user_error("unknown_error_code")

        assert error.error_code == "unknown"

    def test_error_formatting_includes_steps(self):
        from src.utils.error_handling import create_user_error

        error = create_user_error(
            "missing_target_column",
            target_column="price",
            available_columns=["date", "quantity"],
        )

        formatted = str(error)

        assert "How to fix:" in formatted
        assert "price" in formatted


class TestExceptionClassification:
    """Tests for mapping exceptions to error codes."""

    def test_cuda_oom_classification(self):
        from src.utils.error_handling import classify_exception

        class FakeCUDAError(RuntimeError):
            pass

        error = FakeCUDAError("CUDA out of memory. Tried to allocate 2.00 GiB")

        code = classify_exception(error)

        assert code == "cuda_oom"

    def test_file_not_found_classification(self):
        from src.utils.error_handling import classify_exception

        error = FileNotFoundError("No such file or directory: /path/to/file.csv")

        code = classify_exception(error)

        assert code == "dataset_not_found"


class TestFormatExceptionForUser:
    """Tests for user-friendly exception formatting."""

    def test_formats_known_error(self):
        from src.utils.error_handling import format_exception_for_user

        error = FileNotFoundError("No such file: /data/train.csv")

        message = format_exception_for_user(error, {"dataset_path": "/data/train.csv"})

        assert "Dataset Not Found" in message
        assert "/data/train.csv" in message

    def test_formats_unknown_error(self):
        from src.utils.error_handling import format_exception_for_user

        error = ValueError("Some random error")

        message = format_exception_for_user(error)

        assert "Training Error" in message or "unknown" in message.lower()


class TestLegacyErrorFormatting:
    """Tests for legacy error formatting (backward compatibility)."""

    def test_format_error_missing_target(self):
        from src.utils.error_handling import format_error

        message = format_error(
            "missing_target",
            column="price",
            count=10,
            percent=5.0,
        )

        assert "price" in message
        assert "10" in message

    def test_format_error_file_not_found(self):
        from src.utils.error_handling import format_error

        message = format_error(
            "file_not_found",
            filepath="/data/train.csv",
        )

        assert "/data/train.csv" in message

    def test_format_error_invalid_csv(self):
        from src.utils.error_handling import format_error

        message = format_error(
            "invalid_csv",
            error_message="Expected 3 fields, saw 4",
            filepath="/data/train.csv",
        )

        assert "train.csv" in message
