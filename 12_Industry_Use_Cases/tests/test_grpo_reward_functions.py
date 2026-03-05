"""Tests for GRPO reward function implementation."""

import pytest
from src.llm.trainers.grpo_trainer import (
    create_math_reward_func,
    create_format_check_reward_func,
    validate_grpo_config,
    REWARD_TEMPLATE_ERRORS,
    REWARD_TEMPLATES,
)


class TestMathRewardFunction:
    """Test cases for math reward function."""

    def test_correct_answer_simple(self):
        """Reward 1.0 when answer matches ground truth."""
        reward_func = create_math_reward_func("ground_truth")

        completions = ["The answer is 42.", "Result: 42", "42"]
        ground_truths = ["42"]

        rewards = reward_func(
            completions=completions,
            ground_truth=ground_truths * len(completions),
        )

        assert rewards == [1.0, 1.0, 1.0]

    def test_incorrect_answer(self):
        """Reward 0.0 when answer doesn't match ground truth."""
        reward_func = create_math_reward_func("ground_truth")

        completions = ["The answer is 100.", "Result: 50", "12"]
        ground_truths = ["42"]

        rewards = reward_func(
            completions=completions,
            ground_truth=ground_truths * len(completions),
        )

        assert rewards == [0.0, 0.0, 0.0]

    def test_boxed_format_correct(self):
        """Reward 1.0 when answer is boxed correctly."""
        reward_func = create_math_reward_func("ground_truth")

        completions = [r"The solution is \boxed{42}."]
        ground_truths = ["42"]

        rewards = reward_func(
            completions=completions,
            ground_truth=ground_truths * len(completions),
        )

        assert rewards == [1.0]

    def test_boxed_format_incorrect(self):
        """Reward 0.0 when boxed answer doesn't match."""
        reward_func = create_math_reward_func("ground_truth")

        completions = [r"The solution is \boxed{100}."]
        ground_truths = ["42"]

        rewards = reward_func(
            completions=completions,
            ground_truth=ground_truths * len(completions),
        )

        assert rewards == [0.0]

    def test_case_insensitive(self):
        """Reward function should be case-insensitive."""
        reward_func = create_math_reward_func("ground_truth")

        completions = ["The answer is A.", "Result: a", "A"]
        ground_truths = ["a"]

        rewards = reward_func(
            completions=completions,
            ground_truth=ground_truths * len(completions),
        )

        assert rewards == [1.0, 1.0, 1.0]

    def test_whitespace_handling(self):
        """Reward function should handle whitespace correctly."""
        reward_func = create_math_reward_func("ground_truth")

        completions = ["The answer is 42.", "Result:  42  ", "\t42\n"]
        ground_truths = ["42"]

        rewards = reward_func(
            completions=completions,
            ground_truth=ground_truths * len(completions),
        )

        assert rewards == [1.0, 1.0, 1.0]

    def test_custom_field_name(self):
        """Support custom field name for ground truth."""
        reward_func = create_math_reward_func("answer")

        completions = ["The answer is 42."]
        answers = ["42"]

        rewards = reward_func(
            completions=completions,
            answer=answers * len(completions),
        )

        assert rewards == [1.0]

    def test_empty_completions(self):
        """Handle empty completions gracefully."""
        reward_func = create_math_reward_func("ground_truth")

        rewards = reward_func(completions=[], ground_truth=[])

        assert rewards == []

    def test_no_ground_truth_provided(self):
        """Default to 0.0 when no ground truth provided."""
        reward_func = create_math_reward_func("ground_truth")

        completions = ["The answer is 42."]

        rewards = reward_func(completions=completions)

        assert rewards == [0.0]


class TestFormatCheckRewardFunction:
    """Test cases for format check reward function."""

    def test_pattern_found(self):
        """Reward 1.0 when pattern is found."""
        reward_func = create_format_check_reward_func(r"\d+")

        completions = ["The number is 42.", "Value: 123", "100 items"]

        rewards = reward_func(completions=completions)

        assert rewards == [1.0, 1.0, 1.0]

    def test_pattern_not_found(self):
        """Reward 0.0 when pattern is not found."""
        reward_func = create_format_check_reward_func(r"\d+")

        completions = ["No numbers here.", "Just text", "None"]

        rewards = reward_func(completions=completions)

        assert rewards == [0.0, 0.0, 0.0]

    def test_json_pattern(self):
        """Check for JSON formatting."""
        reward_func = create_format_check_reward_func(r"\{.*\}")

        completions = [
            '{"result": 42}',
            'The answer is {"value": 100}',
            "Just text",
        ]

        rewards = reward_func(completions=completions)

        assert rewards == [1.0, 1.0, 0.0]

    def test_code_block_pattern(self):
        """Check for code block formatting."""
        reward_func = create_format_check_reward_func(r"```.*```")

        completions = [
            "Here's the code:\n```\ndef foo():\n    return 42\n```",
            "No code here",
        ]

        rewards = reward_func(completions=completions)

        assert rewards == [1.0, 0.0]

    def test_complex_regex(self):
        """Support complex regex patterns."""
        reward_func = create_format_check_reward_func(r"Step \d+:.*")

        completions = [
            "Step 1: Analyze the problem.\nStep 2: Solve it.",
            "No steps here",
            "step 1: lowercase shouldn't match",  # Case-sensitive
        ]

        rewards = reward_func(completions=completions)

        assert rewards == [1.0, 0.0, 0.0]

    def test_empty_pattern(self):
        """Empty pattern matches everything."""
        reward_func = create_format_check_reward_func(r".*")

        completions = ["Any text", "", "More text"]

        rewards = reward_func(completions=completions)

        assert rewards == [1.0, 1.0, 1.0]

    def test_empty_completions(self):
        """Handle empty completions."""
        reward_func = create_format_check_reward_func(r"\d+")

        rewards = reward_func(completions=[])

        assert rewards == []

    def test_multiline_pattern(self):
        """Support multiline patterns."""
        reward_func = create_format_check_reward_func(r"Answer:.*")

        completions = [
            "First line\nSecond line\nAnswer: 42",
            "No answer here",
        ]

        rewards = reward_func(completions=completions)

        assert rewards == [1.0, 0.0]


class TestValidateGRPOConfig:
    """Test cases for GRPO configuration validation."""

    def test_no_template_or_custom_code(self):
        """Error when no template or custom code provided."""
        dataset = [{"prompt": "test"}]

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template=None,
            custom_reward_code=None,
        )

        assert not is_valid
        assert "reward function" in error_message.lower()
        assert "GRPO" in error_message

    def test_invalid_template_name(self):
        """Error with invalid template name."""
        dataset = [{"prompt": "test"}]

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template="invalid_template",
            custom_reward_code=None,
        )

        assert not is_valid
        assert "Invalid" in error_message or "invalid" in error_message.lower()
        assert "math" in error_message  # Should list available templates

    def test_math_template_missing_ground_truth(self):
        """Error when math template used without ground truth field."""
        dataset = [
            {"prompt": "What is 2+2?", "answer": "4"}  # 'ground_truth' field missing
        ]

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template="math",
            custom_reward_code=None,
        )

        assert not is_valid
        assert "ground_truth" in error_message.lower()

    def test_math_template_with_ground_truth(self):
        """Valid config with math template and ground truth."""
        dataset = [{"prompt": "What is 2+2?", "ground_truth": "4"}]

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template="math",
            custom_reward_code=None,
        )

        assert is_valid
        assert error_message == ""

    def test_format_check_template_missing_pattern(self):
        """Error when format_check template used without pattern field."""
        dataset = [{"prompt": "Format your answer as JSON"}]

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template="format_check",
            custom_reward_code=None,
        )

        assert not is_valid
        assert "pattern" in error_message.lower()

    def test_format_check_template_with_pattern(self):
        """Valid config with format_check template and pattern."""
        dataset = [{"prompt": "Format your answer as JSON", "pattern": r"\{.*\}"}]

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template="format_check",
            custom_reward_code=None,
        )

        assert is_valid
        assert error_message == ""

    def test_custom_reward_code_valid(self):
        """Valid config with custom reward code."""
        dataset = [{"prompt": "test"}]
        custom_code = """
def reward_func(completions, **kwargs):
    return [1.0 for _ in completions]
"""

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template=None,
            custom_reward_code=custom_code.strip(),
        )

        assert is_valid
        assert error_message == ""

    def test_empty_dataset(self):
        """Handle empty dataset gracefully."""
        dataset = []

        is_valid, error_message = validate_grpo_config(
            dataset=dataset,
            reward_template="math",
            custom_reward_code=None,
        )

        assert not is_valid
        assert "ground_truth" in error_message.lower() or "dataset" in error_message.lower()


class TestRewardTemplateErrors:
    """Test that error messages are helpful."""

    def test_no_template_error_message(self):
        """No template error should explain GRPO requirements."""
        error = REWARD_TEMPLATE_ERRORS["no_template"]

        assert "reward function" in error.lower()
        assert "GRPO" in error
        assert (
            "Group Relative Policy Optimization" in error or "multiple responses" in error.lower()
        )

    def test_math_no_ground_truth_message(self):
        """Math ground truth error should show required format."""
        error = REWARD_TEMPLATE_ERRORS["math_no_ground_truth"]

        assert "ground truth" in error.lower()
        assert '{"prompt":' in error
        assert '"ground_truth"' in error

    def test_invalid_template_name_message(self):
        """Invalid template error should list available templates."""
        error = REWARD_TEMPLATE_ERRORS["invalid_template_name"]

        assert "Available" in error or "available" in error.lower()
        assert "math" in error
        assert "format_check" in error


class TestRewardTemplatesRegistry:
    """Test the reward templates registry."""

    def test_registry_contains_expected_templates(self):
        """Registry should have expected templates."""
        assert "math" in REWARD_TEMPLATES
        assert "format_check" in REWARD_TEMPLATES

    def test_math_template_metadata(self):
        """Math template should have correct metadata."""
        math_template = REWARD_TEMPLATES["math"]

        assert "name" in math_template
        assert "description" in math_template
        assert "required_fields" in math_template
        assert "ground_truth" in math_template["required_fields"]
        assert "create_func" in math_template

    def test_format_check_template_metadata(self):
        """Format check template should have correct metadata."""
        format_template = REWARD_TEMPLATES["format_check"]

        assert "name" in format_template
        assert "description" in format_template
        assert "required_fields" in format_template
        assert "pattern" in format_template["required_fields"]
        assert "create_func" in format_template

    def test_create_functions_are_callable(self):
        """Create functions should be callable."""
        math_func = REWARD_TEMPLATES["math"]["create_func"]
        format_func = REWARD_TEMPLATES["format_check"]["create_func"]

        assert callable(math_func)
        assert callable(format_func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
