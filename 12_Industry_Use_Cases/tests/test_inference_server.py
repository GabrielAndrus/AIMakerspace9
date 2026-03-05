"""Tests for LLM Inference Server.

Tests the OpenAI-compatible API client with streaming and non-streaming modes.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestLLMInferenceServer:
    """Test suite for LLMInferenceServer class."""

    def test_initialization(self):
        """Test that server initializes with correct parameters."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            server = self._create_server(base_url="http://test.local:8080/v1", api_key="test-key")

            mock_openai.OpenAI.assert_called_once_with(
                base_url="http://test.local:8080/v1", api_key="test-key"
            )
            assert server.client is not None

    def test_initialization_defaults(self):
        """Test that server uses default parameters."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            server = self._create_server()

            # Check that default values were passed
            call_args = mock_openai.OpenAI.call_args
            assert call_args[1]["base_url"] == "http://192.168.1.79:8080/v1"
            assert call_args[1]["api_key"] == "not-needed"

    def test_initialization_without_openai(self):
        """Test that Initialization fails gracefully without openai package."""
        with patch("src.llm.inference_server.openai", None):
            from src.llm import inference_server

            with pytest.raises(ImportError, match="openai package required"):
                inference_server.LLMInferenceServer()

    def test_generate_non_streaming(self):
        """Test non-streaming generation."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            # Setup mock response
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            result = server.generate(prompt="Hello", system_prompt="You are helpful", stream=False)

            # Verify the call was made correctly
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]

            assert call_kwargs["model"] == "minimax-m2.5-mlx@8bit"
            assert call_kwargs["messages"] == [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 1024
            assert call_kwargs["top_p"] == 1.0
            assert "stream" not in call_kwargs or call_kwargs["stream"] is False

            # Verify result
            assert result == "Test response"

    def test_generate_with_custom_parameters(self):
        """Test generation with custom parameters."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Custom response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            result = server.generate(
                prompt="Test",
                model="custom-model",
                temperature=0.5,
                max_tokens=512,
                top_p=0.9,
                stream=False,
            )

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "custom-model"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 512
            assert call_kwargs["top_p"] == 0.9

    def test_streaming_generation(self):
        """Test streaming generation."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            # Setup streaming chunks
            chunk1 = Mock()
            delta1 = Mock()
            delta1.content = "Hello "
            choice1 = Mock()
            choice1.delta = delta1
            chunk1.choices = [choice1]

            chunk2 = Mock()
            delta2 = Mock()
            delta2.content = "world"
            choice2 = Mock()
            choice2.delta = delta2
            chunk2.choices = [choice2]

            chunk3 = Mock()
            delta3 = Mock()
            delta3.content = None  # End of stream
            choice3 = Mock()
            choice3.delta = delta3
            chunk3.choices = [choice3]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            result_generator = server.generate(prompt="Test", stream=True)

            # Collect results
            chunks = list(result_generator)
            assert chunks == ["Hello ", "world"]

    def test_stream_with_empty_chunks(self):
        """Test that streaming handles empty chunks gracefully."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            # Chunk with no choices
            chunk1 = Mock()
            chunk1.choices = []

            # Chunk with content
            chunk2 = Mock()
            delta2 = Mock()
            delta2.content = "Test"
            choice2 = Mock()
            choice2.delta = delta2
            chunk2.choices = [choice2]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            chunks = list(server.generate(prompt="Test", stream=True))

            assert chunks == ["Test"]

    def test_stream_helper_method(self):
        """Test the _stream_response helper method."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            chunk = Mock()
            delta = Mock()
            delta.content = "Streaming"
            choice = Mock()
            choice.delta = delta
            chunk.choices = [choice]

            mock_client = Mock()
            mock_response = iter([chunk])
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            chunks = list(
                server._stream_response(
                    model="test-model", messages=[{"role": "user", "content": "Test"}]
                )
            )

            assert chunks == ["Streaming"]
            # Verify stream=True was passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["stream"] is True

    def _create_server(self, **kwargs):
        """Helper to create server instance with mocked openai."""
        from src.llm.inference_server import LLMInferenceServer

        return LLMInferenceServer(**kwargs)


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_inference_server_singleton(self):
        """Test that get_inference_server returns singleton instance."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            import src.llm.inference_server as is_module

            is_module._inference_server = None

            server1 = is_module.get_inference_server()
            server2 = is_module.get_inference_server()

            # Should be the same instance
            assert server1 is server2

    def test_get_inference_server_uses_config(self):
        """Test that get_inference_server uses config settings."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            from src.config import settings

            original_url = settings.LLM_INFERENCE_URL
            original_key = settings.LLM_INFERENCE_KEY

            try:
                # Manually patch settings
                object.__setattr__(settings, "LLM_INFERENCE_URL", "http://config-url:8080/v1")
                object.__setattr__(settings, "LLM_INFERENCE_KEY", "config-key")

                import src.llm.inference_server as is_module

                is_module._inference_server = None

                server = is_module.get_inference_server()

                # Verify config values were used
                mock_openai.OpenAI.assert_called_once_with(
                    base_url="http://config-url:8080/v1", api_key="config-key"
                )
            finally:
                # Restore original values
                object.__setattr__(settings, "LLM_INFERENCE_URL", original_url)
                object.__setattr__(settings, "LLM_INFERENCE_KEY", original_key)


class TestHelperFunctions:
    """Test convenience helper functions."""

    def test_generate_response(self):
        """Test generate_response helper function."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Helper response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            import src.llm.inference_server as is_module

            is_module._inference_server = None

            result = is_module.generate_response(
                prompt="Test prompt", system_prompt="Custom system"
            )

            assert result == "Helper response"

    def test_generate_response_defaults(self):
        """Test generate_response uses default parameters."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            import src.llm.inference_server as is_module

            is_module._inference_server = None

            is_module.generate_response("Test")

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "minimax-m2.5-mlx@8bit"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 1024

    def test_generate_streaming(self):
        """Test generate_streaming helper function."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            chunk = Mock()
            delta = Mock()
            delta.content = "Streamed"
            choice = Mock()
            choice.delta = delta
            chunk.choices = [choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = iter([chunk])
            mock_openai.OpenAI.return_value = mock_client

            import src.llm.inference_server as is_module

            is_module._inference_server = None

            result = is_module.generate_streaming("Test prompt")

            chunks = list(result)
            assert chunks == ["Streamed"]


class TestConfiguration:
    """Test configuration integration."""

    def test_config_settings_exist(self):
        """Verify config has required LLM inference settings."""
        from src.config import settings

        assert hasattr(settings, "LLM_INFERENCE_URL")
        assert hasattr(settings, "LLM_INFERENCE_KEY")

    def test_config_default_values(self):
        """Test config has sensible defaults."""
        from src.config import settings

        assert settings.LLM_INFERENCE_URL == "http://192.168.1.79:8080/v1"
        assert settings.LLM_INFERENCE_KEY == "not-needed"


class TestModuleImports:
    """Test that module imports correctly."""

    def test_module_imports(self):
        """Verify all expected exports are available."""
        from src.llm import inference_server

        assert hasattr(inference_server, "LLMInferenceServer")
        assert hasattr(inference_server, "get_inference_server")
        assert hasattr(inference_server, "generate_response")
        assert hasattr(inference_server, "generate_streaming")

    def test_docstrings(self):
        """Verify docstrings are present."""
        from src.llm.inference_server import LLMInferenceServer

        assert LLMInferenceServer.__doc__ is not None
        assert LLMInferenceServer.generate.__doc__ is not None
        assert LLMInferenceServer._stream_response.__doc__ is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = ""
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            result = server.generate(prompt="", stream=False)

            assert result == ""

    def test_special_characters_in_prompt(self):
        """Test handling of special characters."""
        with patch("src.llm.inference_server.openai") as mock_openai:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Special response: ✓★♥"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            server = self._create_server()
            special_prompt = "Test with emoji: 🚀 and unicode: café"

            result = server.generate(prompt=special_prompt, stream=False)

            assert result == "Special response: ✓★♥"

    def _create_server(self, **kwargs):
        """Helper to create server instance with mocked openai."""
        from src.llm.inference_server import LLMInferenceServer

        return LLMInferenceServer(**kwargs)
