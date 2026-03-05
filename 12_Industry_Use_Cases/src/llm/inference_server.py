"""LLM Inference Server - OpenAI-compatible API client."""

from typing import Generator, Optional

try:
    import openai
except ImportError:
    openai = None


class LLMInferenceServer:
    """Client for OpenAI-compatible LLM inference endpoints."""

    def __init__(self, base_url: str = "http://192.168.1.79:8080/v1", api_key: str = "not-needed"):
        """Initialize the inference server.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            api_key: API key (often not needed for local endpoints)
        """
        if openai is None:
            raise ImportError("openai package required. Install with: uv pip install openai")

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "minimax-m2.5-mlx@8bit",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False,
    ):
        """Generate a response from the LLM.

        Args:
            prompt: User message
            system_prompt: System instruction (optional)
            model: Model name to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            Generated text or generator for streaming
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        if stream:
            return self._stream_response(**kwargs)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _stream_response(self, **kwargs) -> Generator[str, None, None]:
        """Stream response from the API."""
        kwargs["stream"] = True
        response = self.client.chat.completions.create(**kwargs)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


_inference_server: Optional[LLMInferenceServer] = None


def get_inference_server() -> LLMInferenceServer:
    """Get or create the global inference server instance."""
    global _inference_server
    if _inference_server is None:
        from src.config import settings

        _inference_server = LLMInferenceServer(
            base_url=settings.LLM_INFERENCE_URL, api_key=settings.LLM_INFERENCE_KEY
        )
    return _inference_server


def generate_response(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "minimax-m2.5-mlx@8bit",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
) -> str:
    """Generate a non-streaming response."""
    server = get_inference_server()
    return server.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False,
    )


def generate_streaming(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "minimax-m2.5-mlx@8bit",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
) -> Generator[str, None, None]:
    """Generate a streaming response."""
    server = get_inference_server()
    return server.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
    )
