"""Model utilities for constructing chat LLM clients.

Centralizes configuration of the default chat model and temperature so graphs can
import a single helper without repeating provider-specific wiring.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI


def get_chat_model(model_name: str | None = None, *, temperature: float = 0) -> Any:
    """Return a configured LangChain ChatOpenAI client.

    - model_name: optional override. If not provided, uses OPENAI_MODEL env var,
      falling back to "minimax-m2.5-mlx@8bit".
    - temperature: sampling temperature for the chat model.

    Returns: a LangChain-compatible chat model instance.
    """
    name = model_name or os.environ.get("OPENAI_MODEL", "minimax-m2.5-mlx@8bit")

    callbacks = None
    if os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler

            callbacks = [CallbackHandler()]
        except Exception:
            pass

    return ChatOpenAI(
        model=name,
        temperature=temperature,
        base_url="http://192.168.1.79:8080/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        callbacks=callbacks,
    )
