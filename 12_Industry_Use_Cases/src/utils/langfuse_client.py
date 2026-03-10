from langfuse import Langfuse
from contextlib import contextmanager

_langfuse_client = None

def get_langfuse_client():
    """Get or create Langfuse client singleton."""
    global _langfuse_client
    if _langfuse_client is None:
        try:
            from src.config import settings
            _langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
        except Exception as e:
            print(f"Warning: Langfuse init failed: {e}")
    return _langfuse_client

@contextmanager
def langfuse_trace(name, metadata=None, user_id=None, input=None):
    """Context manager for wrapping operations in Langfuse traces.

    Uses start_as_current_span which creates a proper trace in Langfuse v3.
    Child spans/generations created inside will be nested under this trace.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    with client.start_as_current_span(name=name, metadata=metadata or {}, input=input) as span:
        try:
            yield span
        finally:
            client.flush()

def trace_llm_call(trace, name, input_text, output_text, model, metadata=None):
    """Record an LLM generation within a trace."""
    if trace is None:
        return
    gen = trace.start_generation(
        name=name,
        input=input_text,
        model=model,
        metadata=metadata or {},
    )
    gen.update(output=output_text)
    gen.end()

def trace_retrieval(trace, name, query, results, metadata=None):
    """Record a retrieval operation within a trace."""
    if trace is None:
        return
    span = trace.start_span(
        name=name,
        input={"query": query},
        metadata=metadata or {},
    )
    span.update(output={"results": results[:5] if isinstance(results, list) else str(results)[:500]})
    span.end()
