import os
import uuid


_langfuse_client = None


def get_langfuse_client():
    """Get or create the Langfuse client for observability."""
    global _langfuse_client

    if _langfuse_client is None:
        try:
            from langfuse import get_client
            _langfuse_client = get_client()
        except Exception as e:
            print(f"Warning: Failed to initialize LangFuse client: {e}")
            return None

    return _langfuse_client


def trace_agent_step(step_name: str, input_data: dict, output_data: dict = None):
    """Trace an agent step in Langfuse using create_score (LangFuse auto-creates traces)."""
    try:
        client = get_langfuse_client()
        if client is None:
            return
        
        trace_id = f"agent_step_{uuid.uuid4().hex[:8]}"
        
        if input_data:
            for key, value in input_data.items():
                client.create_score(
                    trace_id=trace_id,
                    name=f"{step_name}_input_{key}",
                    value=1.0,
                    data_type="NUMERIC",
                    comment=f"Input: {value}"[:200],
                )
        
        if output_data:
            for key, value in output_data.items():
                client.create_score(
                    trace_id=trace_id,
                    name=f"{step_name}_output_{key}",
                    value=1.0,
                    data_type="NUMERIC",
                    comment=f"Output: {value}"[:200],
                )
        
        client.flush()
    except Exception as e:
        print(f"Warning: Langfuse tracing failed: {e}")