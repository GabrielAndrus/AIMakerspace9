"""Recommendation Synthesizer Agent - Combines findings into actionable advice."""

from src.config import settings
from src.utils.langfuse_client import langfuse_trace, trace_llm_call


SYNTHESIZER_PROMPT = """You are an expert ML engineer providing debug recommendations for a specific hardware setup.

## System Environment
- GPU: NVIDIA DGX Spark with 128GB unified memory (CUDA-accelerated)
- CUDA Version: 13.1 (CRITICAL: bitsandbytes quantization NOT compatible - always use float16/bfloat16 instead)
- Framework: PyTorch, Transformers, TRL (Transformer Reinforcement Learning)
- Orchestration: Metaflow
- Training Methods: SFT, DPO, GRPO

## Error Context
{error_context}

## Search Results Summary
{search_summary}

## Fetched Documentation
{fetched_docs}

Provide a clear, actionable recommendation:
1. **Root Cause**: What likely caused this error (be specific)
2. **Fix Steps**: Numbered list of specific steps to fix it
3. **Configuration Changes**: Any parameters that need adjustment
4. **Code Example**: If applicable, show corrected code snippet

Be concise but thorough. Focus on solutions compatible with CUDA 13.1 and the hardware setup.
"""


def synthesizer_node(state: dict) -> dict:
    """LangGraph node that generates final recommendation.
    
    Input state keys used: error_context, search_results, fetched_docs
    Output state keys set: recommendation, confidence_level
    
    Args:
        state: Current graph state containing all investigation results
        
    Returns:
        Partial state dict with final recommendation and confidence
    """
    error_context = state.get("error_context", {})
    search_results = state.get("search_results", [])
    fetched_docs = state.get("fetched_docs", [])
    
    context_str = _format_error_context(error_context)
    search_summary = _summarize_search_results(search_results)
    docs_str = _format_fetched_docs(fetched_docs)
    
    # Try to find similar past errors for context
    similar_errors = []
    try:
        from src.agent.investigation_memory import find_similar_past_errors
        similar_errors = find_similar_past_errors(error_context)
    except Exception:
        pass  # Memory not available
    
    # Add to prompt context
    if similar_errors:
        context_str += "\n## SIMILAR PAST ERRORS\n"
        for i, past in enumerate(similar_errors[:2], 1):
            context_str += f"\nPast Error {i}: {past.get('error_type', 'Unknown')}\n"
            context_str += f"Resolution: {past.get('recommendation', 'N/A')[:300]}...\n"
    
    try:
        with langfuse_trace("synthesis", input={"error_type": error_context.get("error_type"), "search_results_count": len(search_results), "fetched_docs_count": len(fetched_docs)}) as trace:
            from langchain.chat_models import init_chat_model
            model = init_chat_model(
                settings.LLM_MODEL_NAME,
                model_provider="openai",
                base_url=settings.LLM_INFERENCE_URL,
                api_key=settings.LLM_INFERENCE_KEY,
                temperature=0.3,
            )

            prompt = SYNTHESIZER_PROMPT.format(
                error_context=context_str,
                search_summary=search_summary,
                fetched_docs=docs_str
            )

            response = model.invoke(prompt)
            recommendation = response.content.strip()
            trace_llm_call(trace, "synthesize_recommendation", prompt, recommendation, settings.LLM_MODEL_NAME)
            if trace:
                trace.update(output={"recommendation": recommendation})
        
        confidence_level = _assess_confidence(error_context, search_results, fetched_docs)
        
    except Exception as e:
        print(f"[Synthesizer] LLM error: {e}")
        recommendation = _fallback_recommendation(error_context, search_results, fetched_docs)
        confidence_level = "low"
    
    # Save to memory for future reference
    try:
        from src.agent.investigation_memory import save_successful_resolution
        save_successful_resolution(error_context, recommendation)
    except Exception:
        pass  # Memory not critical
    
    return {
        "recommendation": recommendation,
        "confidence_level": confidence_level
    }


def _format_error_context(error_context: dict) -> str:
    """Format error context for prompt."""
    lines = [
        f"Error Type: {error_context.get('error_type', 'Unknown')}",
        f"Error Message: {error_context.get('error_message', 'N/A')[:500]}",
        f"Task Type: {error_context.get('task_type', 'unknown')}",
    ]
    
    if error_context.get("flow_name"):
        lines.append(f"Flow: {error_context['flow_name']}")
    if error_context.get("step_name"):
        lines.append(f"Failed Step: {error_context['step_name']}")
    if error_context.get("training_method"):
        lines.append(f"Training Method: {error_context['training_method']}")
    if error_context.get("base_model"):
        lines.append(f"Base Model: {error_context['base_model']}")
    if error_context.get("data_path"):
        lines.append(f"Data Path: {error_context['data_path']}")
    
    traceback = error_context.get("traceback", "")
    if traceback:
        key_lines = [line for line in traceback.split("\n")[-8:] if line.strip()]
        lines.append(f"\nKey Traceback:\n{chr(10).join(key_lines[:5])}")
    
    return "\n".join(lines)


def _summarize_search_results(search_results: list[dict]) -> str:
    """Summarize search results for prompt."""
    if not search_results:
        return "No search results found."
    
    summaries = []
    for i, result in enumerate(search_results[:5], 1):
        title = result.get("title", "N/A")
        url = result.get("url", "")
        content = result.get("content", "")[:200]
        
        summaries.append(f"{i}. {title}\n   URL: {url}\n   Summary: {content}...")
    
    return "\n\n".join(summaries)


def _format_fetched_docs(fetched_docs: list[dict]) -> str:
    """Format fetched documentation for prompt."""
    if not fetched_docs:
        return "No documentation was successfully fetched."
    
    docs = []
    for doc in fetched_docs[:3]:
        if doc.get("fetch_success") and doc.get("content"):
            source = doc.get("source", "Unknown")
            url = doc.get("url", "")
            content = doc["content"][:1500]
            
            docs.append(f"### {source}\nURL: {url}\n\n{content}")
        else:
            docs.append(f"Failed to fetch from {doc.get('url', 'unknown')}: {doc.get('error', 'Unknown error')}")
    
    return "\n\n---\n\n".join(docs)


def _assess_confidence(error_context: dict, search_results: list[dict], fetched_docs: list[dict]) -> str:
    """Assess confidence level of recommendation."""
    score = 0
    
    if search_results and len(search_results) >= 3:
        score += 1
    if search_results and len(search_results) >= 5:
        score += 1
    
    successful_fetches = sum(1 for d in fetched_docs if d.get("fetch_success"))
    if successful_fetches >= 1:
        score += 2
    if successful_fetches >= 2:
        score += 1
    
    error_type = error_context.get("error_type", "").lower()
    common_errors = ["cuda", "memory", "valueerror", "runtimeerror", "typeerror"]
    if any(e in error_type for e in common_errors):
        score += 1
    
    if score >= 5:
        return "high"
    elif score >= 3:
        return "medium"
    else:
        return "low"


def _fallback_recommendation(error_context: dict, search_results: list[dict], fetched_docs: list[dict]) -> str:
    """Generate fallback recommendation without LLM."""
    error_message = error_context.get("error_message", "").lower()
    error_type = error_context.get("error_type", "")
    
    lines = ["## Recommendation (Fallback Mode)", ""]
    
    if "memory" in error_message or "oom" in error_message:
        lines.extend([
            "**Root Cause**: GPU memory exhaustion",
            "",
            "**Fix Steps**:",
            "1. Reduce `per_device_train_batch_size` (try halving current value)",
            "2. Enable gradient checkpointing: `gradient_checkpointing=True`",
            "3. If still failing, use a smaller model variant or reduce sequence length",
            "4. Consider using DeepSpeed ZeRO for memory-efficient training",
        ])
    
    elif "cuda" in error_message:
        lines.extend([
            "**Root Cause**: CUDA-related error (likely driver/hardware mismatch)",
            "",
            "**Fix Steps**:",
            "1. Verify CUDA version compatibility with PyTorch",
            "2. Check GPU visibility: `nvidia-smi`",
            "3. Ensure docker has `--gpus all --ipc host` flags",
            "4. For NCCL errors, verify shared memory configuration",
        ])
    
    elif "format" in error_message or "validation" in error_message:
        lines.extend([
            "**Root Cause**: Data format validation failure",
            "",
            "**Fix Steps**:",
            "1. Validate dataset file is proper JSONL (one JSON object per line)",
            "2. Check required fields match training method requirements:",
            "   - SFT: `{'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]}`",
            "   - DPO: `{'prompt': [...], 'chosen': [...], 'rejected': [...]}`",
            "3. Use `jq .` to validate JSON structure",
        ])
    
    else:
        lines.extend([
            f"**Error**: {error_type}",
            "",
            "**General Steps**:",
            "1. Review the full error message for specific details",
            "2. Check file paths and permissions",
            "3. Verify all dependencies are installed",
            "4. Consult documentation for the specific library/function",
        ])
    
    if search_results:
        lines.extend(["", "**Related Resources**:", ""])
        for r in search_results[:3]:
            lines.append(f"- {r.get('title', 'N/A')}: {r.get('url', '')}")
    
    return "\n".join(lines)