"""Query Generator Agent - Crafts precise ML-specific search queries."""

import json
from langchain.chat_models import init_chat_model

from src.config import settings


QUERY_GENERATOR_PROMPT = """You are an expert at crafting search queries for ML/LLM debugging.

Error Context: {error_context}

Previous queries tried (if any): {query_history}

Generate 2-3 search queries that will help find solutions. Consider:
1. The specific error type and message
2. The training method (SFT/DPO/GRPO) if applicable  
3. The base model name
4. Hardware context (CUDA, GPU memory)

Return queries as a JSON list: ["query1", "query2", ...]

Important:
- Make queries specific but not too narrow
- Include library names when relevant (TRL, transformers, PyTorch)
- For CUDA errors, include CUDA/GPU keywords
- Avoid overly generic queries like "error in training"
"""

SYSTEM_CONTEXT_BRIEF = """System Environment:
- GPU: NVIDIA DGX Spark with 128GB unified memory
- CUDA Version: 13.1 (bitsandbytes quantization NOT compatible - use float16/bfloat16)
- Framework: PyTorch, Transformers, TRL, Metaflow
- Methods: SFT, DPO, GRPO"""


def query_generator_node(state: dict) -> dict:
    """LangGraph node that generates search queries.
    
    Input state keys used: error_context, iteration_count, query_history
    Output state keys set: search_queries, current_query
    
    Args:
        state: Current graph state containing error context and history
        
    Returns:
        Partial state dict with new search queries
    """
    error_context = state.get("error_context", {})
    iteration_count = state.get("iteration_count", 0)
    query_history = state.get("query_history", [])
    
    error_type = error_context.get("error_type", "Unknown")
    error_message = error_context.get("error_message", "")
    task_type = error_context.get("task_type", "")
    training_method = error_context.get("training_method", "")
    base_model = error_context.get("base_model", "")
    
    context_str = f"""Error Type: {error_type}
Error Message: {error_message[:300]}
Task Type: {task_type or 'unknown'}
Training Method: {training_method or 'N/A'}
Base Model: {base_model or 'N/A'}
{SYSTEM_CONTEXT_BRIEF}"""
    
    history_str = "None (first iteration)" if not query_history else ", ".join(
        [q.get("query", "") for q in query_history[-3:]]
    )
    
    try:
        model = init_chat_model(
            "openai/gpt-oss-120b",
            model_provider="openai",
            config={
                "base_url": settings.LLM_INFERENCE_URL,
                "api_key": settings.LLM_INFERENCE_KEY,
                "temperature": 0.2,
            },
        )
        
        prompt = QUERY_GENERATOR_PROMPT.format(
            error_context=context_str,
            query_history=history_str
        )
        
        response = model.invoke(prompt)
        content = response.content.strip()
        
        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            queries = json.loads(content)
            if not isinstance(queries, list):
                queries = [content]
        except json.JSONDecodeError:
            queries = [line.strip().strip('"').strip("'") 
                      for line in content.split("\n") 
                      if line.strip() and not line.strip().startswith("#")]
        
        queries = queries[:3] if queries else _fallback_queries(error_context)
        
    except Exception as e:
        print(f"[QueryGenerator] LLM error: {e}")
        queries = _fallback_queries(error_context)
    
    current_query = queries[0] if queries else f"{error_type} {error_message[:100]}"
    
    # Track progress using the planning tool
    try:
        from src.agent.investigation_planner import update_investigation_plan
        update_investigation_plan.invoke({
            "steps": ["analyze_error", "generate_query", "search_solutions", "evaluate_results", "fetch_docs", "synthesize"],
            "current_step_index": 1,  # We just completed query generation
            "status": "completed"
        })
    except Exception:
        pass  # Planning tool not critical
    
    return {
        "search_queries": queries,
        "current_query": current_query
    }


def _fallback_queries(error_context: dict) -> list[str]:
    """Generate fallback queries without LLM."""
    error_type = error_context.get("error_type", "")
    error_message = error_context.get("error_message", "")[:100]
    training_method = error_context.get("training_method", "")
    
    queries = []
    
    base_query = f"{error_type} {error_message}"
    if training_method:
        base_query += f" {training_method}"
    queries.append(base_query)
    
    if "cuda" in error_message.lower() or "memory" in error_message.lower():
        queries.append(f"CUDA out of memory pytorch transformers fix")
    
    if "format" in error_message.lower() or "validation" in error_message.lower():
        queries.append(f"{training_method} data format jsonl validation error")
    
    return queries[:3]