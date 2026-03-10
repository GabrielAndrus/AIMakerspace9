"""Search Analyzer Agent - Evaluates and filters search results."""

import json

from src.config import settings
from src.utils.langfuse_client import langfuse_trace, trace_llm_call


QUALITY_EVALUATION_PROMPT = """Evaluate the relevance of these search results for debugging an ML error.

Error: {error_type}: {error_message}

System Context:
- CUDA 13.1 (no bitsandbytes quantization)
- PyTorch, Transformers, TRL framework
- Metaflow orchestration

Search Results:
{search_results}

Rate overall quality (0.0-1.0) and identify the most relevant URLs.
Consider:
1. Does result directly address this error type?
2. Is it specific to ML/LLM training context?
3. Is the source authoritative (GitHub issues, HF docs, Stack Overflow)?
4. Are there code examples or configuration fixes?

Return JSON: {{"quality_score": 0.7, "relevant_urls": ["url1", ...], "reasoning": "..."}}
"""


def search_analyzer_node(state: dict) -> dict:
    """LangGraph node that analyzes search results quality.
    
    Input state keys used: error_context, search_results
    Output state keys set: result_quality_score, needs_refinement, relevant_urls
    
    Args:
        state: Current graph state containing error context and search results
        
    Returns:
        Partial state dict with quality assessment
    """
    error_context = state.get("error_context", {})
    search_results = state.get("search_results", [])
    
    if not search_results:
        return {
            "result_quality_score": 0.0,
            "needs_refinement": True,
            "relevant_urls": [],
            "iteration_count": state.get("iteration_count", 0),
        }
    
    error_type = error_context.get("error_type", "Unknown")
    error_message = error_context.get("error_message", "")[:200]
    
    results_str = _format_search_results(search_results)
    
    try:
        with langfuse_trace("search_quality_evaluation", input={"error_type": error_type, "results_count": len(search_results)}) as trace:
            from langchain.chat_models import init_chat_model
            model = init_chat_model(
                settings.LLM_MODEL_NAME,
                model_provider="openai",
                base_url=settings.LLM_INFERENCE_URL,
                api_key=settings.LLM_INFERENCE_KEY,
                temperature=0.1,
            )

            prompt = QUALITY_EVALUATION_PROMPT.format(
                error_type=error_type,
                error_message=error_message,
                search_results=results_str
            )

            response = model.invoke(prompt)
            content = response.content.strip()
            trace_llm_call(trace, "evaluate_quality", prompt, content, settings.LLM_MODEL_NAME)
            if trace:
                trace.update(output={"evaluation": content})
        
        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            evaluation = json.loads(content)
            
            quality_score = float(evaluation.get("quality_score", 0.5))
            relevant_urls = evaluation.get("relevant_urls", [])
            reasoning = evaluation.get("reasoning", "")
            
        except (json.JSONDecodeError, ValueError):
            quality_score, relevant_urls = _fallback_evaluation(error_context, search_results)
            reasoning = "Fallback heuristic evaluation"
        
    except Exception as e:
        print(f"[SearchAnalyzer] LLM error: {e}")
        quality_score, relevant_urls = _fallback_evaluation(error_context, search_results)
        reasoning = f"Error occurred: {str(e)[:50]}"
    
    needs_refinement = quality_score < 0.5
    
    return {
        "result_quality_score": quality_score,
        "needs_refinement": needs_refinement,
        "relevant_urls": relevant_urls[:3],
        "iteration_count": state.get("iteration_count", 0),
    }


def _format_search_results(search_results: list[dict]) -> str:
    """Format search results for LLM prompt."""
    formatted = []
    for i, result in enumerate(search_results[:8], 1):
        title = result.get("title", "N/A")
        url = result.get("url", "")
        content = result.get("content", "")[:300]
        engine = result.get("engine", "")
        
        formatted.append(f"{i}. {title}\n   URL: {url}\n   Source: {engine}\n   Snippet: {content}...")
    
    return "\n\n".join(formatted)


def _fallback_evaluation(error_context: dict, search_results: list[dict]) -> tuple[float, list[str]]:
    """Fallback evaluation without LLM using heuristics."""
    error_message = error_context.get("error_message", "").lower()
    relevant_urls = []
    
    priority_domains = ["github.com", "huggingface.co", "stackoverflow.com"]
    
    for result in search_results[:5]:
        url = result.get("url", "")
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        
        is_priority_domain = any(domain in url for domain in priority_domains)
        error_keywords_present = any(
            kw in title or kw in content 
            for kw in ["error", "fix", "solution", "solved"]
        )
        context_match = any(
            kw in title or kw in content
            for kw in ["pytorch", "transformers", "trl", "cuda", "training"]
        )
        
        if is_priority_domain and (error_keywords_present or context_match):
            relevant_urls.append(url)
    
    quality_score = min(len(relevant_urls) * 0.25, 1.0)
    
    return quality_score, relevant_urls