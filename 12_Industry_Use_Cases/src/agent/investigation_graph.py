"""Investigation Graph - LangGraph StateGraph for multi-agent error investigation.

Architecture:
  START -> query_generator -> search_executor -> search_analyzer -> [fetch | refine] -> doc_fetcher -> synthesizer -> END
  
The search_analyzer has a conditional edge:
  - If result_quality_score >= 0.5: proceed to doc_fetcher
  - If result_quality_score < 0.5 and iteration_count < MAX_REFINEMENT_ITERATIONS: loop back to query_generator for refinement
"""

from langgraph.graph import StateGraph, END, START
from typing import Literal

from src.agent.state import InvestigationState
from src.agent.sub_agents import (
    query_generator_node,
    search_executor_node,
    search_analyzer_node,
    doc_fetcher_node,
    synthesizer_node,
)
from src.config import settings

MAX_REFINEMENT_ITERATIONS = settings.MAX_REFINEMENT_ITERATIONS


def route_after_search(state: dict) -> Literal["fetch", "refine"]:
    """Conditional edge: determine if we should fetch docs or refine the query.
    
    Args:
        state: Current graph state containing result_quality_score and iteration_count
        
    Returns:
        "fetch" - Results are good enough, proceed to documentation fetching
        "refine" - Results need improvement, loop back for refined query
    """
    quality_score = state.get("result_quality_score", 0.5)
    iteration_count = state.get("iteration_count", 0)
    
    if quality_score >= 0.5:
        return "fetch"
    elif iteration_count < MAX_REFINEMENT_ITERATIONS:
        return "refine"
    else:
        return "fetch"


def build_investigation_graph():
    """Build and compile the investigation multi-agent graph.
    
    Returns:
        Compiled LangGraph ready for invocation
    """
    workflow = StateGraph(InvestigationState)
    
    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("search_executor", search_executor_node)
    workflow.add_node("search_analyzer", search_analyzer_node)
    workflow.add_node("doc_fetcher", doc_fetcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.add_edge(START, "query_generator")
    workflow.add_edge("query_generator", "search_executor")
    workflow.add_edge("search_executor", "search_analyzer")
    
    workflow.add_conditional_edges(
        "search_analyzer",
        route_after_search,
        {
            "fetch": "doc_fetcher",
            "refine": "query_generator"
        }
    )
    
    workflow.add_edge("doc_fetcher", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


def run_investigation(
    error_context: dict,
    verbose: bool = True
) -> dict:
    """Convenience function to run the investigation graph.
    
    Args:
        error_context: Dict with error info (error_type, error_message, traceback, etc.)
        verbose: Print progress messages
        
    Returns:
        Final state dict with recommendation
    """
    initial_state = {
        "error_context": error_context,
        "investigation_steps": [
            "analyze_error",
            "generate_query", 
            "search_solutions",
            "evaluate_results",
            "fetch_docs",
            "synthesize"
        ],
        "current_step_index": 0,
        "iteration_count": 0,
        "search_queries": [],
        "current_query": "",
        "query_history": [],
        "search_results": [],
        "result_quality_score": 0.5,
        "needs_refinement": False,
        "fetched_docs": [],
        "recommendation": "",
        "confidence_level": "",
        "messages": [],
        "relevant_urls": [],
    }
    
    if verbose:
        error_type = error_context.get("error_type", "Unknown")
        print(f"[Investigation] Starting investigation for: {error_type}")
        print(f"[Investigation] Max refinement iterations: {MAX_REFINEMENT_ITERATIONS}")
    
    graph = build_investigation_graph()
    final_state = graph.invoke(initial_state)
    
    if verbose:
        recommendation = final_state.get("recommendation", "No recommendation generated")
        confidence = final_state.get("confidence_level", "unknown")
        iterations = final_state.get("iteration_count", 0)
        print(f"[Investigation] Completed in {iterations} refinement iteration(s)")
        print(f"[Investigation] Confidence: {confidence}")
        print(f"[Investigation] Recommendation preview: {recommendation[:200]}...")
    
    return final_state


def build_investigation_graph_with_memory(checkpointer=None):
    """Build graph with optional checkpointer for conversation continuity.
    
    Args:
        checkpointer: LangGraph checkpointer instance (e.g., MemorySaver)
                     If None, creates a default MemorySaver
        
    Returns:
        Compiled LangGraph with memory/checkpointing enabled
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    
    workflow = StateGraph(InvestigationState)
    
    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("search_executor", search_executor_node)
    workflow.add_node("search_analyzer", search_analyzer_node)
    workflow.add_node("doc_fetcher", doc_fetcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.add_edge(START, "query_generator")
    workflow.add_edge("query_generator", "search_executor")
    workflow.add_edge("search_executor", "search_analyzer")
    
    workflow.add_conditional_edges(
        "search_analyzer",
        route_after_search,
        {
            "fetch": "doc_fetcher",
            "refine": "query_generator"
        }
    )
    
    workflow.add_edge("doc_fetcher", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile(checkpointer=checkpointer)


def run_investigation_with_memory(
    error_context: dict,
    thread_id: str = "default",
    verbose: bool = True
) -> dict:
    """Run investigation with memory/checkpointing for multi-turn conversations.
    
    Args:
        error_context: Dict with error info
        thread_id: Thread ID for conversation continuity
        verbose: Print progress messages
        
    Returns:
        Final state dict with recommendation
    """
    initial_state = {
        "error_context": error_context,
        "investigation_steps": [
            "analyze_error",
            "generate_query",
            "search_solutions", 
            "evaluate_results",
            "fetch_docs",
            "synthesize"
        ],
        "current_step_index": 0,
        "iteration_count": 0,
        "search_queries": [],
        "current_query": "",
        "query_history": [],
        "search_results": [],
        "result_quality_score": 0.5,
        "needs_refinement": False,
        "fetched_docs": [],
        "recommendation": "",
        "confidence_level": "",
        "messages": [],
        "relevant_urls": [],
    }
    
    if verbose:
        error_type = error_context.get("error_type", "Unknown")
        print(f"[Investigation] Starting investigation (thread: {thread_id}) for: {error_type}")
    
    graph = build_investigation_graph_with_memory()
    config = {"configurable": {"thread_id": thread_id}}
    
    final_state = graph.invoke(initial_state, config=config)
    
    if verbose:
        recommendation = final_state.get("recommendation", "No recommendation generated")
        print(f"[Investigation] Recommendation preview: {recommendation[:200]}...")
    
    return final_state