"""Search Executor Agent - Executes web searches and populates state."""

from src.agent.error_investigator import search_searxng, detect_error_domain, ENGINE_PRESETS
from src.agent.investigation_planner import MAX_REFINEMENT_ITERATIONS


def search_executor_node(state: dict) -> dict:
    """LangGraph node that executes web searches.
    
    Input state keys used: current_query, error_context, iteration_count
    
    Output state keys set:
    - search_results: list of search result dicts
    - iteration_count: incremented for tracking
    
    Args:
        state: Current graph state
        
    Returns:
        Partial state dict with search results and updated iteration count
    """
    current_query = state.get("current_query", "")
    error_context = state.get("error_context", {})
    
    # Increment iteration count for tracking
    iteration_count = state.get("iteration_count", 0) + 1
    
    # Determine search domain from error context
    error_type = error_context.get("error_type", "")
    error_message = error_context.get("error_message", "")
    domain = detect_error_domain(error_type, error_message)
    
    # Execute search
    if current_query:
        try:
            result_str = search_searxng.invoke({"query": current_query, "num_results": 8})
            
            # Parse results - convert string to list of dicts
            search_results = []
            lines = result_str.split("\n")
            current_result = {}
            
            for line in lines:
                if line.startswith(f"{search_results.count() + 1}. "):
                    if current_result:
                        search_results.append(current_result)
                    current_result = {"title": line.split(".", 2)[2].strip()}
                elif line.strip().startswith("URL:"):
                    current_result["url"] = line.split("URL:")[1].strip()
                elif line.strip().startswith("Source:"):
                    current_result["source"] = line.split("Source:")[1].strip()
                elif line.strip().startswith("Content:"):
                    current_result["content"] = line.split("Content:", 1)[1].strip()
            
            if current_result:
                search_results.append(current_result)
                
        except Exception as e:
            search_results = [{"error": str(e)}]
    else:
        search_results = []
    
    return {
        "search_results": search_results,
        "iteration_count": iteration_count
    }