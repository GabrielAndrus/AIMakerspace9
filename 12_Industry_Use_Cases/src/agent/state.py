from typing import TypedDict, Optional, Annotated, Any
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the AutoML Agent."""

    dataset_path: str
    target_column: str
    problem_type: str
    profile: dict
    query: str
    retrieved_contexts: list[dict]
    recommendation: str
    model_config: dict
    training_result: Optional[dict]
    error: Optional[str]


class InvestigationState(TypedDict):
    """State for the Error Investigation multi-agent graph."""

    error_context: dict

    investigation_steps: list[str]
    current_step_index: int
    iteration_count: int

    search_queries: list[str]
    current_query: str
    query_history: list[dict]

    search_results: list[dict]
    result_quality_score: float
    needs_refinement: bool

    fetched_docs: list[dict]

    recommendation: str
    confidence_level: str

    messages: Annotated[list[Any], add_messages]
