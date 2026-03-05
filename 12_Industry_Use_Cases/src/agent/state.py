from typing import TypedDict, Optional
from langgraph.graph import StateGraph


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
