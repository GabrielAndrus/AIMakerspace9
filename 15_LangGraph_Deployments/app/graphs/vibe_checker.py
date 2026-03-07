"""An agent graph with a post-response vibe checker evaluation node.

After the agent responds, a secondary node evaluates if the response matches
a positive/casual vibe. If it does, end; otherwise, loop back or terminate after a safe limit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class VibeCheckResult(BaseModel):
    is_positive_casual: bool = Field(
        description="Whether the response matches a positive/casual vibe"
    )


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_vibe_check(state: MessagesState):
    """Decide whether to execute tools or run the vibe checker."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "vibe_check"


_vibe_prompt = ChatPromptTemplate.from_template(
    "Given an initial query and a final response, determine if the final response "
    "matches a positive/casual vibe (friendly, relaxed, conversational tone).\n\n"
    "Initial Query:\n{initial_query}\n\n"
    "Final Response:\n{final_response}"
)


def vibe_check_node(state: MessagesState) -> dict:
    """Evaluate if the latest response matches a positive/casual vibe."""
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="VIBE_CHECK:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(
        VibeCheckResult
    )
    result = (_vibe_prompt | structured_model).invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if result.is_positive_casual else "N"
    return {"messages": [AIMessage(content=f"VIBE_CHECK:{decision}")]}


def vibe_check_decision(state: MessagesState):
    """Terminate on 'VIBE_CHECK:Y' or loop otherwise; guard against infinite loops."""
    if any(
        getattr(m, "content", "") == "VIBE_CHECK:END" for m in state["messages"][-1:]
    ):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "VIBE_CHECK:Y" in text:
        return "end"
    return "continue"


def build_graph():
    """Build an agent graph with an auxiliary vibe checker evaluation subgraph."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("vibe_check", vibe_check_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_vibe_check,
        {"action": "action", "vibe_check": "vibe_check"},
    )
    graph.add_conditional_edges(
        "vibe_check",
        vibe_check_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
