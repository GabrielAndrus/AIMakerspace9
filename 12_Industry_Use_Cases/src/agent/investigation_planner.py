"""
Investigation planner for Deep Error Investigation Agent System.

This module provides planning and tracking tools for the multi-step
error investigation workflow, including:
- Step-by-step investigation plan management
- Todo tracking (no-op context management tool)
- Iteration/refinement logic for poor search results

Based on LangChain Deep Agents patterns.
"""

from dataclasses import dataclass, field
from typing import Optional
from langchain_core.tools import tool


INVESTIGATION_STEPS = [
    ("analyze_error", "Analyze error context and identify key information"),
    ("generate_query", "Generate search query for solutions"),
    ("search_solutions", "Search for relevant solutions"),
    ("evaluate_results", "Evaluate quality of search results"),
    ("fetch_docs", "Fetch documentation from top URLs"),
    ("synthesize", "Synthesize recommendation")
]

from src.config import settings

MAX_REFINEMENT_ITERATIONS = settings.MAX_REFINEMENT_ITERATIONS


@tool
def update_investigation_plan(
    steps: list[str],
    current_step_index: int,
    status: str
) -> str:
    """Track investigation progress. This is a context management tool.

    Use this to maintain awareness of the investigation plan and progress.
    It does not execute any actions - it's purely for context tracking.

    Args:
        steps: List of investigation step names
        current_step_index: Current step (0-indexed)
        status: "in_progress", "completed", or "needs_refinement"

    Returns:
        Confirmation string with plan state
    """
    valid_statuses = {"in_progress", "completed", "needs_refinement"}
    if status not in valid_statuses:
        return f"Error: Invalid status. Must be one of {valid_statuses}"

    if current_step_index < 0 or current_step_index >= len(steps):
        return f"Error: Step index out of bounds (0-{len(steps) - 1})"

    step_name = steps[current_step_index]
    progress_pct = ((current_step_index + 1) / len(steps)) * 100

    status_emoji = {
        "in_progress": "[~]",
        "completed": "[✓]",
        "needs_refinement": "[!]"
    }.get(status, "[?]")

    return (
        f"Investigation Plan Updated\n"
        f"─────────────────────────\n"
        f"Status: {status_emoji} {status}\n"
        f"Current Step ({current_step_index + 1}/{len(steps)}): {step_name}\n"
        f"Progress: {progress_pct:.0f}%\n"
        f"Steps: {' → '.join(steps)}"
    )


def should_refine_search(
    search_results_quality: float,
    iteration_count: int
) -> bool:
    """Determine if search results are poor enough to warrant refinement.

    Args:
        search_results_quality: LLM-judged quality score (0.0-1.0)
        iteration_count: Current iteration number

    Returns:
        True if should try refining the query
    """
    return (
        search_results_quality < 0.5
        and iteration_count < MAX_REFINEMENT_ITERATIONS
    )


def generate_refined_query(
    original_query: str,
    error_context: dict,
    search_feedback: str
) -> str:
    """Generate a refined query based on feedback from poor results.

    This would be called by the LLM agent with context about why results were bad.
    The actual refinement logic should be implemented by the calling agent
    using its reasoning capabilities.

    Args:
        original_query: The initial search query that produced poor results
        error_context: Dictionary containing error details (error_type,
                       message, stack_trace, environment info)
        search_feedback: Description of why results were inadequate

    Returns:
        A placeholder string - the LLM agent should generate the actual refined query
    """
    return (
        f"REFINEMENT_NEEDED:\n"
        f"Original: {original_query}\n"
        f"Feedback: {search_feedback}\n"
        f"Context keys available: {list(error_context.keys())}\n"
        f"Agent should generate improved query using this context."
    )


@dataclass
class InvestigationPlan:
    """Stateful plan for tracking investigation progress."""

    steps: list[tuple[str, str]]
    current_step_index: int = 0
    iteration_count: int = 0
    refinement_history: list[dict] = field(default_factory=list)

    def get_current_step(self) -> tuple[str, str]:
        """Get the current step (step_id, description)."""
        if self.current_step_index >= len(self.steps):
            return self.steps[-1]
        return self.steps[self.current_step_index]

    def advance(self) -> bool:
        """Advance to next step. Returns True if successful."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return True
        return False

    def record_refinement(
        self,
        original_query: str,
        refined_query: str,
        reason: str
    ) -> None:
        """Record a query refinement for history tracking."""
        self.refinement_history.append({
            "iteration": self.iteration_count,
            "original": original_query,
            "refined": refined_query,
            "reason": reason
        })
        self.iteration_count += 1

    def can_refine(self) -> bool:
        """Check if more refinement iterations are allowed."""
        return self.iteration_count < MAX_REFINEMENT_ITERATIONS

    def get_progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        return ((self.current_step_index + 1) / len(self.steps)) * 100

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        step_id, description = self.get_current_step()
        lines = [
            "Investigation Plan Status",
            "─" * 40,
        ]
        for i, (sid, desc) in enumerate(self.steps):
            marker = "→" if i == self.current_step_index else " "
            status_mark = "[✓]" if i < self.current_step_index else "[ ]"
            lines.append(f"{marker} {status_mark} {i+1}. {sid}: {desc}")

        lines.extend([
            "─" * 40,
            f"Iterations: {self.iteration_count}/{MAX_REFINEMENT_ITERATIONS}",
            f"Progress: {self.get_progress_percentage():.0f}%",
        ])

        if self.refinement_history:
            lines.append(f"Refinements made: {len(self.refinement_history)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert plan state to dictionary for serialization."""
        step_id, description = self.get_current_step()
        return {
            "steps": self.steps,
            "current_step_index": self.current_step_index,
            "current_step": {"id": step_id, "description": description},
            "iteration_count": self.iteration_count,
            "refinement_history": self.refinement_history,
            "progress_percentage": self.get_progress_percentage(),
            "can_refine": self.can_refine()
        }

    @classmethod
    def from_default_steps(cls) -> "InvestigationPlan":
        """Create a plan with default investigation steps."""
        return cls(steps=INVESTIGATION_STEPS.copy())

    def reset(self) -> None:
        """Reset the plan to initial state."""
        self.current_step_index = 0
        self.iteration_count = 0
        self.refinement_history.clear()