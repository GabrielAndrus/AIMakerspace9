"""Metaflow flow execution helpers.

This module provides functions to run Metaflow flows programmatically
using the Python API, retrieve artifacts from completed runs, and poll
for flow status during execution.
"""

import os
import logging
from typing import Optional

os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

logger = logging.getLogger(__name__)


def run_ml_training_flow(
    data_path: str,
    target_column: str,
    wait_for_completion: bool = True,
) -> str:
    """Run ML training flow via Metaflow Python API.

    Args:
        data_path: Path to CSV file containing training data
        target_column: Name of the target column to predict
        wait_for_completion: If True, block until flow completes

    Returns:
        Flow run ID (e.g., "MLTrainingFlow/123")

    Raises:
        RuntimeError: If flow fails to execute
    """
    from metaflow import Runner

    logger.info(f"Starting MLTrainingFlow with data_path={data_path}, target_column={target_column}")
    
    try:
        with Runner("src/flows/ml_training_flow.py", show_output=False).run(
            data_path=data_path,
            target_column=target_column,
        ) as running:
            
            if wait_for_completion and running.status == "failed":
                error_details = running.stderr or "No stderr available"
                logger.error(f"Flow failed with status: {running.status}")
                logger.error(f"Error details: {error_details}")
                raise RuntimeError(
                    f"MLTrainingFlow failed.\n"
                    f"Data path: {data_path}\n"
                    f"Target column: {target_column}\n"
                    f"Error: {error_details}"
                )

            logger.info(f"Flow completed successfully, pathspec: {running.run.pathspec}")
            return f"{running.run.pathspec}"
            
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error running MLTrainingFlow: {e}")
        raise RuntimeError(f"Failed to execute MLTrainingFlow: {e}")


def run_llm_training_flow(
    data_path: str,
    training_method: str = "SFT",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    epochs: int = 3,
    learning_rate: float = 2e-4,
    reward_template: str | None = None,
    wait_for_completion: bool = True,
) -> str:
    """Run LLM training flow via Metaflow Python API.

    Args:
        data_path: Path to training file (TXT/PDF/JSONL)
        training_method: Training method ('SFT', 'DPO', or 'GRPO')
        base_model: Base model to fine-tune
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        reward_template: GRPO reward template ('math', 'format_check'). Auto-detected if None.
        wait_for_completion: If True, block until flow completes

    Returns:
        Flow run ID (e.g., "LLMTrainingFlow/123")

    Raises:
        RuntimeError: If flow fails to execute
    """
    from metaflow import Runner

    runner_args = {
        "data_path": data_path,
        "training_method": training_method,
        "base_model": base_model,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    
    if reward_template:
        runner_args["reward_template"] = reward_template

    with Runner("src/flows/llm_training_flow.py", show_output=False).run(
        **runner_args
    ) as running:
        if wait_for_completion and running.status == "failed":
            raise RuntimeError(f"Flow failed: {running.stderr}")

        return f"{running.run.pathspec}"


def get_flow_artifacts(run_id: str) -> dict:
    """Get artifacts from a completed flow run.

    Args:
        run_id: Flow run ID (e.g., "MLTrainingFlow/123")

    Returns:
        Dictionary of artifacts from the run including:
            - model_path: Path to saved model
            - task_type: Type of ML task (classification/regression)
            - metrics: Dictionary of evaluation metrics

    Raises:
        KeyError: If run ID is invalid
        ValueError: If required artifacts are missing
    """
    from metaflow import Flow

    # Parse run_id: "FlowName/run_number" or full pathspec
    parts = run_id.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid run ID format: {run_id}")

    flow_name = parts[0]
    run_number = str(parts[-1])

    # Get the run
    flow = Flow(flow_name)
    run = flow[run_number]

    # Extract common artifacts
    try:
        artifacts = {
            "model_path": run.data.model_path,
        }
    except AttributeError as e:
        raise ValueError(f"Missing required artifact 'model_path': {e}")

    # Extract task-specific artifacts
    if flow_name == "MLTrainingFlow":
        try:
            artifacts.update(
                {
                    "task_type": run.data.task_type,
                    "metrics": run.data.metrics,
                }
            )
        except AttributeError as e:
            raise ValueError(f"Missing ML training artifacts: {e}")
    elif flow_name == "LLMTrainingFlow":
        try:
            artifacts.update(
                {
                    "training_method": run.data.training_method,
                    "base_model": run.data.base_model,
                    "metrics": getattr(run.data, "metrics", {}),
                }
            )
        except AttributeError as e:
            raise ValueError(f"Missing LLM training artifacts: {e}")

    return artifacts


def poll_flow_status(run_id: str) -> dict:
    """Poll a running flow for status and progress.

    Args:
        run_id: Flow run ID

    Returns:
        Status dictionary with:
            - state: Current state ('running', 'completed', 'failed', 'not_found')
            - progress: Progress percentage (0.0 to 1.0)
            - current_step: Name of the current step being executed
    """
    from metaflow import Flow

    # Parse run_id
    parts = run_id.split("/")
    if len(parts) < 2:
        return {"state": "invalid", "progress": 0.0, "current_step": None}

    flow_name = parts[0]
    run_number = str(parts[-1])

    try:
        flow = Flow(flow_name)
        run = flow[run_number]
    except KeyError:
        return {"state": "not_found", "progress": 0.0, "current_step": None}

    # Check if run is finished
    if run.finished:
        return {
            "state": "completed" if run.successful else "failed",
            "progress": 1.0,
            "current_step": "end" if run.successful else "failed",
        }

    # Run is still in progress
    # Try to determine current step by checking task status
    current_step = "unknown"
    progress = 0.5  # Default midpoint for running flows

    try:
        # Get all tasks and find the one that's currently executing
        steps = [
            "start",
            "load_data",
            "validate_data",
            "preprocess",
            "train_model",
            "evaluate",
            "save_model",
        ]

        current_step_idx = 0
        for i, step_name in enumerate(steps):
            try:
                task = run[step_name]
                if task.finished and not task.successful:
                    # Step failed
                    return {
                        "state": "failed",
                        "progress": i / len(steps),
                        "current_step": step_name,
                    }
                if task.finished:
                    current_step_idx = i + 1
                else:
                    # This step is currently running or hasn't started yet
                    current_step = step_name
                    break
            except KeyError:
                # Step doesn't exist (might be from different flow type)
                pass

        current_step = steps[current_step_idx] if current_step_idx < len(steps) else "end"
        progress = current_step_idx / len(steps)

    except Exception:
        # If we can't determine exact step, just return running state
        pass

    return {
        "state": "running",
        "progress": progress,
        "current_step": current_step,
    }


def get_flow_cards(run_id: str) -> list[dict]:
    """Get cards from a completed flow run.

    Args:
        run_id: Flow run ID

    Returns:
        List of card dictionaries with metadata and content
    """
    from metaflow import Flow

    parts = run_id.split("/")
    flow_name = parts[0]
    run_number = str(parts[-1])

    try:
        flow = Flow(flow_name)
        run = flow[run_number]
    except KeyError:
        return []

    cards = []
    for task in run:
        try:
            task_cards = task.get_cards()
            for card in task_cards:
                cards.append(
                    {
                        "type": card.type,
                        "step": task.step_name,
                        "pathspec": card.pathspec,
                    }
                )
        except Exception:
            # Cards might not be available for this task
            continue

    return cards
