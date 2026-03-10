"""Metaflow flow execution helpers.

This module provides functions to run Metaflow flows programmatically
using the Python API, retrieve artifacts from completed runs, and poll
for flow status during execution.
"""

import os
import logging
import traceback
from typing import Optional

os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

from src.utils.langfuse_client import langfuse_trace

logger = logging.getLogger(__name__)


def _investigate_training_error(
    error: Exception,
    tb_str: str,
    task_type: str,
    flow_name: str,
    flow_args: dict,
) -> str:
    """Investigate a training error and return recommendations.

    Returns:
        Investigation text with recommendations for the user.
    """
    investigation_results = []
    
    try:
        from src.agent.error_investigator import investigate_error
        
        print("\n" + "=" * 60)
        print("⚠️  TRAINING ERROR OCCURRED - Starting Error Investigation")
        print("=" * 60)
        
        for message in investigate_error(
            error=error,
            traceback_str=tb_str,
            task_type=task_type,
            flow_name=flow_name,
            flow_args=flow_args,
            data_path=flow_args.get("data_path"),
            training_method=flow_args.get("training_method"),
            base_model=flow_args.get("base_model"),
        ):
            print(message)
            investigation_results.append(message)
            
    except Exception as e:
        error_msg = f"[ErrorInvestigator] Failed to investigate error: {e}"
        print(error_msg)
        investigation_results.append(error_msg)
    
    return "\n".join(investigation_results)


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

    flow_args = {
        "data_path": data_path,
        "target_column": target_column,
    }
    
    logger.info(f"Starting MLTrainingFlow with data_path={data_path}, target_column={target_column}")
    
    try:
        with langfuse_trace("run_ml_training_flow", input=flow_args) as trace:
            with Runner("src/flows/ml_training_flow.py", show_output=False).run(
                data_path=data_path,
                target_column=target_column,
            ) as running:

                if wait_for_completion and running.status == "failed":
                    error_details = running.stderr or "No stderr available"
                    logger.error(f"Flow failed with status: {running.status}")
                    logger.error(f"Error details: {error_details}")

                    tb_str = f"MLTrainingFlow failed: {error_details}"
                    investigation_text = _investigate_training_error(
                        error=RuntimeError(error_details),
                        tb_str=tb_str,
                        task_type="ml_training",
                        flow_name="MLTrainingFlow",
                        flow_args=flow_args,
                    )

                    if trace:
                        trace.update(output={"status": "failed", "error": error_details[:500]})

                    raise RuntimeError(
                        f"MLTrainingFlow failed.\n"
                        f"Data path: {data_path}\n"
                        f"Target column: {target_column}\n"
                        f"Error: {error_details}"
                        + (f"\n\nInvestigation Results:\n{investigation_text}" if investigation_text else "")
                    )

                pathspec = f"{running.run.pathspec}"
                logger.info(f"Flow completed successfully, pathspec: {pathspec}")
                if trace:
                    trace.update(output={"status": "completed", "run_id": pathspec})
                return pathspec
            
    except RuntimeError:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Unexpected error running MLTrainingFlow: {e}")
        
        investigation_text = _investigate_training_error(
            error=e,
            tb_str=tb_str,
            task_type="ml_training",
            flow_name="MLTrainingFlow",
            flow_args=flow_args,
        )
        
        raise RuntimeError(
            f"Failed to execute MLTrainingFlow: {e}"
            + (f"\n\nInvestigation Results:\n{investigation_text}" if investigation_text else "")
        )


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

    flow_args = {
        "data_path": data_path,
        "training_method": training_method,
        "base_model": base_model,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    
    if reward_template:
        flow_args["reward_template"] = reward_template

    try:
        with langfuse_trace("run_llm_training_flow", input=flow_args) as trace:
            with Runner("src/flows/llm_training_flow.py", show_output=False).run(
                **flow_args
            ) as running:
                if wait_for_completion and running.status == "failed":
                    error_details = running.stderr or "No stderr available"

                    tb_str = f"LLMTrainingFlow failed: {error_details}"
                    investigation_text = _investigate_training_error(
                        error=RuntimeError(error_details),
                        tb_str=tb_str,
                        task_type="llm_training",
                        flow_name="LLMTrainingFlow",
                        flow_args=flow_args,
                    )

                    if trace:
                        trace.update(output={"status": "failed", "error": error_details[:500]})

                    raise RuntimeError(
                        f"Flow failed: {error_details}"
                        + (f"\n\nInvestigation Results:\n{investigation_text}" if investigation_text else "")
                    )

                pathspec = f"{running.run.pathspec}"
                if trace:
                    trace.update(output={"status": "completed", "run_id": pathspec})
                return pathspec
            
    except RuntimeError:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        
        investigation_text = _investigate_training_error(
            error=e,
            tb_str=tb_str,
            task_type="llm_training",
            flow_name="LLMTrainingFlow",
            flow_args=flow_args,
        )
        
        raise RuntimeError(
            f"Failed to execute LLMTrainingFlow: {e}"
            + (f"\n\nInvestigation Results:\n{investigation_text}" if investigation_text else "")
        )


def get_flow_artifacts(run_id: str) -> dict:
    """Get artifacts from a completed flow run using the Metaflow Python API.

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

    try:
        flow = Flow(flow_name)
        run = flow[run_number]
    except KeyError:
        raise ValueError(f"Run not found: {flow_name}/{run_number}")

    # Read artifacts from the end step (has all accumulated self.* attributes)
    try:
        end_task = run["end"].task
        data = end_task.data
    except Exception as e:
        raise ValueError(f"Could not read artifacts from run {run_id}: {e}")

    artifacts = {}

    if flow_name == "MLTrainingFlow":
        artifacts["model_path"] = getattr(data, "model_path", "")
        artifacts["task_type"] = getattr(data, "task_type", "classification")
        artifacts["metrics"] = getattr(data, "metrics", {})

    elif flow_name == "LLMTrainingFlow":
        artifacts["model_path"] = getattr(data, "model_path", "")
        artifacts["training_method"] = getattr(data, "training_method", "SFT")
        artifacts["base_model"] = getattr(data, "base_model", "")
        artifacts["metrics"] = getattr(data, "metrics", {})

    if not artifacts.get("model_path"):
        raise ValueError(f"Model not found for run {flow_name}/{run_number}")

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
