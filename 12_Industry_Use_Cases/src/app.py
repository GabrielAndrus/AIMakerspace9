import json
import os
import traceback
import typing
from pathlib import Path
from typing import Generator, Optional

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.config import settings

# Metaflow's vendored typing_extensions monkey-patches typing._collect_parameters,
# which breaks langsmith's class definitions (used transitively by langgraph).
# Save the original and restore it after metaflow imports.
_original_collect_parameters = typing._collect_parameters
from src.flows.runner import get_flow_artifacts, run_llm_training_flow, run_ml_training_flow
from src.flows.runner import _investigate_training_error
typing._collect_parameters = _original_collect_parameters
from src.ml.auto_ensemble import evaluate_model, train_ensemble
from src.ml.data_validator import detect_task_type, get_column_types, validate_csv
from src.job_queue.job_manager import JobManager
from src.utils.error_handling import format_exception_for_user, create_user_error
from src.utils.model_packaging import create_downloadable_zip


def _investigate_inference_error(
    error: Exception,
    tb_str: str,
    task_type: str,
    model_path: Optional[str] = None,
    prompt: Optional[str] = None,
):
    """Investigate an inference error and print recommendations."""
    try:
        from src.agent.error_investigator import investigate_error
        
        print("\n" + "=" * 60)
        print("⚠️  INFERENCE ERROR OCCURRED - Starting Error Investigation")
        print("=" * 60)
        
        for message in investigate_error(
            error=error,
            traceback_str=tb_str,
            task_type=task_type,
            model_path=model_path,
        ):
            print(message)
            
    except Exception as e:
        print(f"[ErrorInvestigator] Failed to investigate error: {e}")

job_manager = JobManager(db_path=settings.JOB_DB_PATH)

BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def create_sample_csv() -> str:
    features, target = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        n_classes=3,
        random_state=42,
    )

    feature_names = [
        "age",
        "income",
        "credit_score",
        "account_balance",
        "transaction_count",
        "loan_amount",
        "employment_years",
        "debt_ratio",
    ]

    df = pd.DataFrame(features, columns=feature_names)
    class_labels = ["Low Risk", "Medium Risk", "High Risk"]
    df["risk_category"] = [class_labels[label] for label in target]

    sample_path = os.path.join(settings.DATA_DIR, "sample_credit_risk.csv")
    df.to_csv(sample_path, index=False)

    return sample_path


def format_error_html(message: str) -> str:
    return f'<div style="background-color: #fee2e2 !important; border-left: 4px solid #ef4444; padding: 12px; border-radius: 4px; color: #991b1b !important;">{message}</div>'


def format_warning_html(message: str) -> str:
    return f'<div style="background-color: #fef3c7 !important; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; color: #92400e !important;">{message}</div>'


def format_success_html(message: str) -> str:
    return f'<div style="background-color: #d1fae5 !important; border-left: 4px solid #10b981; padding: 12px; border-radius: 4px; color: #065f46 !important;">{message}</div>'


def _run_error_investigation(error, tb_str, task_type, flow_name, flow_args, data_path=None):
    """Run the error investigator and return the investigation text."""
    try:
        from src.agent.error_investigator import investigate_error
        import traceback as tb_module
        results = []
        for msg in investigate_error(
            error=error,
            traceback_str=tb_str,
            task_type=task_type,
            flow_name=flow_name,
            flow_args=flow_args,
            data_path=data_path or flow_args.get("data_path"),
            training_method=flow_args.get("training_method"),
            base_model=flow_args.get("base_model"),
        ):
            results.append(msg)
        return "\n".join(results)
    except Exception as inv_err:
        import traceback as tb_module
        inv_tb = tb_module.format_exc()
        print(f"[ErrorInvestigation] Investigation failed: {inv_err}\n{inv_tb}")
        return f"Error investigation failed: {inv_err}\n\nInvestigation traceback:\n{inv_tb}"


def _build_error_html(user_message, investigation_text, tb_str):
    """Build a consistent error HTML block with investigation results and traceback."""
    return f"""
<div style="background-color: #fee2e2 !important; border-left: 4px solid #ef4444; padding: 12px; color: #991b1b !important;">
    <h3 style="color: #991b1b !important;">Training Failed</h3>
    <p style="color: #991b1b !important;">{user_message.replace(chr(10), '<br>')}</p>
    {f'''
    <hr style="margin: 12px 0;">
    <details open>
        <summary style="cursor: pointer; font-weight: bold; color: #991b1b !important;">🔍 Error Investigation Results</summary>
        <div style="background-color: #f3f4f6 !important; padding: 12px; margin-top: 8px; border-radius: 4px;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; color: #1f2937 !important; background-color: #f3f4f6 !important;">{investigation_text}</pre>
        </div>
    </details>
    ''' if investigation_text else ''}
    <hr style="margin: 12px 0;">
    <details>
        <summary style="cursor: pointer; color: #991b1b !important;">View Technical Details</summary>
        <pre style="background-color: #f3f4f6 !important; padding: 12px; overflow-x: auto; margin-top: 8px; color: #1f2937 !important;">{tb_str}</pre>
    </details>
</div>"""


def handle_csv_upload(
    filepath: Optional[str],
) -> tuple[gr.Dataframe, gr.JSON, str, str, str, gr.Dropdown]:
    if filepath is None:
        return (
            gr.Dataframe(value=None),
            gr.JSON(value={}),
            "",
            "",
            "",
            gr.Dropdown(choices=[], value=None),
        )

    result = validate_csv(filepath)
    if not result["valid"]:
        return (
            gr.Dataframe(value=None),
            gr.JSON(value={}),
            format_error_html(result["message"]),
            "",
            "",
            gr.Dropdown(choices=[], value=None),
        )

    df = pd.read_csv(filepath)
    preview_df = df.head(10)

    column_types = get_column_types(df)
    type_summary = {}
    for col, col_type in column_types.items():
        if col_type not in type_summary:
            type_summary[col_type] = []
        type_summary[col_type].append(col)

    columns = result["columns"]
    return (
        gr.Dataframe(value=preview_df),
        gr.JSON(value=type_summary),
        format_success_html("CSV validated successfully"),
        "",
        "",
        gr.Dropdown(choices=columns, value=None),
    )


def handle_target_selection(
    filepath: Optional[str],
    target_column: Optional[str],
) -> tuple[str, str, gr.Plot]:
    if filepath is None or target_column is None:
        return "", "", gr.Plot()

    df = pd.read_csv(filepath)
    y = df[target_column]
    task_type = detect_task_type(y)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if task_type == "classification":
        n_classes = y.nunique()
        class_counts = y.value_counts()

        info_text = f"**Classification** | {n_classes} classes"
        details_text = f"Classes: {', '.join(str(c) for c in class_counts.index[:5])}"
        if n_classes > 5:
            details_text += f" ... and {n_classes - 5} more"

        fig, ax = plt.subplots(figsize=(8, 4))
        class_counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"Class Distribution: {target_column}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return info_text, details_text, gr.Plot(value=fig)

    else:
        value_min = float(y.min())
        value_max = float(y.max())
        value_mean = float(y.mean())

        info_text = "**Regression**"
        details_text = f"Range: [{value_min:.2f}, {value_max:.2f}] | Mean: {value_mean:.2f}"

        fig, ax = plt.subplots(figsize=(8, 4))
        y.hist(ax=ax, bins=30, color="steelblue", edgecolor="black")
        ax.set_title(f"Value Distribution: {target_column}")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        return info_text, details_text, gr.Plot(value=fig)


def train_tabular_model(
    filepath: Optional[str],
    target_column: Optional[str],
    progress: gr.Progress = None,
) -> tuple[str, str, str, str, str, gr.Plot, gr.JSON, gr.Dataframe, str]:
    if progress is None:
        def progress(x, desc=""):
            pass

    if filepath is None:
        return (
            format_error_html("Please upload a CSV file first."),
            gr.File(),
            "",
            "",
            {},
            gr.Plot(),
            gr.Plot(),
            "",
        )

    if target_column is None or target_column == "":
        return (
            format_error_html("Please select a target column."),
            gr.File(),
            "",
            "",
            {},
            gr.Plot(),
            gr.Plot(),
            "",
        )

    progress(0.05, desc="[1/5] Validating data...")

    result = validate_csv(filepath, target_column)
    if not result["valid"]:
        investigation_text = _run_error_investigation(
            error=ValueError(result["message"]),
            tb_str=result["message"],
            task_type="data_validation",
            flow_name="MLTrainingFlow",
            flow_args={"data_path": filepath, "target_column": target_column},
        )

        error_html = _build_error_html(
            f"Data Validation Failed: {result['message']}",
            investigation_text,
            result["message"],
        )

        return (
            error_html,
            gr.File(),
            "",
            "",
            {},
            gr.Plot(),
            gr.Plot(),
            "",
        )

    progress(0.15, desc="[2/5] Analyzing features...")

    df = pd.read_csv(filepath)
    y = df[target_column]
    task_type = detect_task_type(y)

    progress(0.25, desc="[3/5] Submitting to Metaflow...")

    job_id = job_manager.submit_job(
        job_type="ml_training",
        params={
            "data_path": filepath,
            "target_column": target_column,
            "task_type": task_type,
        },
    )

    progress(0.35, desc="[4/5] Training via Metaflow flow...")

    try:
        job_manager.start_job(job_id)

        # Run ML training via Metaflow
        run_id = run_ml_training_flow(
            data_path=filepath,
            target_column=target_column,
            wait_for_completion=True,
        )

        # Get artifacts from the completed run
        training_result = get_flow_artifacts(run_id)
        
        model_path = training_result["model_path"]
        task_type = training_result.get("task_type", "unknown")
        metrics = training_result["metrics"]

        if not os.path.isdir(model_path):
            raise ValueError(f"model_path is not a valid directory: {model_path}")

        model_file = os.path.join(model_path, "model.joblib")
        if not os.path.isfile(model_file):
            raise ValueError(f"model.joblib not found in model_path: {model_path}")

        job_manager.complete_job(job_id, model_path)

        progress(0.85, desc="[5/5] Processing results...")

        if task_type == "classification":
            accuracy = metrics.get("accuracy", 0)
            metric_text = f"Accuracy: {accuracy:.2%}"
        else:
            rmse = metrics.get("rmse", 0)
            metric_text = f"RMSE: {rmse:.4f}"

        feature_importances = {}
        try:
            model = joblib.load(model_file)

            # If model is a Pipeline, extract the actual estimator and transformed feature names
            if hasattr(model, "named_steps"):
                estimator = model.named_steps.get("model", model)
                try:
                    feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
                except Exception:
                    feature_names = list(df.drop(columns=[target_column]).columns)
            else:
                estimator = model
                feature_names = list(df.drop(columns=[target_column]).columns)

            # Clean up feature names (remove prefixes like "num__" and "cat__")
            feature_names = [
                n.split("__", 1)[1] if "__" in n else n for n in feature_names
            ]

            if hasattr(estimator, "feature_importances_"):
                for name, importance in zip(feature_names, estimator.feature_importances_):
                    feature_importances[name] = float(importance)
            elif hasattr(estimator, "estimators_"):
                importances_list = []
                for est in estimator.estimators_:
                    if hasattr(est, "feature_importances_"):
                        importances_list.append(est.feature_importances_)
                if importances_list:
                    mean_importances = np.mean(importances_list, axis=0)
                    for name, importance in zip(feature_names, mean_importances):
                        feature_importances[name] = float(importance)
            elif hasattr(estimator, "coef_"):
                coef = np.abs(estimator.coef_)
                if coef.ndim == 1:
                    for name, importance in zip(feature_names, coef):
                        feature_importances[name] = float(importance)
                else:
                    mean_coef = np.mean(coef, axis=0)
                    for name, importance in zip(feature_names, mean_coef):
                        feature_importances[name] = float(importance)
        except Exception:
            pass

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        if feature_importances:
            sorted_fi = dict(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            ax.barh(list(sorted_fi.keys()), list(sorted_fi.values()), color="steelblue")
            ax.set_xlabel("Importance")
            ax.set_title("Top 10 Feature Importances")
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, "Feature importances not available", ha="center")
            ax.set_axis_off()

        cm_fig = None
        if task_type == "classification" and "confusion_matrix" in metrics:
            try:
                cm = np.array(metrics["confusion_matrix"])
                if cm.ndim == 2 and cm.shape[0] > 0:
                    classes = sorted(y.unique())
                    if len(classes) != cm.shape[0]:
                        classes = [f"Class {i}" for i in range(cm.shape[0])]
                    classes = [str(c) for c in classes]
                    cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
                    im = cm_ax.imshow(cm, interpolation="nearest", cmap="Blues")
                    cm_fig.colorbar(im, ax=cm_ax)
                    cm_ax.set(
                        xticks=range(len(classes)),
                        yticks=range(len(classes)),
                        xticklabels=classes,
                        yticklabels=classes,
                        xlabel="Predicted",
                        ylabel="Actual",
                        title="Confusion Matrix",
                    )
                    thresh = cm.max() / 2.0
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            cm_ax.text(j, i, str(cm[i, j]),
                                       ha="center", va="center",
                                       color="white" if cm[i, j] > thresh else "black")
                    cm_fig.tight_layout()
            except Exception:
                cm_fig = None

        success_msg = format_success_html(f"Training complete! Model saved to: {model_file}")

        return (
            success_msg,
            model_file,
            metric_text,
            task_type.capitalize(),
            {k: v for k, v in metrics.items() if k != "confusion_matrix"},
            gr.Plot(value=fig),
            gr.Plot(value=cm_fig) if cm_fig is not None else gr.Plot(),
            model_file,
        )

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()

        job_manager.fail_job(job_id, str(e))

        error_msg = str(e)
        investigation_text = ""

        if "Investigation Results:" in error_msg:
            parts = error_msg.split("Investigation Results:", 1)
            investigation_text = parts[1].strip()

        if not investigation_text:
            investigation_text = _run_error_investigation(
                error=e, tb_str=tb_str, task_type="ml_training",
                flow_name="MLTrainingFlow",
                flow_args={"data_path": filepath, "target_column": target_column},
            )

        user_friendly_message = format_exception_for_user(
            e,
            context={
                "data_path": filepath,
                "target_column": target_column,
            }
        )

        error_html = _build_error_html(user_friendly_message, investigation_text, tb_str)

        return (
            error_html,
            gr.File(),
            "",
            "",
            {},
            gr.Plot(),
            gr.Plot(),
            "",
        )


def load_model_for_inference(
    model_path: Optional[str],
) -> tuple[str, gr.JSON]:
    if model_path is None or model_path == "":
        return format_warning_html("Please provide a model path."), gr.JSON(value={})

    if not os.path.exists(model_path):
        return format_error_html(f"Model not found at: {model_path}"), gr.JSON(value={})

    try:
        model = joblib.load(model_path)

        model_type = type(model).__name__
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            actual_model = model.named_steps["model"]
            model_type = f"Pipeline({type(actual_model).__name__})"

        info = {
            "model_type": model_type,
            "path": model_path,
        }

        # Load metadata if available
        metadata_path = Path(model_path).parent / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            info["feature_columns"] = metadata.get("feature_columns", [])
            info["task_type"] = metadata.get("task_type", "")
            if metadata.get("description"):
                info["description"] = metadata["description"]

        return (
            format_success_html(f"Model loaded: {model_type}"),
            gr.JSON(value=info),
        )

    except Exception as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            error=e, tb_str=tb_str, task_type="ml_inference",
            flow_name="ModelLoading",
            flow_args={"model_path": model_path},
        )
        return _build_error_html(f"Failed to load model: {str(e)}", investigation_text, tb_str), gr.JSON(value={})


def _prepare_input_for_model(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """Prepare raw input data for model prediction.

    For Pipeline models (new): selects feature columns from metadata, passes
    raw data through — the Pipeline handles preprocessing internally.
    For legacy models (no Pipeline): applies manual one-hot encoding and
    column alignment as a fallback.
    """
    import re

    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        return df

    metadata = json.loads(metadata_path.read_text())
    expected_features = metadata.get("feature_columns", [])
    target_column = metadata.get("target_column", "")

    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    # If the model is a Pipeline with ColumnTransformer, just select
    # the expected feature columns — the Pipeline handles the rest.
    if expected_features:
        # Add any missing expected columns with NaN (Pipeline imputer handles it)
        for col in expected_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected_features]

    return df


def predict_with_model(model_path: str, input_data: str) -> str:
    if not model_path or not input_data:
        return format_warning_html("Please provide both model path and input data.")

    try:
        features = json.loads(input_data)
        model = joblib.load(model_path)
        df = pd.DataFrame([features])

        model_dir = str(Path(model_path).parent)
        df = _prepare_input_for_model(df, model_dir)

        prediction = model.predict(df)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            confidence = max(proba)
            return format_success_html(f"Prediction: {prediction} | Confidence: {confidence:.2%}")

        return format_success_html(f"Prediction: {prediction}")

    except json.JSONDecodeError:
        return format_error_html("Invalid JSON input. Please provide valid JSON.")

    except Exception as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            error=e, tb_str=tb_str, task_type="ml_inference",
            flow_name="Inference",
            flow_args={"model_path": model_path, "input_data": input_data[:500]},
        )
        return _build_error_html(f"Prediction failed: {str(e)}", investigation_text, tb_str)


def create_tabular_ml_tab() -> gr.Tab:
    with gr.Tab("Tabular ML") as tab:
        gr.Markdown("# Tabular ML Training")
        gr.Markdown("Upload a CSV file and select the target column to train an ensemble model.")

        with gr.Row():
            sample_btn = gr.DownloadButton(
                label="Download Sample Dataset",
                variant="secondary",
            )

        def download_sample():
            return create_sample_csv()

        sample_btn.click(fn=download_sample, outputs=[sample_btn])

        with gr.Row():
            csv_file = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath",
            )

        with gr.Accordion("Data Preview", open=True):
            data_preview = gr.Dataframe(label="First 10 Rows")
            with gr.Row():
                column_types_json = gr.JSON(label="Column Types")

        validation_status = gr.HTML()

        with gr.Row():
            target_dropdown = gr.Dropdown(
                label="Target Column",
                choices=[],
                interactive=True,
            )

        with gr.Accordion("Task Type Analysis", open=True):
            with gr.Row():
                task_type_info = gr.Markdown(
                    label="Detected Task Type",
                )
                task_type_details = gr.Textbox(
                    label="Details",
                    interactive=False,
                )
            target_distribution_plot = gr.Plot(label="Target Distribution")

        csv_file.change(
            fn=handle_csv_upload,
            inputs=[csv_file],
            outputs=[
                data_preview,
                column_types_json,
                validation_status,
                task_type_info,
                task_type_details,
                target_dropdown,
            ],
        )

        target_dropdown.change(
            fn=handle_target_selection,
            inputs=[csv_file, target_dropdown],
            outputs=[
                task_type_info,
                task_type_details,
                target_distribution_plot,
            ],
        )

        train_btn = gr.Button("Train Model", variant="primary")

        with gr.Accordion("Training Progress", open=True):
            training_status = gr.HTML()
            with gr.Row():
                task_type_display = gr.Textbox(
                    label="Task Type",
                    interactive=False,
                )
                metric_card = gr.Textbox(
                    label="Primary Metric",
                    interactive=False,
                )

        with gr.Accordion("Model Results", open=True):
            model_download = gr.File(
                label="Download Trained Model",
                interactive=False,
            )
            with gr.Row():
                metrics_json = gr.JSON(label="All Metrics")
                feature_importance_plot = gr.Plot(label="Feature Importances")

            confusion_matrix_display = gr.Plot(label="Confusion Matrix")

        with gr.Row():
            inference_model_path = gr.Textbox(
                label="Model Path for Inference",
                interactive=True,
            )

        train_btn.click(
            fn=train_tabular_model,
            inputs=[csv_file, target_dropdown],
            outputs=[
                training_status,
                model_download,
                metric_card,
                task_type_display,
                metrics_json,
                feature_importance_plot,
                confusion_matrix_display,
                inference_model_path,
            ],
        )

    return tab


def handle_llm_dataset_upload(
    filepath: Optional[str],
) -> tuple[gr.JSON, str, str, gr.Dropdown]:
    if filepath is None:
        return (
            gr.JSON(value={}),
            "",
            "",
            gr.Dropdown(choices=["SFT", "DPO", "GRPO"], value="SFT"),
        )
    
    try:
        from src.agent.dataset_analyzer import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer()
        analysis = analyzer.analyze(filepath)
        
        sample_preview = {
            "file_type": analysis["file_type"],
            "detected_format": analysis["detected_format"],
            "sample_data": analysis["sample_data"][:3] if isinstance(analysis["sample_data"], list) else str(analysis["sample_data"])[:500],
        }
        
        report_html = analyzer.generate_report(analysis)
        
        recommended = analysis["recommended_method"]
        method_match_msg = format_success_html(f"Analysis complete. Recommended method: {recommended}")
        
        return (
            gr.JSON(value=sample_preview),
            report_html,
            method_match_msg,
            gr.Dropdown(choices=["SFT", "DPO", "GRPO"], value=recommended),
        )
    
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()

        investigation_text = _run_error_investigation(
            error=e, tb_str=tb_str, task_type="dataset_analysis",
            flow_name="DatasetAnalyzer",
            flow_args={"data_path": filepath},
        )

        error_html = _build_error_html(
            f"Dataset analysis failed: {str(e)}",
            investigation_text,
            tb_str,
        )

        return (
            gr.JSON(value={}),
            error_html,
            "",
            gr.Dropdown(choices=["SFT", "DPO", "GRPO"], value="SFT"),
        )


def train_llm_model(
    train_file: Optional[str],
    training_method: str,
    base_model: str,
    epochs: int,
    learning_rate: float,
    reward_template: Optional[str],
    custom_reward_code: str,
) -> tuple[str, gr.File]:
    if train_file is None:
        return (
            format_error_html("Please upload a training file first."),
            gr.File(),
        )

    analysis_report = ""
    recommended_method = training_method
    
    try:
        from src.agent.dataset_analyzer import get_training_recommendation
        
        recommended_method, analysis_report = get_training_recommendation(train_file)
        
        if training_method != recommended_method:
            analysis_report += f"""
<div style="background-color: #fef3c7; padding: 12px; border-radius: 4px; margin-top: 12px;">
<b>Note:</b> Agent recommended <b>{recommended_method}</b>, but you selected <b>{training_method}</b>.
Your selection will be used.
</div>"""
    except Exception as e:
        analysis_report = format_warning_html(f"Could not analyze dataset: {str(e)}")

    try:
        run_id = run_llm_training_flow(
            data_path=train_file,
            training_method=training_method,
            base_model=base_model,
            epochs=epochs,
            learning_rate=learning_rate,
            reward_template=reward_template if training_method == "GRPO" else None,
        )
        
        artifacts = get_flow_artifacts(run_id)
        model_path = artifacts.get("model_path", "")
        
        zip_path = create_downloadable_zip(model_path)
        
        success_msg = f"""
{analysis_report}
<hr>
<div style="background-color: #d1fae5; border-left: 4px solid #10b981; padding: 12px; margin-top: 12px;">
<b>Training Complete!</b><br>
Run ID: {run_id}<br>
Model saved to: {model_path}
</div>"""
        
        return (success_msg, zip_path)
    
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()

        error_msg = str(e)
        investigation_text = ""

        if "Investigation Results:" in error_msg:
            parts = error_msg.split("Investigation Results:", 1)
            investigation_text = parts[1].strip()

        if not investigation_text:
            investigation_text = _run_error_investigation(
                error=e, tb_str=tb_str, task_type="llm_training",
                flow_name="LLMTrainingFlow",
                flow_args={
                    "data_path": train_file,
                    "training_method": training_method,
                    "base_model": base_model,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                },
            )

        user_friendly_message = format_exception_for_user(
            e,
            context={
                "data_path": train_file,
                "training_method": training_method,
                "base_model": base_model,
            }
        )

        error_html = f"""
{analysis_report}
<hr>
{_build_error_html(user_friendly_message, investigation_text, tb_str)}"""

        return (error_html, gr.File())


def create_llm_finetuning_tab() -> gr.Tab:
    with gr.Tab("LLM Fine-tuning") as tab:
        gr.Markdown("# LLM Fine-tuning")
        gr.Markdown("Upload a dataset and the agent will analyze it with RAG to recommend the best training method.")

        with gr.Row():
            train_file = gr.File(
                label="Upload Training File (JSONL/TXT/PDF)",
                file_types=[".jsonl", ".txt", ".pdf"],
                type="filepath",
            )

        with gr.Accordion("Dataset Preview & Analysis", open=True):
            sample_preview = gr.JSON(label="Sample Data")
            analysis_report = gr.HTML()
        
        validation_status = gr.HTML()

        with gr.Row():
            training_method = gr.Dropdown(
                label="Training Method (auto-selected from analysis)",
                choices=["SFT", "DPO", "GRPO"],
                value="SFT",
                interactive=True,
            )

            base_model = gr.Dropdown(
                label="Base Model",
                choices=BASE_MODELS,
                value=BASE_MODELS[0],
            )

        train_file.change(
            fn=handle_llm_dataset_upload,
            inputs=[train_file],
            outputs=[
                sample_preview,
                analysis_report,
                validation_status,
                training_method,
            ],
        )

        with gr.Accordion("Training Parameters", open=False):
            epochs = gr.Slider(
                label="Epochs",
                minimum=1,
                maximum=10,
                value=3,
                step=1,
            )

            learning_rate = gr.Slider(
                label="Learning Rate",
                minimum=1e-6,
                maximum=1e-3,
                value=2e-4,
                step=1e-6,
            )

        with gr.Accordion(
            "GRPO Reward Configuration", open=False, visible=False
        ) as grpo_reward_accordion:
            gr.Markdown("### Configure Reward Function for GRPO Training")
            gr.Markdown(
                "GRPO (Group Relative Policy Optimization) requires a reward function to guide the model. "
                "Choose a built-in template or provide custom Python code."
            )

            reward_template = gr.Dropdown(
                label="Reward Template",
                choices=["math", "format_check"],
                value=None,
                allow_custom_value=False,
            )

            gr.Markdown(
                "**Built-in Templates:**\n"
                "- **math**: Checks if model output contains the correct answer (requires 'ground_truth' field in dataset)\n"
                "- **format_check**: Validates response matches a regex pattern (requires 'pattern' field in dataset)"
            )

            custom_reward_code = gr.Code(
                label="Custom Reward Function (Python)",
                language="python",
                value='def reward_func(completions, **kwargs):\n    # Your reward logic here\n    # Return a list of floats (one reward per completion)\n    return [1.0 if "correct" in c.lower() else 0.0 for c in completions]',
            )

        train_btn = gr.Button("Start Training", variant="primary")

        llm_training_status = gr.HTML()

        adapter_download = gr.File(
            label="Download LoRA Adapter",
            interactive=False,
        )

        training_method.change(
            fn=lambda x: gr.Accordion(visible=(x == "GRPO")),
            inputs=[training_method],
            outputs=[grpo_reward_accordion],
        )

        train_btn.click(
            fn=train_llm_model,
            inputs=[
                train_file,
                training_method,
                base_model,
                epochs,
                learning_rate,
                reward_template,
                custom_reward_code,
            ],
            outputs=[llm_training_status, adapter_download],
        )

    return tab


def create_inference_playground_tab() -> gr.Tab:
    with gr.Tab("Inference Playground") as tab:
        gr.Markdown("# Inference Playground")
        gr.Markdown("Load a trained model and make predictions.")

        with gr.Row():
            inference_model_path = gr.Textbox(
                label="Model Path",
                placeholder="Path to trained model (.joblib)",
            )

            load_btn = gr.Button("Load Model", variant="secondary")

        load_status = gr.HTML()
        model_info_json = gr.JSON(label="Model Information")

        with gr.Row():
            inference_input = gr.Code(
                label="Input Data (JSON)",
                language="json",
                value='{"feature1": "value1", "feature2": 42}',
            )

        predict_btn = gr.Button("Predict", variant="primary")

        prediction_result = gr.HTML()

        load_btn.click(
            fn=load_model_for_inference,
            inputs=[inference_model_path],
            outputs=[load_status, model_info_json],
        )

        predict_btn.click(
            fn=predict_with_model,
            inputs=[inference_model_path, inference_input],
            outputs=[prediction_result],
        )

    return tab


# ============================================================================
# Local LLM Inference with LoRA Adapters
# ============================================================================

LLM_BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Llama-3.2-1B-Instruct",
]


def load_lora_adapter(
    lora_path: Optional[str],
    base_model_override: Optional[str],
) -> tuple[gr.HTML, gr.Dropdown]:
    """Load LoRA adapter and return status.

    Args:
        lora_path: Path to LoRA adapter directory
        base_model_override: Optional manual override for base model

    Returns:
        Tuple of (status HTML, updated dropdown)
    """
    from src.llm.local_inference import get_local_inference, validate_lora_adapter

    if not lora_path:
        return format_warning_html("No adapter path provided."), gr.Dropdown()

    # Validate the adapter
    validation = validate_lora_adapter(lora_path)

    if not validation["valid"]:
        error_msg = "Invalid adapter: " + "; ".join(validation["issues"])
        return format_error_html(error_msg), gr.Dropdown()

    detected_base = validation.get("base_model")

    # Use override if provided, otherwise use detected
    base_model = base_model_override or detected_base

    if not base_model:
        return (
            format_warning_html("Could not auto-detect base model. Please select one manually."),
            gr.Dropdown(choices=LLM_BASE_MODELS, value=None),
        )

    try:
        inferencer = get_local_inference()
        inferencer.load_trained_lora(lora_path, base_model=base_model)

        # Build info message
        msg_parts = [f"Model loaded successfully!", f"Base model: {base_model}"]
        if validation["warnings"]:
            msg_parts.append(f"Warnings: {'; '.join(validation['warnings'])}")

        return (
            format_success_html("<br>".join(msg_parts)),
            gr.Dropdown(choices=LLM_BASE_MODELS, value=base_model),
        )

    except FileNotFoundError as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            e, tb_str, "llm_inference", "LLMInference", {"lora_path": lora_path, "base_model": base_model}
        )
        return _build_error_html(f"File not found: {e}", investigation_text, tb_str), gr.Dropdown()
    except ValueError as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            e, tb_str, "llm_inference", "LLMInference", {"lora_path": lora_path, "base_model": base_model}
        )
        return _build_error_html(f"Invalid configuration: {e}", investigation_text, tb_str), gr.Dropdown()
    except Exception as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            e, tb_str, "llm_inference", "LLMInference", {"lora_path": lora_path, "base_model": base_model}
        )
        return _build_error_html(f"Error loading model: {e}", investigation_text, tb_str), gr.Dropdown()


def generate_with_lora(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate response using loaded LoRA adapter.

    Args:
        prompt: User input text
        system_prompt: System instruction
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text or error message
    """
    from src.llm.local_inference import get_local_inference

    inferencer = get_local_inference()

    if inferencer.model is None:
        return format_warning_html("Please load a LoRA adapter first.")

    if not prompt or not prompt.strip():
        return format_warning_html("Please enter a prompt.")

    try:
        response = inferencer.generate(
            prompt=prompt,
            system_prompt=system_prompt if system_prompt else None,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return response

    except RuntimeError as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            e, tb_str, "llm_inference", "LLMInference", {"prompt": prompt[:200] if prompt else ""}
        )
        return _build_error_html(f"Generation error: {e}", investigation_text, tb_str)
    except Exception as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            e, tb_str, "llm_inference", "LLMInference", {"prompt": prompt[:200] if prompt else ""}
        )
        return _build_error_html(f"Unexpected error: {e}", investigation_text, tb_str)


def generate_with_lora_streaming(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Generate streaming response using loaded LoRA adapter.

    Args:
        prompt: User input text
        system_prompt: System instruction
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Yields:
        Generated text chunks as they are produced
    """
    from src.llm.local_inference import get_local_inference

    inferencer = get_local_inference()

    if inferencer.model is None:
        yield format_warning_html("Please load a LoRA adapter first.")
        return

    if not prompt or not prompt.strip():
        yield format_warning_html("Please enter a prompt.")
        return

    try:
        for chunk in inferencer.generate_streaming(
            prompt=prompt,
            system_prompt=system_prompt if system_prompt else None,
            max_new_tokens=max_tokens,
            temperature=temperature,
        ):
            yield chunk

    except RuntimeError as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            error=e, tb_str=tb_str, task_type="llm_inference",
            flow_name="LLMInference",
            flow_args={"prompt": prompt[:200]},
        )
        yield _build_error_html(f"Generation error: {e}", investigation_text, tb_str)
    except Exception as e:
        tb_str = traceback.format_exc()
        investigation_text = _run_error_investigation(
            error=e, tb_str=tb_str, task_type="llm_inference",
            flow_name="LLMInference",
            flow_args={"prompt": prompt[:200]},
        )
        yield _build_error_html(f"Unexpected error: {e}", investigation_text, tb_str)


def merge_lora_adapter_wrapper(
    lora_path: Optional[str],
    output_dir: str,
    base_model_override: Optional[str],
) -> gr.HTML:
    """Merge LoRA adapter into base model.

    Args:
        lora_path: Path to LoRA adapter
        output_dir: Output directory for merged model
        base_model_override: Optional manual override for base model

    Returns:
        Status HTML message
    """
    from src.llm.local_inference import merge_lora_to_base, validate_lora_adapter

    if not lora_path:
        return format_warning_html("No adapter path provided.")

    # Validate first
    validation = validate_lora_adapter(lora_path)
    if not validation["valid"]:
        return format_error_html("Invalid adapter: " + "; ".join(validation["issues"]))

    try:
        merged_path = merge_lora_to_base(
            lora_path=lora_path,
            output_dir=output_dir,
            base_model=base_model_override or validation.get("base_model"),
        )
        return format_success_html(f"Merge complete! Merged model saved to: {merged_path}")

    except Exception as e:
        return format_error_html(f"Merge failed: {e}")


def validate_lora_adapter_wrapper(lora_path: Optional[str]) -> gr.HTML:
    """Validate LoRA adapter and show detailed results.

    Args:
        lora_path: Path to LoRA adapter directory

    Returns:
        Validation results as HTML
    """
    from src.llm.local_inference import validate_lora_adapter

    if not lora_path:
        return format_warning_html("No adapter path provided.")

    result = validate_lora_adapter(lora_path)

    if not result["valid"]:
        html = format_error_html(f"Invalid adapter: {lora_path}")
        html += "<br><strong>Issues:</strong><ul>"
        for issue in result["issues"]:
            html += f"<li>{issue}</li>"
        html += "</ul>"
        return html

    html = format_success_html(f"Valid adapter: {lora_path}")
    if result["base_model"]:
        html += f"<br><strong>Base Model:</strong> {result['base_model']}"
    if result["warnings"]:
        html += "<br><strong>Warnings:</strong><ul>"
        for warning in result["warnings"]:
            html += f"<li>{warning}</li>"
        html += "</ul>"

    return html


def run_ragas_evaluation(retrieval_method: str = "dense", compare_methods: bool = False) -> tuple[gr.HTML, gr.Dataframe]:
    """Run RAGAS evaluation and return results."""
    from src.evaluation.ragas_evaluator import RAGASEvaluator

    try:
        evaluator = RAGASEvaluator()

        if compare_methods:
            results = evaluator.compare_retrieval_methods()
            
            if "error" in results.get("comparison", {}).get("dense", {}):
                return format_error_html(f"Evaluation failed: {results['comparison']['dense'].get('error', 'Unknown error')}"), pd.DataFrame()
            
            summary = results["summary"]
            
            comparison_table_data = []
            for metric in summary["metrics"]:
                dense_score = summary["dense_scores"][metric]
                hybrid_score = summary["hybrid_scores"][metric]
                improvement = ((hybrid_score - dense_score) / dense_score * 100) if dense_score > 0 else 0
                
                comparison_table_data.append({
                    "Metric": metric,
                    "Dense": f"{dense_score:.4f}",
                    "Hybrid": f"{hybrid_score:.4f}",
                    "Improvement": f"{improvement:+.1f}%"
                })
            
            df_comparison = pd.DataFrame(comparison_table_data)
            
            summary_html = f"""
<div style="background-color: #d1fae5; border-left: 4px solid #10b981; padding: 12px; margin-top: 12px;">
<b>Comparison Complete!</b><br>
Compared Dense vs Hybrid retrieval methods<br><br>
<b>Average Scores:</b><br>
- Faithfulness: {summary['hybrid_scores']['faithfulness']:.4f} (Hybrid)<br>
- Context Precision: {summary['hybrid_scores']['context_precision']:.4f} (Hybrid)<br>
- Context Recall: {summary['hybrid_scores']['context_recall']:.4f} (Hybrid)
</div>"""

            return summary_html, df_comparison
        
        else:
            evaluator.set_retrieval_method(retrieval_method)
            results = evaluator.evaluate_dataset()

            if "error" in results:
                return format_error_html(f"Evaluation failed: {results['error']}"), pd.DataFrame()

            avg_scores = results["average_scores"]
            
            summary_html = f"""
<div style="background-color: #d1fae5; border-left: 4px solid #10b981; padding: 12px; margin-top: 12px;">
<b>Evaluation Complete!</b><br>
Retrieval Method: {retrieval_method.upper()}<br>
Total Questions: {results['total_evaluated']}<br><br>
<b>Average Scores:</b><br>
- Faithfulness: {avg_scores['faithfulness']:.4f}<br>
- Context Precision: {avg_scores['context_precision']:.4f}<br>
- Context Recall: {avg_scores['context_recall']:.4f}
</div>"""

            df = evaluator.results_to_dataframe(results)
            
            return summary_html, df

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        
        error_html = f"""
<div style="background-color: #fee2e2 !important; border-left: 4px solid #ef4444; padding: 12px; color: #991b1b !important;">
<h3 style="color: #991b1b !important;">Evaluation Failed</h3>
<p style="color: #991b1b !important;">{str(e)}</p>
<details>
<summary style="cursor: pointer; margin-top: 12px; color: #991b1b !important;"><b>View Full Traceback</b></summary>
<pre style="background-color: #f3f4f6 !important; padding: 12px; overflow-x: auto; margin-top: 8px; color: #1f2937 !important;">
{tb_str}
</pre>
</details>
</div>"""
        
        return error_html, pd.DataFrame()


def create_ragas_evaluation_tab() -> gr.Tab:
    """Create the RAGAS Evaluation tab."""
    with gr.Tab("RAGAS Evaluation") as tab:
        gr.Markdown("# RAGAS Evaluation")
        gr.Markdown(
            "Evaluate the performance of your RAG pipeline using faithfulness, "
            "context precision, and context recall metrics."
        )

        gr.Markdown(
            """
            **Metrics Explained:**
            - **Faithfulness**: Measures how factually consistent the answer is with the retrieved context
            - **Context Precision**: Measures signal-to-noise ratio in retrieved contexts (are all contexts relevant?)
            - **Context Recall**: Measures if the ground truth answer can be derived from the retrieved contexts
            """
        )

        with gr.Row():
            retrieval_method = gr.Radio(
                label="Retrieval Method",
                choices=["dense", "sparse", "hybrid"],
                value="dense",
                info="Dense: semantic search | Sparse: keyword matching (BM25) | Hybrid: combines both"
            )

        with gr.Row():
            compare_methods = gr.Checkbox(
                label="Compare Methods",
                value=False,
                info="Run evaluation with both Dense and Hybrid retrieval for comparison"
            )

        run_eval_btn = gr.Button("Run Evaluation", variant="primary")

        eval_status = gr.HTML()
        
        results_table = gr.Dataframe(
            label="Evaluation Results",
            interactive=False,
        )

        run_eval_btn.click(
            fn=run_ragas_evaluation,
            inputs=[retrieval_method, compare_methods],
            outputs=[eval_status, results_table],
        )

    return tab


def create_llm_inference_tab() -> gr.Tab:
    """Create the LLM Inference tab with LoRA adapter support."""
    with gr.Tab("LLM Inference") as tab:
        gr.Markdown("# LLM Inference with LoRA Adapters")
        gr.Markdown(
            "Load your trained LoRA adapters and test them locally. "
            "No external inference server required."
        )

        with gr.Row():
            lora_path = gr.Textbox(
                label="LoRA Adapter Path",
                placeholder="/path/to/your/lora_adapter",
                value=os.path.join(settings.MODEL_DIR, "sft_adapter"),
            )

        with gr.Row():
            lora_base_model = gr.Dropdown(
                label="Base Model",
                choices=LLM_BASE_MODELS,
                value=None,
                info="Auto-detected from adapter metadata if not specified",
            )

        load_lora_btn = gr.Button("Load LoRA Adapter", variant="primary")

        lora_load_status = gr.HTML()

        with gr.Accordion("Adapter Validation", open=False):
            validate_btn = gr.Button("Validate Adapter", variant="secondary")
            validation_result = gr.HTML()

        validate_btn.click(
            fn=validate_lora_adapter_wrapper,
            inputs=[lora_path],
            outputs=[validation_result],
        )

        load_lora_btn.click(
            fn=load_lora_adapter,
            inputs=[lora_path, lora_base_model],
            outputs=[lora_load_status, lora_base_model],
        )

        gr.Markdown("## Generate Responses")

        with gr.Row():
            lora_system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are a helpful assistant.",
                lines=2,
            )

        lora_prompt = gr.Textbox(
            label="User Prompt",
            placeholder="Enter your question or instruction here...",
            lines=3,
        )

        with gr.Row():
            lora_max_tokens = gr.Slider(64, 2048, value=512, label="Max Tokens")
            lora_temperature = gr.Slider(
                0.1,
                2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            )

        with gr.Row():
            lora_generate_btn = gr.Button("Generate", variant="primary")
            lora_streaming_btn = gr.Button("Generate (Streaming)", variant="secondary")

        with gr.Accordion("Generation Options", open=False):
            use_streaming = gr.Checkbox(
                label="Enable Streaming",
                value=True,
                info="Show tokens as they are generated",
            )

        lora_output = gr.Textbox(
            label="Generated Response",
            lines=10,
            interactive=False,
        )

        # Non-streaming generation
        lora_generate_btn.click(
            fn=generate_with_lora,
            inputs=[lora_prompt, lora_system_prompt, lora_max_tokens, lora_temperature],
            outputs=[lora_output],
        )

        # Streaming generation
        lora_streaming_btn.click(
            fn=generate_with_lora_streaming,
            inputs=[lora_prompt, lora_system_prompt, lora_max_tokens, lora_temperature],
            outputs=[lora_output],
        )

        gr.Markdown("## Merge LoRA Adapter")

        with gr.Accordion("Merge to Standalone Model", open=False):
            gr.Markdown(
                "Merge your LoRA adapter into the base model for deployment "
                "to systems that don't support PEFT/LoRA loading."
            )

            merge_output_dir = gr.Textbox(
                label="Output Directory",
                placeholder="/path/to/output/merged_model",
            )

            merge_btn = gr.Button("Merge Adapter", variant="secondary")

            merge_result = gr.HTML()

        merge_btn.click(
            fn=merge_lora_adapter_wrapper,
            inputs=[lora_path, merge_output_dir, lora_base_model],
            outputs=[merge_result],
        )

    return tab


with gr.Blocks(title="Agentic AutoML Platform", theme="JohnSmith9982/small_and_pretty") as demo:
    gr.Markdown("# Agentic AutoML Platform")

    create_tabular_ml_tab()
    create_llm_finetuning_tab()
    create_inference_playground_tab()
    create_ragas_evaluation_tab()
    create_llm_inference_tab()


if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=5)
    demo.launch(server_name="0.0.0.0", server_port=7860)
