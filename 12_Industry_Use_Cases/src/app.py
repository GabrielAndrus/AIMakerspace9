import json
import os
from typing import Optional

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.config import settings
from src.ml.auto_ensemble import evaluate_model, train_ensemble
from src.ml.data_validator import detect_task_type, get_column_types, validate_csv
from src.queue.job_manager import JobManager
from src.utils.serialization import save_model

job_manager = JobManager(db_path=settings.JOB_DB_PATH)

BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def train_ml_model_direct(filepath: str, target_column: str) -> dict:
    result = validate_csv(filepath, target_column)
    if not result["valid"]:
        raise ValueError(result["message"])

    df = pd.read_csv(filepath)
    y = df[target_column]
    x_features = df.drop(columns=[target_column])

    categorical_cols = x_features.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        x_features = pd.get_dummies(x_features, columns=categorical_cols)

    task_type = detect_task_type(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y, test_size=0.2, random_state=42
    )

    training_result = train_ensemble(x_train, y_train, task_type)
    model = training_result["model"]

    metrics = evaluate_model(model, x_test, y_test, task_type)

    model_filename = f"ensemble_{task_type}_model.joblib"
    model_path = os.path.join(settings.MODEL_DIR, model_filename)
    saved_model_path = save_model(model, model_path)

    return {
        "task_type": task_type,
        "metrics": metrics,
        "model_path": saved_model_path,
    }


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
    return f'<div style="background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 12px; border-radius: 4px; color: #991b1b;">{message}</div>'


def format_warning_html(message: str) -> str:
    return f'<div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; color: #92400e;">{message}</div>'


def format_success_html(message: str) -> str:
    return f'<div style="background-color: #d1fae5; border-left: 4px solid #10b981; padding: 12px; border-radius: 4px; color: #065f46;">{message}</div>'


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
    progress: gr.Progress,
) -> tuple[str, str, str, str, str, gr.Plot, gr.JSON, gr.Dataframe]:
    if filepath is None:
        return (
            format_error_html("Please upload a CSV file first."),
            "",
            "",
            "",
            "",
            gr.Plot(),
            gr.JSON(value={}),
            gr.Dataframe(value=None),
        )

    if target_column is None or target_column == "":
        return (
            format_error_html("Please select a target column."),
            "",
            "",
            "",
            "",
            gr.Plot(),
            gr.JSON(value={}),
            gr.Dataframe(value=None),
        )

    progress(0.05, desc="[1/5] Validating data...")

    result = validate_csv(filepath, target_column)
    if not result["valid"]:
        return (
            format_error_html(result["message"]),
            "",
            "",
            "",
            "",
            gr.Plot(),
            gr.JSON(value={}),
            gr.Dataframe(value=None),
        )

    progress(0.15, desc="[2/5] Analyzing features...")

    df = pd.read_csv(filepath)
    y = df[target_column]
    task_type = detect_task_type(y)

    progress(0.25, desc="[3/5] Preparing data pipeline...")

    job_id = job_manager.submit_job(
        job_type="ml_training",
        params={
            "data_path": filepath,
            "target_column": target_column,
            "task_type": task_type,
        },
    )

    progress(0.35, desc="[4/5] Training ensemble model...")

    try:
        job_manager.start_job(job_id)

        training_result = train_ml_model_direct(filepath, target_column)

        model_path = training_result["model_path"]
        task_type = training_result["task_type"]
        metrics = training_result["metrics"]

        job_manager.complete_job(job_id, model_path)

        progress(0.85, desc="[5/5] Evaluating model...")

        if task_type == "classification":
            accuracy = metrics.get("accuracy", 0)
            metric_text = f"Accuracy: {accuracy:.2%}"
        else:
            rmse = metrics.get("rmse", 0)
            metric_text = f"RMSE: {rmse:.4f}"

        feature_importances = {}
        try:
            model = joblib.load(model_path)
            if hasattr(model, "feature_importances_"):
                features_df = df.drop(columns=[target_column])
                for name, importance in zip(features_df.columns, model.feature_importances_):
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

        confusion_data = None
        if task_type == "classification" and "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            classes = (
                list(y.unique())
                if len(list(y.unique())) == cm.shape[0]
                else [f"Class {i}" for i in range(cm.shape[0])]
            )
            confusion_data = pd.DataFrame(cm, index=classes, columns=classes)

        success_msg = format_success_html(f"Training complete! Model saved to: {model_path}")

        return (
            success_msg,
            model_path,
            metric_text,
            task_type.capitalize(),
            json.dumps(metrics, indent=2),
            gr.Plot(value=fig),
            gr.JSON(value=metrics),
            confusion_data if confusion_data is not None else pd.DataFrame(),
        )

    except Exception as e:
        job_manager.fail_job(job_id, str(e))
        return (
            format_error_html(f"Training failed: {str(e)}"),
            "",
            "",
            "",
            "",
            gr.Plot(),
            gr.JSON(value={}),
            pd.DataFrame(),
        )


def train_llm_model(
    filepath: Optional[str],
    training_method: str,
    base_model: str,
    epochs: int,
    learning_rate: float,
    progress: gr.Progress,
) -> tuple[str, Optional[str]]:
    if filepath is None:
        return format_error_html("Please upload a training file first."), None

    progress(0.1, desc="[1/4] Processing document...")

    output_dir = os.path.join(settings.MODEL_DIR, f"{training_method.lower()}_adapter")

    job_id = job_manager.submit_job(
        job_type=f"llm_{training_method.lower()}",
        params={
            "data_path": filepath,
            "base_model": base_model,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
    )

    progress(0.2, desc=f"[2/4] Loading {training_method} components...")

    try:
        job_manager.start_job(job_id)

        if training_method == "SFT":
            from src.llm.dataset_converter import convert_to_sft_format
            from src.llm.trainers.sft_trainer import train_sft

            progress(0.3, desc="[3/4] Training SFT model...")

            dataset = convert_to_sft_format(filepath)

            progress(0.4, desc="Encoding and tokenizing...")

            adapter_path = train_sft(
                dataset=dataset,
                base_model=base_model,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
            )

        elif training_method == "DPO":
            from src.llm.dataset_converter import convert_to_dpo_format
            from src.llm.trainers.dpo_trainer import train_dpo

            progress(0.3, desc="[3/4] Training DPO model...")

            dataset = convert_to_dpo_format(filepath)

            progress(0.4, desc="Preparing preference pairs...")

            adapter_path = train_dpo(
                dataset=dataset,
                base_model=base_model,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
            )

        elif training_method == "GRPO":
            from src.llm.dataset_converter import convert_to_grpo_format
            from src.llm.trainers.grpo_trainer import train_grpo

            progress(0.3, desc="[3/4] Training GRPO model...")

            dataset = convert_to_grpo_format(filepath)

            progress(0.4, desc="Optimizing with rewards...")

            adapter_path = train_grpo(
                dataset=dataset,
                base_model=base_model,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
            )

        else:
            return (
                format_error_html(f"Unknown training method: {training_method}"),
                None,
            )

        progress(0.9, desc="[4/4] Finalizing adapter...")
        job_manager.complete_job(job_id, adapter_path)

        return (
            format_success_html(
                f"{training_method} training complete! Adapter saved to: {adapter_path}"
            ),
            adapter_path,
        )

    except Exception as e:
        job_manager.fail_job(job_id, str(e))
        return format_error_html(f"Training failed: {str(e)}"), None


def load_model_for_inference(
    model_path: Optional[str],
) -> tuple[str, gr.JSON]:
    if model_path is None or model_path == "":
        return format_warning_html("Please provide a model path."), gr.JSON(value={})

    if not os.path.exists(model_path):
        return format_error_html(f"Model not found at: {model_path}"), gr.JSON(value={})

    try:
        model = joblib.load(model_path)

        info = {
            "model_type": type(model).__name__,
            "path": model_path,
        }

        return (
            format_success_html(f"Model loaded: {info['model_type']}"),
            gr.JSON(value=info),
        )

    except Exception as e:
        return format_error_html(f"Failed to load model: {str(e)}"), gr.JSON(value={})


def predict_with_model(model_path: str, input_data: str) -> str:
    if not model_path or not input_data:
        return format_warning_html("Please provide both model path and input data.")

    try:
        features = json.loads(input_data)
        model = joblib.load(model_path)
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            confidence = max(proba)
            return format_success_html(f"Prediction: {prediction} | Confidence: {confidence:.2%}")

        return format_success_html(f"Prediction: {prediction}")

    except json.JSONDecodeError:
        return format_error_html("Invalid JSON input. Please provide valid JSON.")

    except Exception as e:
        return format_error_html(f"Prediction failed: {str(e)}")


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

            confusion_matrix_display = gr.Dataframe(label="Confusion Matrix")

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
            ],
        ).then(
            fn=lambda x: gr.Textbox(value=x),
            inputs=[model_download],
            outputs=[inference_model_path],
        )

    return tab


def create_llm_finetuning_tab() -> gr.Tab:
    with gr.Tab("LLM Fine-tuning") as tab:
        gr.Markdown("# LLM Fine-tuning")
        gr.Markdown("Upload training data and configure fine-tuning parameters.")

        with gr.Row():
            train_file = gr.File(
                label="Upload Training File (TXT/PDF)",
                file_types=[".txt", ".pdf"],
                type="filepath",
            )

        with gr.Row():
            training_method = gr.Radio(
                label="Training Method",
                choices=["SFT", "DPO", "GRPO"],
                value="SFT",
            )

            base_model = gr.Dropdown(
                label="Base Model",
                choices=BASE_MODELS,
                value=BASE_MODELS[0],
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

        train_btn = gr.Button("Start Training", variant="primary")

        llm_training_status = gr.HTML()

        adapter_download = gr.File(
            label="Download LoRA Adapter",
            interactive=False,
        )

        train_btn.click(
            fn=train_llm_model,
            inputs=[train_file, training_method, base_model, epochs, learning_rate],
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


with gr.Blocks(title="Agentic AutoML Platform") as demo:
    gr.Markdown("# Agentic AutoML Platform")

    create_tabular_ml_tab()
    create_llm_finetuning_tab()
    create_inference_playground_tab()


if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=5)
    demo.launch(server_name="0.0.0.0", server_port=7860)
