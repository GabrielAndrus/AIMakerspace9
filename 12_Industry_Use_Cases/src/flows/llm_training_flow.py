"""LLM Training Flow with Metaflow.

This module implements a Metaflow pipeline for LLM fine-tuning using
various training methods (SFT, DPO, GRPO).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os

from metaflow import FlowSpec, Parameter, step, card, current
from metaflow.cards import Markdown, Table

from src.config import settings
from src.utils.model_packaging import save_lora_adapter_package


class LLMTrainingFlow(FlowSpec):
    """
    Metaflow pipeline for automated LLM fine-tuning.

    This flow handles the complete LLM training pipeline from data loading
    through model fine-tuning and adapter serialization.
    """

    data_path = Parameter(
        "data_path",
        help="Path to the training file (TXT/PDF)",
        required=True,
    )

    training_method = Parameter(
        "training_method",
        help="Training method: SFT, DPO, or GRPO",
        default="SFT",
    )

    base_model = Parameter(
        "base_model",
        help="Base model to fine-tune",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    epochs = Parameter(
        "epochs",
        help="Number of training epochs",
        default=3,
    )

    learning_rate = Parameter(
        "learning_rate",
        help="Learning rate for training",
        default=2e-4,
    )

    reward_template = Parameter(
        "reward_template",
        help="GRPO reward template ('math', 'format_check'). Auto-detected from dataset if not specified.",
        default=None,
    )

    use_quantization = Parameter(
        "use_quantization",
        help="Use 4-bit quantization (requires bitsandbytes). Set to False for newer CUDA versions.",
        default=False,
    )

    @card(type="default")
    @step
    def start(self):
        """
        Initialize the flow and validate parameters.
        """
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """
        Load and validate the training data file.

        Creates a card showing file information.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Get file info
        file_ext = Path(self.data_path).suffix.lower()
        file_size = os.path.getsize(self.data_path)

        # Add data summary card
        self.card = Markdown(
            f"""
        ## Data Loaded

        - File: {self.data_path}
        - Type: {file_ext.upper()}
        - Size: {file_size:,} bytes
        - Training Method: {self.training_method}
        """
        )

        self.next(self.validate_data)

    @step
    def validate_data(self):
        """
        Validate the data file for the selected training method.
        """
        # Check if file extension matches training method requirements
        file_ext = Path(self.data_path).suffix.lower()

        if file_ext not in [".txt", ".pdf", ".jsonl"]:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .txt, .pdf, .jsonl")

        self.next(self.convert_data)

    @step
    def convert_data(self):
        """
        Convert the raw data to the appropriate training format.
        """
        from src.llm.dataset_converter import (
            convert_to_dpo_format,
            convert_to_grpo_format,
            convert_to_sft_format,
        )

        if self.training_method == "SFT":
            from datasets import Dataset

            converted_data = convert_to_sft_format(self.data_path)
            self.dataset = Dataset.from_list(converted_data)

        elif self.training_method == "DPO":
            from datasets import Dataset

            converted_data = convert_to_dpo_format(self.data_path)
            self.dataset = Dataset.from_list(converted_data)

        elif self.training_method == "GRPO":
            from datasets import Dataset

            converted_data = convert_to_grpo_format(self.data_path)
            self.dataset = Dataset.from_list(converted_data)

        else:
            raise ValueError(f"Unknown training method: {self.training_method}")

        self.next(self.train_model)

    @card
    @step
    def train_model(self):
        """
        Train the LLM model using the selected training method.

        Creates a card showing training configuration and progress.
        """
        output_dir = os.path.join(settings.MODEL_DIR, f"{self.training_method.lower()}_adapter")

        # Add training configuration card
        self.card = Markdown(
            f"""
        ## Training Configuration

        | Parameter | Value |
        |-----------|-------|
        | Base Model | {self.base_model} |
        | Training Method | {self.training_method} |
        | Epochs | {self.epochs} |
        | Learning Rate | {self.learning_rate:.2e} |
        | Output Directory | {output_dir} |

        ### Dataset Information

        - Number of examples: {len(self.dataset)}
        """
        )

        if self.training_method == "SFT":
            from src.llm.trainers.sft_trainer import train_sft

            self.adapter_path = train_sft(
                dataset=self.dataset,
                base_model=self.base_model,
                output_dir=output_dir,
                epochs=int(self.epochs),
                learning_rate=float(self.learning_rate),
                use_quantization=self.use_quantization,
            )

        elif self.training_method == "DPO":
            from src.llm.trainers.dpo_trainer import train_dpo

            self.adapter_path = train_dpo(
                dataset=self.dataset,
                base_model=self.base_model,
                output_dir=output_dir,
                epochs=int(self.epochs),
                use_quantization=self.use_quantization,
            )

        elif self.training_method == "GRPO":
            from src.llm.trainers.grpo_trainer import train_grpo

            # Auto-detect reward template if not specified
            detected_template = self.reward_template
            if detected_template is None and len(self.dataset) > 0:
                first_example = self.dataset[0]
                if "ground_truth" in first_example:
                    detected_template = "math"
                elif "pattern" in first_example:
                    detected_template = "format_check"

            self.adapter_path = train_grpo(
                dataset=self.dataset,
                base_model=self.base_model,
                output_dir=output_dir,
                reward_template=detected_template,
                use_quantization=self.use_quantization,
            )

        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """
        Evaluate the trained model and show results.

        Creates a card with training summary.
        """
        # Basic metrics - can be expanded based on actual training results
        self.metrics = {
            "training_method": self.training_method,
            "base_model": self.base_model,
            "epochs": int(self.epochs),
            "learning_rate": float(self.learning_rate),
        }

        # Create summary table
        metrics_table = Table(
            data=[
                ["Training Method", self.training_method],
                ["Base Model", self.base_model],
                ["Epochs", str(self.epochs)],
                ["Learning Rate", f"{self.learning_rate:.2e}"],
                ["Adapter Path", self.adapter_path],
            ],
            headers=["Metric", "Value"],
        )

        current.card.append(metrics_table)

        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Save the LoRA adapter with metadata.

        The trainer already saves the model, but we add metadata here.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.adapter_path,
                device_map="cpu",
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)

            self.model_path = save_lora_adapter_package(
                model=model,
                tokenizer=tokenizer,
                output_dir=settings.MODEL_DIR,
                model_name=f"llm_{self.training_method.lower()}",
                version=str(current.run_id),
                base_model=self.base_model,
                lora_config=None,
                training_args={
                    "method": self.training_method.lower(),
                    "num_epochs": int(self.epochs),
                    "learning_rate": float(self.learning_rate),
                },
                metrics={"train_loss": self.metrics.get("train_loss")},
            )

        except Exception as e:
            print(f"Warning: Could not save with full metadata: {e}")
            self.model_path = self.adapter_path

        print(f"LoRA adapter saved to {self.model_path}")
        self.next(self.end)

    @step
    def end(self):
        """
        Complete the flow and log final results.
        """
        print("LLM Training complete!")
        print(f"Training Method: {self.training_method}")
        print(f"Base Model: {self.base_model}")
        print(f"Adapter saved to: {self.model_path}")


if __name__ == "__main__":
    LLMTrainingFlow()
