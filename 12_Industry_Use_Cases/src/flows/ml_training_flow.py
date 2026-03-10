import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from metaflow import FlowSpec, Parameter, step, card, current
from metaflow.cards import Markdown, Table
from sklearn.model_selection import train_test_split

from src.ml.auto_ensemble import evaluate_model, train_ensemble
from src.ml.data_validator import detect_task_type_and_recommend_model, validate_csv
from src.utils.model_packaging import save_sklearn_model_package

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLTrainingFlow(FlowSpec):
    """
    Metaflow pipeline for automated ML model training on tabular data.

    This flow handles the complete ML pipeline from data validation through
    model training, evaluation, and serialization. Preprocessing (imputation,
    scaling, one-hot encoding) is handled inside the sklearn Pipeline so it
    is automatically applied during inference.
    """

    data_path = Parameter(
        "data_path",
        help="Path to the CSV file containing training data",
        required=True,
    )

    target_column = Parameter(
        "target_column",
        help="Name of the target column to predict",
        required=True,
    )

    @step
    def start(self):
        """Initialize the flow and validate parameters."""
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """Load the CSV file into a pandas DataFrame."""
        self.df = pd.read_csv(self.data_path)

        self.card = Markdown(
            f"""
        ## Data Loaded

        - Rows: {len(self.df)}
        - Columns: {len(self.df.columns)}
        - File: {self.data_path}

        ### Column Types
        | Column | Type |
        |--------|------|
        {chr(10).join(f"| {col} | {dtype} |" for col, dtype in self.df.dtypes.items())}
        """
        )

        self.next(self.validate_data)

    @step
    def validate_data(self):
        """Validate the CSV file structure and target column."""
        result = validate_csv(self.data_path, self.target_column)
        if not result["valid"]:
            raise ValueError(result["message"])
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Separate features/target, detect task type, and train/test split.

        Preprocessing (imputation, scaling, one-hot encoding) is NOT done here.
        It is handled inside the sklearn Pipeline built by train_ensemble(),
        ensuring the same transformations are applied during inference.
        """
        logger.info(f"preprocess: target_column={self.target_column}, df shape={self.df.shape}")

        try:
            y = self.df[self.target_column]
            x_features = self.df.drop(columns=[self.target_column])
        except KeyError as e:
            raise ValueError(f"Target column '{self.target_column}' not found in data: {e}")

        logger.info("Detecting task type and recommending model...")
        try:
            detection_result = detect_task_type_and_recommend_model(self.df, self.target_column)
            logger.info(f"Task type detected: {detection_result['task_type']}")

            if detection_result.get('fallback_used', False):
                logger.warning("RAG/LLM services unavailable - using fallback model recommendation")
            else:
                logger.info(f"Model recommendation received (length: {len(detection_result.get('model_recommendation', ''))})")

        except ValueError as e:
            logger.error(f"Task detection failed with validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Task detection failed with unexpected error: {e}")
            raise RuntimeError(f"Failed to detect task type and recommend model: {e}")

        self.task_type = detection_result["task_type"]
        self.model_recommendation = detection_result["model_recommendation"]
        self.fallback_used = detection_result.get("fallback_used", False)

        logger.info("Performing train/test split (80/20)...")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_features, y, test_size=0.2, random_state=42
        )

        logger.info(f"Training set: {len(self.x_train)} samples, Test set: {len(self.x_test)} samples")
        logger.info(f"Feature columns: {list(self.x_train.columns)}")
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the best model via GridSearchCV over multiple model families.

        Uses a sklearn Pipeline with ColumnTransformer for preprocessing
        and GridSearchCV to find the best model + hyperparameters.
        """
        result = train_ensemble(self.x_train, self.y_train, self.task_type)
        self.model = result["model"]
        self.estimators = result["estimators"]
        self.grid_search_results = result.get("grid_search_results", {})

        best_model = self.grid_search_results.get("best_model", "Unknown")
        best_score = self.grid_search_results.get("best_score", 0)
        logger.info(f"Best model from GridSearchCV: {best_model} (cv_score={best_score:.4f})")

        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """Evaluate the trained model on the test set."""
        self.metrics = evaluate_model(self.model, self.x_test, self.y_test, self.task_type)

        if self.task_type == "classification":
            metrics_table = Table(
                data=[
                    ["Accuracy", f"{self.metrics.get('accuracy', 0):.2%}"],
                    ["F1 (Macro)", f"{self.metrics.get('f1_macro', 0):.4f}"],
                    ["Precision", f"{self.metrics.get('precision_macro', 0):.4f}"],
                    ["Recall", f"{self.metrics.get('recall_macro', 0):.4f}"],
                ],
                headers=["Metric", "Value"],
            )
        else:
            metrics_table = Table(
                data=[
                    ["RMSE", f"{self.metrics.get('rmse', 0):.4f}"],
                    ["MAE", f"{self.metrics.get('mae', 0):.4f}"],
                    ["R²", f"{self.metrics.get('r2', 0):.4f}"],
                ],
                headers=["Metric", "Value"],
            )

        current.card.append(metrics_table)

        self.next(self.save_model)

    @step
    def save_model(self):
        """Save the trained Pipeline to disk with full metadata."""
        from src.config import settings

        self.model_path = save_sklearn_model_package(
            pipeline=self.model,
            output_dir=settings.MODEL_DIR,
            model_name=f"automl_{self.task_type}",
            version=str(current.run_id),
            task_type=self.task_type,
            target_column=self.target_column,
            feature_columns=list(self.x_train.columns),
            metrics={
                "accuracy" if self.task_type == "classification" else "rmse": self.metrics.get(
                    "accuracy", 0
                )
                if self.task_type == "classification"
                else self.metrics.get("rmse", float("inf"))
            },
            description=f"Best model: {self.grid_search_results.get('best_model', 'Unknown')}. "
                        f"CV score: {self.grid_search_results.get('best_score', 0):.4f}",
        )

        print(f"Model saved to {self.model_path}")
        self.next(self.end)

    @step
    def end(self):
        """Complete the flow and log final results."""
        logger.info("Training complete!")
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"Best model: {self.grid_search_results.get('best_model', 'Unknown')}")

        if getattr(self, 'fallback_used', False):
            logger.warning("Model recommendation used fallback (RAG/LLM services were unavailable)")

        logger.info(f"Model saved to: {self.model_path}")
        logger.info(f"Metrics: {self.metrics}")


if __name__ == "__main__":
    MLTrainingFlow()
