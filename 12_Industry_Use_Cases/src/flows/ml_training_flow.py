import pandas as pd
from metaflow import FlowSpec, Parameter, step
from sklearn.model_selection import train_test_split

from src.ml.auto_ensemble import evaluate_model, train_ensemble
from src.ml.data_validator import detect_task_type, validate_csv
from src.utils.serialization import save_model


class MLTrainingFlow(FlowSpec):
    """
    Metaflow pipeline for automated ML model training on tabular data.

    This flow handles the complete ML pipeline from data validation through
    model training, evaluation, and serialization.
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
        """
        Initialize the flow and validate parameters.
        """
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.data_path)
        self.next(self.validate_data)

    @step
    def validate_data(self):
        """
        Validate the CSV file structure and target column.

        Raises:
            ValueError: If validation fails for file or target column.
        """
        result = validate_csv(self.data_path, self.target_column)
        if not result["valid"]:
            raise ValueError(result["message"])
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """
        Preprocess data: split features/target, encode categoricals, train/test split.

        This step:
            - Separates features from target
            - One-hot encodes categorical columns
            - Detects task type (classification/regression)
            - Creates train/test split (80/20)
        """
        y = self.df[self.target_column]
        x_features = self.df.drop(columns=[self.target_column])

        categorical_cols = x_features.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            x_features = pd.get_dummies(x_features, columns=categorical_cols)

        self.task_type = detect_task_type(y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_features, y, test_size=0.2, random_state=42
        )

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train an ensemble model based on the detected task type.

        Uses a voting ensemble combining RandomForest, XGBoost, and LightGBM.
        """
        result = train_ensemble(self.x_train, self.y_train, self.task_type)
        self.model = result["model"]
        self.estimators = result["estimators"]
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """
        Evaluate the trained model on the test set.

        Computes appropriate metrics based on task type:
            - Classification: accuracy, f1_macro, precision_macro, recall_macro
            - Regression: mse, rmse, mae, r2
        """
        self.metrics = evaluate_model(self.model, self.x_test, self.y_test, self.task_type)
        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Save the trained model to disk.

        Raises:
            OSError: If the model cannot be saved.
        """
        import os

        from src.config import settings

        model_filename = f"ensemble_{self.task_type}_model.joblib"
        model_path = os.path.join(settings.MODEL_DIR, model_filename)
        self.model_path = save_model(self.model, model_path)
        self.next(self.end)

    @step
    def end(self):
        """
        Complete the flow and log final results.
        """
        print("Training complete!")
        print(f"Task type: {self.task_type}")
        print(f"Metrics: {self.metrics}")
        print(f"Model saved to: {self.model_path}")


if __name__ == "__main__":
    MLTrainingFlow()
