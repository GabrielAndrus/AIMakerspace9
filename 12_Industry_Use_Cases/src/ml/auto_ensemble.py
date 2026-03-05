from typing import Union

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from xgboost import XGBClassifier, XGBRegressor


def create_ensemble(task_type: str) -> Union[VotingClassifier, VotingRegressor]:
    """
    Create an ensemble model based on the task type.

    Args:
        task_type: Either 'classification' or 'regression'

    Returns:
        VotingClassifier for classification tasks, VotingRegressor for regression tasks

    Raises:
        ValueError: If task_type is not 'classification' or 'regression'
    """
    if task_type == "classification":
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBClassifier(verbosity=0)),
            ("lgbm", LGBMClassifier(verbose=-1)),
        ]
        return VotingClassifier(estimators, voting="soft")
    elif task_type == "regression":
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=100)),
            ("xgb", XGBRegressor(verbosity=0)),
            ("lgbm", LGBMRegressor(verbose=-1)),
        ]
        return VotingRegressor(estimators)
    else:
        raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")


def train_ensemble(x_train: pd.DataFrame, y_train: pd.Series, task_type: str) -> dict:
    """
    Train an ensemble model on the provided data.

    Args:
        x_train: Training features dataframe
        y_train: Target values series
        task_type: Either 'classification' or 'regression'

    Returns:
        Dictionary with keys:
            - 'model': Trained ensemble model
            - 'estimators': Dict of individual models

    Raises:
        ValueError: If task_type is not supported
    """
    ensemble = create_ensemble(task_type)
    ensemble.fit(x_train, y_train)

    estimators_dict = {}
    for name, estimator in zip([n for n, _ in ensemble.estimators], ensemble.estimators_):
        estimators_dict[name] = estimator

    return {"model": ensemble, "estimators": estimators_dict}


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained sklearn-compatible model
        x_test: Test features dataframe
        y_test: Test target values series
        task_type: Either 'classification' or 'regression'

    Returns:
        Dictionary with metric names as keys and metric values as values.
        For classification: accuracy, f1_macro, precision_macro, recall_macro
        For regression: mse, rmse, mae, r2

    Raises:
        ValueError: If task_type is not supported
    """
    y_pred = model.predict(x_test)

    if task_type == "classification":
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
        }
    elif task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        return {
            "mse": mse,
            "rmse": float(mse**0.5),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
    else:
        raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")
