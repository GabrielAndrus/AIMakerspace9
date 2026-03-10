import logging
from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

# High-cardinality threshold: columns with more unique values than this
# are dropped instead of one-hot encoded to avoid feature explosion.
HIGH_CARDINALITY_THRESHOLD = 50


def build_preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    """Build a ColumnTransformer that handles numeric and categorical features.

    Numeric columns get imputation (median) + standard scaling.
    Categorical columns (object dtype, <=HIGH_CARDINALITY_THRESHOLD unique) get
    imputation (most frequent) + one-hot encoding.
    High-cardinality object columns are dropped.

    Args:
        x: Feature DataFrame (without target column).

    Returns:
        Tuple of (ColumnTransformer, list of columns to keep).
    """
    num_cols = x.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [
        c for c in x.select_dtypes(include=["object"]).columns
        if x[c].nunique() <= HIGH_CARDINALITY_THRESHOLD
    ]
    dropped = [
        c for c in x.select_dtypes(include=["object"]).columns
        if x[c].nunique() > HIGH_CARDINALITY_THRESHOLD
    ]
    if dropped:
        logger.info(f"Dropping high-cardinality columns: {dropped}")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False, drop="if_binary", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    kept_cols = num_cols + cat_cols
    return preprocessor, kept_cols


def _get_search_space(task_type: str) -> list[dict]:
    """Define the GridSearchCV search space for multiple model families.

    Each dict in the list swaps the 'model' step and tunes its hyperparams.
    """
    if task_type == "classification":
        return [
            {
                "model": [RandomForestClassifier(n_estimators=100)],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
            {
                "model": [XGBClassifier(verbosity=0, use_label_encoder=False, eval_metric="logloss")],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.01, 0.1],
            },
            {
                "model": [LGBMClassifier(verbose=-1)],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.01, 0.1],
            },
            {
                "model": [GradientBoostingClassifier()],
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
            },
            {
                "model": [LogisticRegression(max_iter=2000)],
                "model__C": [0.1, 1.0, 10.0],
            },
            {
                "model": [SVC(probability=True)],
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear"],
            },
            {
                "model": [DecisionTreeClassifier()],
                "model__max_depth": [5, 10, 20],
            },
            {
                "model": [KNeighborsClassifier()],
                "model__n_neighbors": [3, 5, 7],
            },
        ]
    else:
        return [
            {
                "model": [RandomForestRegressor(n_estimators=100)],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
            {
                "model": [XGBRegressor(verbosity=0)],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.01, 0.1],
            },
            {
                "model": [LGBMRegressor(verbose=-1)],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.01, 0.1],
            },
            {
                "model": [GradientBoostingRegressor()],
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
            },
            {
                "model": [Ridge()],
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            {
                "model": [Lasso(max_iter=2000)],
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            {
                "model": [SVR()],
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear"],
            },
            {
                "model": [DecisionTreeRegressor()],
                "model__max_depth": [5, 10, 20],
            },
            {
                "model": [KNeighborsRegressor()],
                "model__n_neighbors": [3, 5, 7],
            },
        ]


def train_ensemble(x_train: pd.DataFrame, y_train: pd.Series, task_type: str) -> dict:
    """Train the best model using GridSearchCV over multiple model families.

    Builds a sklearn Pipeline with preprocessing (imputation, scaling,
    one-hot encoding) and a model step. GridSearchCV searches over
    different models and hyperparameters to find the best combination.

    Args:
        x_train: Training features DataFrame.
        y_train: Target values Series.
        task_type: 'classification' or 'regression'.

    Returns:
        Dictionary with:
            - 'model': Best trained Pipeline (preprocessing + model)
            - 'estimators': Dict with the best model info
            - 'grid_search_results': Summary of GridSearchCV results

    Raises:
        ValueError: If task_type is not supported.
    """
    if task_type not in ("classification", "regression"):
        raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")

    preprocessor, kept_cols = build_preprocessor(x_train)

    # Use a placeholder model — GridSearchCV will swap it
    if task_type == "classification":
        placeholder = RandomForestClassifier()
        scoring = "f1_macro"
    else:
        placeholder = RandomForestRegressor()
        scoring = "neg_mean_squared_error"

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", placeholder),
    ])

    search_space = _get_search_space(task_type)

    logger.info(f"Starting GridSearchCV with {len(search_space)} model families, scoring={scoring}")
    gs = GridSearchCV(
        pipeline,
        param_grid=search_space,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        error_score="raise",
    )
    gs.fit(x_train[kept_cols] if kept_cols else x_train, y_train)

    best_pipeline = gs.best_estimator_
    best_model = best_pipeline.named_steps["model"]
    best_model_name = type(best_model).__name__
    best_params = best_model.get_params()

    logger.info(f"Best model: {best_model_name} (score: {gs.best_score_:.4f})")
    logger.info(f"Best params: {best_params}")

    # Collect top results summary
    results_df = pd.DataFrame(gs.cv_results_)
    top_results = (
        results_df[["params", "mean_test_score", "rank_test_score"]]
        .sort_values("rank_test_score")
        .head(5)
    )

    return {
        "model": best_pipeline,
        "estimators": {best_model_name: best_model},
        "grid_search_results": {
            "best_model": best_model_name,
            "best_score": float(gs.best_score_),
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
            "top_5": top_results.to_dict("records"),
        },
    }


def create_ensemble(task_type: str) -> Union[VotingClassifier, VotingRegressor]:
    """Create a simple ensemble model (used as fallback).

    Args:
        task_type: Either 'classification' or 'regression'.

    Returns:
        VotingClassifier or VotingRegressor.
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


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> dict:
    """Evaluate a trained model on test data.

    Works with both Pipeline models (new) and raw models (legacy).

    Args:
        model: Trained sklearn Pipeline or model.
        x_test: Test features DataFrame.
        y_test: Test target values Series.
        task_type: 'classification' or 'regression'.

    Returns:
        Dictionary of metric names to values.
    """
    y_pred = model.predict(x_test)

    if task_type == "classification":
        cm = confusion_matrix(y_test, y_pred)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "confusion_matrix": cm.tolist(),
        }
    elif task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        return {
            "mse": float(mse),
            "rmse": float(mse**0.5),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
    else:
        raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")
