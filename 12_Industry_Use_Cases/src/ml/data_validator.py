import os
import time
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from src.utils.error_handling import format_error
from ..retrieval import embed_query, QdrantRetriever

logger = logging.getLogger(__name__)


SUPPORTED_FORMATS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


def detect_file_format(filepath: str) -> str:
    """
    Detect file format from extension.

    Returns:
        'csv', 'tsv', or raises ValueError

    Raises:
        ValueError: If file type is not supported
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    format_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    if suffix not in format_map:
        raise ValueError(
            f"Unsupported file type: {suffix}\nSupported types: {', '.join(SUPPORTED_FORMATS)}"
        )

    return format_map[suffix]


def read_data_file(filepath: str) -> pd.DataFrame:
    """
    Read data file into pandas DataFrame.

    Supports CSV, TSV, and Excel formats.

    Args:
        filepath: Path to the data file

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If file format is not supported or cannot be read
    """
    fmt = detect_file_format(filepath)

    if fmt == "csv":
        return pd.read_csv(filepath)
    elif fmt == "tsv":
        return pd.read_csv(filepath, sep="\t")
    elif fmt == "excel":
        return pd.read_excel(filepath)

    raise ValueError(f"Unknown format: {fmt}")


def validate_csv(filepath: str, target_column: Optional[str] = None) -> dict:
    """
    Validate a data file (CSV, TSV, or Excel) for ML training.

    Args:
        filepath: Path to the data file
        target_column: Optional target column name to validate

    Returns:
        Dictionary with keys:
            - 'valid' (bool): Whether validation passed
            - 'message' (str): Success or error message
            - 'columns' (list): List of column names if valid, empty otherwise

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed or target validation fails
    """
    path = Path(filepath)

    if not path.exists():
        return {
            "valid": False,
            "message": format_error("file_not_found", filepath=filepath),
            "columns": [],
        }

    try:
        df = read_data_file(filepath)
    except ValueError as e:
        return {
            "valid": False,
            "message": format_error(
                "invalid_csv",
                error_message=str(e),
                filepath=filepath,
                delimiter="unknown",
            ),
            "columns": [],
        }
    except Exception as e:
        return {
            "valid": False,
            "message": format_error(
                "invalid_csv",
                error_message=str(e),
                filepath=filepath,
                delimiter="unknown",
            ),
            "columns": [],
        }

    if target_column is not None:
        if target_column not in df.columns:
            return {
                "valid": False,
                "message": f"Target column '{target_column}' not found in CSV. Available columns: {df.columns.tolist()}",
                "columns": [],
            }

        missing_count = df[target_column].isna().sum()
        if missing_count > 0:
            total_count = len(df)
            missing_pct = (missing_count / total_count) * 100
            return {
                "valid": False,
                "message": format_error(
                    "missing_target",
                    column=target_column,
                    count=missing_count,
                    percent=missing_pct,
                ),
                "columns": [],
            }

    return {
        "valid": True,
        "message": f"Successfully validated data file with {len(df)} rows and {len(df.columns)} columns.",
        "columns": df.columns.tolist(),
    }


def detect_task_type(y: pd.Series) -> str:
    """
    Detect whether the target variable is for classification or regression.

    Args:
        y: Target variable as a pandas Series

    Returns:
        'classification' or 'regression'
    """
    if y.dtype == "object":
        return "classification"

    n_unique = y.nunique()
    n_total = len(y)

    if n_unique < 10:
        return "classification"

    unique_ratio = n_unique / n_total
    if unique_ratio < 0.05:
        return "classification"

    return "regression"


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Categorize DataFrame columns by data type for ML processing.

    Args:
        df: Input DataFrame to analyze

    Returns:
        Dictionary mapping column names to types:
            - 'numeric': Integer or float columns
            - 'categorical': Object/string columns with <= 50 unique values
            - 'text': Object/string columns with > 50 unique values
    """
    column_types = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif df[col].dtype == "object":
            n_unique = df[col].nunique()
            if n_unique <= 50:
                column_types[col] = "categorical"
            else:
                column_types[col] = "text"
        else:
            column_types[col] = "categorical"

    return column_types


def _get_fallback_recommendation(task_type: str, n_rows: int, n_features: int) -> str:
    """Generate a fallback model recommendation when RAG/LLM is unavailable."""
    if task_type == "classification":
        return f"""Fallback Recommendation (RAG/LLM unavailable):

For classification with {n_rows} samples and {n_features} features:
- RandomForestClassifier: Robust, handles mixed data types well
- GradientBoostingClassifier: Often achieves best performance
- LogisticRegression: Good baseline for simpler problems

Recommended ensemble: VotingClassifier combining the above models."""
    else:
        return f"""Fallback Recommendation (RAG/LLM unavailable):

For regression with {n_rows} samples and {n_features} features:
- RandomForestRegressor: Robust, handles non-linear relationships
- GradientBoostingRegressor: Often achieves best performance  
- Ridge/Lasso Regression: Good baseline for linear problems

Recommended ensemble: VotingRegressor combining the above models."""


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Execute function with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


def detect_task_type_and_recommend_model(
    df: pd.DataFrame, target_column: str
) -> dict[str, Any]:
    """
    Detect task type and recommend scikit-learn model using RAG-based approach.

    Args:
        df: Input DataFrame
        target_column: Name of the target column

    Returns:
        Dictionary with:
            - 'task_type': 'classification' or 'regression'
            - 'model_recommendation': LLM-generated model recommendation
            - 'retrieved_contexts': Contexts from knowledge base
            - 'fallback_used': True if fallback recommendation was used

    Raises:
        ValueError: If target_column cannot be extracted from DataFrame
    """
    logger.info(f"detect_task_type_and_recommend_model: target={target_column}, df shape={df.shape}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame. Columns: {df.columns.tolist()}")
    
    try:
        y = df[target_column]
    except Exception as e:
        logger.error(f"ERROR extracting target column: {e}")
        raise ValueError(f"Failed to extract target column '{target_column}': {e}")

    task_type = detect_task_type(y)
    logger.info(f"Task type detected: {task_type}")

    inference_url = os.getenv("LLM_INFERENCE_URL", "http://192.168.1.185:8080/v1")
    inference_key = os.getenv("LLM_INFERENCE_KEY", "not-needed")

    column_types = get_column_types(df)
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target_column": target_column,
        "task_type": task_type,
        "column_types": column_types,
    }

    query = f"Best scikit-learn model for {task_type} with {len(df)} rows and {len(df.columns)} features"

    contexts = []
    recommendation = None
    fallback_used = False

    # Step 1: Try embedding query with retry
    try:
        logger.info(f"Embedding query: {query[:50]}...")
        
        def embed_func():
            return embed_query(query)
        
        query_embedding = _retry_with_backoff(embed_func, max_retries=2, base_delay=0.5)
        logger.info("Query embedding successful")
        
    except Exception as e:
        logger.warning(f"Embedding failed after retries: {e}. Using fallback recommendation.")
        fallback_used = True
        return {
            "task_type": task_type,
            "model_recommendation": _get_fallback_recommendation(task_type, len(df), len(df.columns)),
            "retrieved_contexts": [],
            "fallback_used": True,
        }

    # Step 2: Try Qdrant search with retry
    try:
        logger.info("Searching Qdrant...")
        retriever = QdrantRetriever()
        logger.debug(f"QDRANT URL: {retriever.url}")
        
        def search_func():
            return retriever.search(query_embedding, limit=5)
        
        retrieved = _retry_with_backoff(search_func, max_retries=2, base_delay=0.5)
        logger.info(f"Retrieved {len(retrieved)} contexts from Qdrant")
        
        if retrieved:
            contexts = [r["payload"]["content"] for r in retrieved]
        else:
            logger.warning("Qdrant returned no results. Proceeding without RAG context.")
            
    except Exception as e:
        logger.warning(f"Qdrant search failed after retries: {e}. Proceeding without RAG context.")
        contexts = []

    # Step 3: Try LLM inference with retry
    try:
        logger.info("Calling LLM for model recommendation...")
        client = OpenAI(base_url=inference_url, api_key=inference_key)

        context_text = "\n\n".join([f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)]) if contexts else "No additional context available."

        prompt = f"""Given the following dataset profile and relevant documentation contexts, recommend the best ML model(s) to use.

Dataset Profile:
{profile}

Relevant Documentation:
{context_text}

Based on the above, provide:
1. Recommended model(s) with justification
2. Key hyperparameters to consider

Provide your recommendation in a clear, structured format."""

        def llm_func():
            return client.chat.completions.create(
                model="minimax-m2.5-mlx@8bit",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert ML engineer. Recommend models based on dataset characteristics and scikit-learn best practices.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
        
        response = _retry_with_backoff(llm_func, max_retries=2, base_delay=1.0)
        recommendation = response.choices[0].message.content
        logger.info(f"LLM response received, length: {len(recommendation)}")
        
    except (APIConnectionError, APITimeoutError) as e:
        logger.warning(f"LLM connection failed: {e}. Using fallback recommendation.")
        fallback_used = True
        recommendation = _get_fallback_recommendation(task_type, len(df), len(df.columns))
        
    except APIError as e:
        logger.warning(f"LLM API error: {e}. Using fallback recommendation.")
        fallback_used = True
        recommendation = _get_fallback_recommendation(task_type, len(df), len(df.columns))
        
    except Exception as e:
        logger.warning(f"LLM inference failed: {e}. Using fallback recommendation.")
        fallback_used = True
        recommendation = _get_fallback_recommendation(task_type, len(df), len(df.columns))

    return {
        "task_type": task_type,
        "model_recommendation": recommendation,
        "retrieved_contexts": contexts,
        "fallback_used": fallback_used,
    }
