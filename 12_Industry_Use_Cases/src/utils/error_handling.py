AUTOML_ERRORS = {
    "missing_target": """
The target column '{column}' contains {count} missing values ({percent:.1f}%).

To fix this issue:
1. Open your CSV file
2. Either remove rows with missing target values, or
3. Fill missing values with an appropriate default

Example pandas code:
    df = df.dropna(subset=['{column}'])
""",
    "invalid_types": """
Column '{column}' contains inconsistent data types.

Found {count} values that don't match the expected type '{expected_type}'.
First problematic value: '{sample_value}' at row {row_number}

To fix this issue:
1. Open your CSV file and inspect column '{column}'
2. Ensure all values are of type '{expected_type}'
3. Convert or remove incompatible values

Example pandas code:
    df['{column}'] = pd.to_numeric(df['{column}'], errors='coerce')
    df = df.dropna(subset=['{column}'])
""",
    "file_not_found": """
File not found: '{filepath}'

The specified file does not exist at the given path.

To fix this issue:
1. Verify the file path is correct
2. Check if the file has been moved or renamed
3. Ensure you have read permissions for the directory

Example:
    import os
    print(os.path.exists('{filepath}'))  # Should return True
""",
    "invalid_csv": """
CSV parsing error: {error_message}

The file '{filepath}' could not be parsed as a valid CSV.

To fix this issue:
1. Open the file in a text editor to inspect its structure
2. Check for mismatched quotes, delimiters, or line endings
3. Ensure the file is saved with proper CSV formatting

Common issues:
- Mixed quote styles (use consistent single or double quotes)
- Incorrect delimiter (expected comma, found: {delimiter})
- Broken rows with different column counts
""",
    "missing_features": """
No feature columns found in the dataset.

After excluding the target column '{target_column}', no valid feature columns remain.
This typically occurs when:
- The dataset only contains the target column
- All non-target columns are empty or invalid

To fix this issue:
1. Verify your CSV contains multiple columns
2. Ensure feature columns have valid data types (numeric or categorical)
3. Check for empty or completely null columns

Example pandas code:
    print(df.columns.tolist())  # List all available columns
    print(df.drop(columns=['{target_column}']).dtypes)  # Check feature types
""",
    "too_few_samples": """
Insufficient training data: {count} samples found.

Training requires at least {minimum} samples for reliable results.
Your dataset has {count} rows, which is below the minimum threshold.

To fix this issue:
1. Collect more data for your dataset
2. Reduce model complexity if limited data is unavoidable
3. Consider using cross-validation with fewer folds

Current dataset size: {count} rows
Minimum required: {minimum} rows
Recommended for robust training: {recommended} rows

Example calculation:
    import math
    n_samples = len(df)
    min_required = max(100, int(len(df.columns) * 10))
""",
    "encoding_error": """
File encoding error: {error_message}

The file '{filepath}' could not be read with the detected encoding.

Detected encoding: {detected_encoding}
Expected encoding: utf-8

To fix this issue:
1. Open the file in a text editor and save with UTF-8 encoding
2. Or specify the correct encoding when reading

Example pandas code:
    df = pd.read_csv('{filepath}', encoding='latin-1')  # Try alternative encodings
    df = pd.read_csv('{filepath}', encoding='utf-8-sig')  # For UTF-8 with BOM
    df = pd.read_csv('{filepath}', encoding='cp1252')  # Windows-1252
""",
}


def format_error(error_key: str, **kwargs) -> str:
    """
    Format an error message from AUTOML_ERRORS with provided parameters.

    Args:
        error_key: Key identifying the error type in AUTOML_ERRORS dictionary
        **kwargs: Parameters to substitute into the error message template

    Returns:
        Formatted error message string with placeholders replaced by values

    Raises:
        KeyError: If error_key is not found in AUTOML_ERRORS
    """
    template = AUTOML_ERRORS[error_key]
    return template.format(**kwargs)
