"""Core preparation functions."""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, List, Tuple, Any
import warnings


def prepare_data(
    df: pd.DataFrame,
    target: str,
    sharpness: Union[float, Dict[str, float]] = 0.25,
    min_per_interval: int = 5,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prepare DataFrame for entropy analysis.

    Args:
        df: Input DataFrame
        target: Name of target column
        sharpness: Either a single float for all quantitative columns,
                  or a dict mapping column names to sharpness values
        min_per_interval: Minimum samples per interval for warnings
        verbose: Print progress information

    Returns:
        Tuple of (data_matrix, info_dict)

    Raises:
        ValueError: If target column not found or no quantitative columns
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    # Separate columns by type
    quant_cols = []
    cat_cols = []

    for col in df.columns:
        if col == target:
            continue
        # Проверяем тип
        if pd.api.types.is_numeric_dtype(df[col]):
            quant_cols.append(col)
        else:
            # Всё остальное (object, str, category) считаем категориальным
            cat_cols.append(col)

    if not quant_cols and not cat_cols:
        raise ValueError("No quantitative or categorical columns found")

    if verbose:
        print(f"Found {len(quant_cols)} quantitative columns")
        print(f"Found {len(cat_cols)} categorical columns")

    # Process quantitative columns
    quant_data = []
    quant_info = {}

    if quant_cols:
        # Import C++ core (will be implemented in bindings)
        from ._core import discretize_column

        for col in quant_cols:
            # Get sharpness for this column
            if isinstance(sharpness, dict):
                col_sharpness = sharpness.get(col, 0.25)
            else:
                col_sharpness = sharpness

            # Convert to float32 for C++
            values = df[col].values.astype(np.float32)

            # Discretize
            binned = discretize_column(values, col_sharpness)

            # Check coverage
            unique, counts = np.unique(binned, return_counts=True)
            min_count = counts.min() if len(counts) > 0 else 0

            if min_count < min_per_interval and verbose:
                print(f"  Warning: Column '{col}' has interval with only {min_count} samples")

            quant_data.append(binned.reshape(-1, 1))
            quant_info[col] = {
                'type': 'quantitative',
                'sharpness': col_sharpness,
                'intervals': len(unique),
                'min_count': min_count
            }

    # Process categorical columns
    cat_data = []
    cat_info = {}

    for col in cat_cols:
        # Convert to categorical codes
        codes, uniques = pd.factorize(df[col], use_na_sentinel=True)

        # Handle NaN (pd.factorize gives -1 for NaN)
        if -1 in codes:
            # Add as separate category
            codes = np.where(codes == -1, len(uniques), codes)
            uniques = np.append(uniques, 'NaN')

        cat_data.append(codes.reshape(-1, 1))
        cat_info[col] = {
            'type': 'categorical',
            'categories': list(uniques),
            'n_categories': len(uniques)
        }

    # Combine all data
    if quant_data and cat_data:
        data_matrix = np.hstack(quant_data + cat_data)
    elif quant_data:
        data_matrix = np.hstack(quant_data)
    else:
        data_matrix = np.hstack(cat_data)

    # Target column (keep as is)
    target_data = df[target].values.reshape(-1, 1)

    # Final matrix
    final_matrix = np.hstack([data_matrix, target_data]).astype(np.int32)

    # Build info dict
    info = {
        'shape': final_matrix.shape,
        'columns': list(quant_info.keys()) + list(cat_info.keys()) + [target],
        'quantitative': quant_info,
        'categorical': cat_info,
        'target': target,
        'sharpness': sharpness,
    }

    if verbose:
        print(f"\nFinal matrix: {final_matrix.shape[0]} rows, {final_matrix.shape[1]} columns")

    return final_matrix, info