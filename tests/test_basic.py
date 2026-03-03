"""Basic tests for ih-prep."""

import pandas as pd
import numpy as np
from ih_prep import prepare_data

def test_basic_preparation():
    """Test with minimal data."""
    df = pd.DataFrame({
        'x': [1.0, 2.0, 3.0],
        'cat': ['a', 'b', 'c'],  # переименовал, чтобы не путать с y
        'target': [0, 1, 0]
    })
    
    data, info = prepare_data(df, target='target', sharpness=0.5)
    
    # Должно быть: x (колич) + cat (катег) + target = 3 колонки
    assert data.shape == (3, 3), f"Expected (3,3), got {data.shape}"
    assert data.dtype == np.int32
    assert 'target' in info['columns']
    print("✓ Basic test passed")

def test_categorical_na():
    """Test NaN handling."""
    df = pd.DataFrame({
        'cat': ['a', 'b', None],
        'target': [0, 1, 0]
    })
    
    data, info = prepare_data(df, target='target', sharpness=0.5)
    
    assert len(info['categorical']['cat']['categories']) == 3
    print("✓ NA test passed")

if __name__ == "__main__":
    test_basic_preparation()
    test_categorical_na()
    print("\n✅ All tests passed!")