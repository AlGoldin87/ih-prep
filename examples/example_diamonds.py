"""Example usage with diamonds dataset."""

import pandas as pd
import numpy as np
from ih_prep import prepare_data

def main():
    # Load data
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
    df = df.head(100)  # Use first 100 rows for demo

    print("Original data shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)

    # Define sharpness for each quantitative column
    sharpness_map = {
        'carat': 0.3,
        'depth': 0.25,
        'table': 0.25,
        'x': 0.3,
        'y': 0.3,
        'z': 0.3
    }

    # Prepare data
    data, info = prepare_data(
        df=df,
        target='price',
        sharpness=sharpness_map,
        verbose=True
    )

    print("\nFinal data matrix shape:", data.shape)
    print("Data type:", data.dtype)
    print("\nFirst 5 rows:")
    print(data[:5])

    print("\nInfo dictionary:")
    for key, value in info.items():
        if key != 'quantitative' and key != 'categorical':
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()