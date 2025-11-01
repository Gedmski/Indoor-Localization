# src/data_io.py
import pandas as pd


def load_uji(train_path: str, val_path: str):
    """Load UJIIndoorLoc training and validation datasets.

    Args:
        train_path: Path to TrainingData.csv
        val_path: Path to ValidationData.csv

    Returns:
        Tuple of (train_df, val_df) pandas DataFrames
    """
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    return train, val


if __name__ == "__main__":
    # Quick test
    import os
    print(f"Current working directory: {os.getcwd()}")
    train_path = os.path.join(os.path.dirname(__file__), "..", "data",
                              "TrainingData.csv")
    val_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "ValidationData.csv")
    print(f"Train path: {os.path.abspath(train_path)}")
    print(f"Val path: {os.path.abspath(val_path)}")
    train, val = load_uji(train_path, val_path)
    print(f"Training shape: {train.shape}, Validation shape: {val.shape}")
    print(f"Training columns (first 10): {train.columns[:10].tolist()}")
    print("Training BUILDINGID, FLOOR, LATITUDE, LONGITUDE describe:")
    print(train[['BUILDINGID', 'FLOOR', 'LATITUDE', 'LONGITUDE']].describe())
