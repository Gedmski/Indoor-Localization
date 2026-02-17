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


def load_bldg10(final_data_path: str):
    """Load and standardize the Building 10 dataset.

    The bldg10 dataset uses AP1..AP143 and includes room_id/floor labels.
    This function normalizes column names and adds expected fields so the
    existing pipeline can be reused.

    Args:
        final_data_path: Path to data/bldg10/final_data.csv

    Returns:
        Standardized pandas DataFrame with:
        - WAP* columns (renamed from AP*)
        - FLOOR (int)
        - ROOMID (string)
        - BUILDINGID (int, fixed to 10)
    """
    df = pd.read_csv(final_data_path)

    # Rename AP* -> WAP* for compatibility with feature pipeline
    ap_cols = [c for c in df.columns if c.startswith("AP")]
    rename_map = {c: f"WAP{c[2:]}" for c in ap_cols}
    df = df.rename(columns=rename_map)

    # Standardized labels
    if "floor" in df.columns:
        df["FLOOR"] = df["floor"].astype(int)
    if "room_id" in df.columns:
        df["ROOMID"] = df["room_id"].astype(str)

    # Single-building dataset
    df["BUILDINGID"] = 10

    return df


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
