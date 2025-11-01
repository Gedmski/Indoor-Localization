# src/utils_geo.py
from pyproj import Transformer
import numpy as np
import pandas as pd


def build_transformer(lat0: float, lon0: float):
    """Build a coordinate transformer from lat/lon to local ENU coordinates.

    Args:
        lat0: Reference latitude (degrees)
        lon0: Reference longitude (degrees)

    Returns:
        PyProj Transformer object
    """
    # Local tangent plane (ENU) around campus centroid
    return Transformer.from_crs(
        {"proj": "latlong", "datum": "WGS84"},
        {"proj": "aeqd", "lat_0": lat0, "lon_0": lon0, "x_0": 0, "y_0": 0,
         "units": "m", "no_defs": True},
        always_xy=True
    )


def add_xy(df: pd.DataFrame, lat_col='LATITUDE', lon_col='LONGITUDE'):
    """Add X_M and Y_M columns with local Cartesian coordinates.

    For UJIIndoorLoc, LATITUDE/LONGITUDE appear to already be in a projected
    coordinate system (likely UTM). This function creates local coordinates
    relative to the median point.

    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        Tuple of (modified_df, (lat0, lon0)) where lat0/lon0 are
        the reference point
    """
    lat0 = df[lat_col].median()
    lon0 = df[lon_col].median()

    # For UJIIndoorLoc, coordinates appear to already be projected
    # Just center them around the median
    df = df.copy()
    df['X_M'] = df[lon_col] - lon0  # Use longitude as X (East-West)
    df['Y_M'] = df[lat_col] - lat0  # Use latitude as Y (North-South)
    return df, (lat0, lon0)


def meter_error(y_true_xy, y_pred_xy):
    """Calculate Euclidean distance error in meters.

    Args:
        y_true_xy: True (x,y) coordinates, shape (n_samples, 2)
        y_pred_xy: Predicted (x,y) coordinates, shape (n_samples, 2)

    Returns:
        Array of distance errors in meters
    """
    return np.linalg.norm(y_true_xy - y_pred_xy, axis=1)


if __name__ == "__main__":
    # Quick test with sample data
    import pandas as pd
    import os
    from data_io import load_uji

    train_path = os.path.join(os.path.dirname(__file__), "..", "data",
                              "TrainingData.csv")
    val_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "ValidationData.csv")
    train, val = load_uji(train_path, val_path)
    print("Before adding XY coordinates:")
    print(train[['LATITUDE', 'LONGITUDE']].head())

    train_with_xy, center = add_xy(train)
    print(f"\nReference point: lat={center[0]:.6f}, lon={center[1]:.6f}")
    print("After adding XY coordinates:")
    print(train_with_xy[['LATITUDE', 'LONGITUDE', 'X_M', 'Y_M']].head())

    # Test error calculation
    true_xy = train_with_xy[['X_M', 'Y_M']].values[:5]
    pred_xy = true_xy + np.random.normal(0, 1, true_xy.shape)  # Add some noise
    errors = meter_error(true_xy, pred_xy)
    print(f"\nSample errors: {errors}")
