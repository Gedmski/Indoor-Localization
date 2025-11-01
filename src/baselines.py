# src/baselines.py
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from .features import RssiCleaner, APSelector

# Optional advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

from sklearn.neural_network import MLPClassifier, MLPRegressor


def knn_building_pipeline():
    """kNN pipeline for building classification."""
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("clf",   KNeighborsClassifier(n_neighbors=7, metric="cosine"))
    ])


def knn_floor_pipeline():
    """kNN pipeline for floor classification."""
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("clf",   KNeighborsClassifier(n_neighbors=7, metric="cosine"))
    ])


def knn_xy_pipeline():
    """kNN pipeline for (x,y) position regression."""
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("reg",   KNeighborsRegressor(n_neighbors=5, metric="euclidean",
                                      weights="distance"))
    ])


def knn_xy_per_building_pipelines():
    """Create separate kNN pipelines for each building.

    Returns:
        Dict mapping building_id to pipeline
    """
    pipelines = {}
    for building_id in [0, 1, 2]:  # UJIIndoorLoc buildings
        pipelines[building_id] = Pipeline([
            ("clean", RssiCleaner()),
            ("ap",    APSelector(coverage_min=0.02, top_k=200)),
            ("sc",    RobustScaler(with_centering=True)),
            ("reg",   KNeighborsRegressor(n_neighbors=5, metric="euclidean",
                                          weights="distance"))
        ])
    return pipelines


def xgb_building_pipeline(n_estimators=50, max_depth=4):
    """XGBoost pipeline for building classification (fast settings)."""
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is not installed")
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("clf",   XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 use_label_encoder=False, eval_metric='mlogloss'))
    ])


def xgb_xy_pipeline(n_estimators=50, max_depth=4):
    """XGBoost pipeline for XY regression (fast settings)."""
    if XGBRegressor is None:
        raise RuntimeError("XGBoost is not installed")
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("reg",   XGBRegressor(n_estimators=n_estimators, max_depth=max_depth))
    ])


def mlp_building_pipeline(hidden_layer_sizes=(50,), max_iter=200):
    """MLP pipeline for building classification."""
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("clf",   MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter))
    ])


def mlp_xy_pipeline(hidden_layer_sizes=(50,), max_iter=200):
    """MLP pipeline for XY regression."""
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("reg",   MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter))
    ])


if __name__ == "__main__":
    # Quick test with sample data
    import os
    import numpy as np
    from data_io import load_uji
    from utils_geo import add_xy

    train_path = os.path.join(os.path.dirname(__file__), "..", "data",
                              "TrainingData.csv")
    val_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "ValidationData.csv")
    train, val = load_uji(train_path, val_path)
    train, _ = add_xy(train)
    val, _ = add_xy(val)

    # Sample a small subset for testing
    train_sample = train.sample(1000, random_state=42)
    val_sample = val.sample(200, random_state=42)

    wap_cols = [c for c in train.columns if c.startswith("WAP")]
    X_tr = train_sample[wap_cols]
    X_va = val_sample[wap_cols]

    # Test building classification
    print("Testing kNN Building Classification...")
    building_model = knn_building_pipeline()
    building_model.fit(X_tr, train_sample['BUILDINGID'])
    building_pred = building_model.predict(X_va)
    building_acc = (building_pred == val_sample['BUILDINGID']).mean()
    print(f"Building accuracy: {building_acc:.3f}")

    # Test floor classification
    print("Testing kNN Floor Classification...")
    floor_model = knn_floor_pipeline()
    floor_model.fit(X_tr, train_sample['FLOOR'])
    floor_pred = floor_model.predict(X_va)
    floor_acc = (floor_pred == val_sample['FLOOR']).mean()
    print(f"Floor accuracy: {floor_acc:.3f}")

    # Test position regression
    print("Testing kNN Position Regression...")
    xy_model = knn_xy_pipeline()
    xy_tr = train_sample[['X_M', 'Y_M']].values
    xy_va = val_sample[['X_M', 'Y_M']].values
    xy_model.fit(X_tr, xy_tr)
    xy_pred = xy_model.predict(X_va)

    from utils_geo import meter_error
    pos_errors = meter_error(xy_va, xy_pred)
    print(f"Mean position error: {pos_errors.mean():.2f} meters")
    print(f"Median position error: {np.median(pos_errors):.2f} meters")
