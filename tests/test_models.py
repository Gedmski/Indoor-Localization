import numpy as np
import pandas as pd

from src.baselines import knn_floor_pipeline, knn_room_pipeline, mlp_room_pipeline
from src.data_io import build_inference_frame, sorted_ap_columns


def make_synthetic_ap_data(n_samples=120, n_aps=20):
    columns = [f"AP{i}" for i in range(1, n_aps + 1)]
    rng = np.random.default_rng(42)
    data = rng.integers(-95, -35, size=(n_samples, n_aps))
    mask = rng.random(data.shape) < 0.3
    data[mask] = -100
    return pd.DataFrame(data, columns=columns)


def test_room_and_floor_pipelines_fit_predict():
    X = make_synthetic_ap_data()
    y_room = np.array([f"R{i%6}" for i in range(len(X))])
    y_floor = np.array([1 if i % 2 == 0 else 2 for i in range(len(X))])

    room_knn = knn_room_pipeline()
    room_knn.fit(X, y_room)
    room_pred = room_knn.predict(X)
    assert len(room_pred) == len(X)

    room_mlp = mlp_room_pipeline(max_iter=50)
    room_mlp.fit(X, y_room)
    room_pred_mlp = room_mlp.predict(X)
    assert len(room_pred_mlp) == len(X)

    floor_knn = knn_floor_pipeline()
    floor_knn.fit(X, y_floor)
    floor_pred = floor_knn.predict(X)
    assert len(floor_pred) == len(X)


def test_build_inference_frame_uses_sorted_ap_columns():
    X = make_synthetic_ap_data(n_samples=10, n_aps=5)
    ap_cols = sorted_ap_columns(X.columns)

    frame = build_inference_frame({"AP1": -60, "AP3": -75}, ap_cols)
    assert list(frame.columns) == ap_cols
    assert frame.iloc[0]["AP1"] == -60
    assert frame.iloc[0]["AP2"] == -100.0
