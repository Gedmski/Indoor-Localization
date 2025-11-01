import numpy as np
import pandas as pd

from src.baselines import (
    knn_building_pipeline,
    knn_floor_pipeline,
    knn_xy_pipeline,
)


def make_synthetic_wap_data(n_samples=100, n_waps=10):
    """Create synthetic WAP DataFrame with some missing (-100) values."""
    cols = [f"WAP{i:03d}" for i in range(1, n_waps + 1)]
    rng = np.random.default_rng(42)
    data = rng.integers(-90, -30, size=(n_samples, n_waps))
    # randomly set 30% to -100 (missing)
    mask = rng.random(data.shape) < 0.3
    data[mask] = -100
    df = pd.DataFrame(data, columns=cols)
    return df


def test_knn_pipelines_quick():
    X = make_synthetic_wap_data(80, 10)
    y_building = np.random.randint(0, 3, size=len(X))
    y_floor = np.random.randint(0, 5, size=len(X))
    xy = np.column_stack([
        np.random.normal(0, 50, size=len(X)),
        np.random.normal(0, 50, size=len(X)),
    ])

    # Train small kNN building pipeline
    b_pipe = knn_building_pipeline()
    b_pipe.fit(X, y_building)
    preds_b = b_pipe.predict(X)
    assert len(preds_b) == len(X)

    # Train small kNN floor pipeline
    f_pipe = knn_floor_pipeline()
    f_pipe.fit(X, y_floor)
    preds_f = f_pipe.predict(X)
    assert len(preds_f) == len(X)

    # Train small kNN xy pipeline
    xy_pipe = knn_xy_pipeline()
    xy_pipe.fit(X, xy)
    preds_xy = xy_pipe.predict(X)
    assert preds_xy.shape[0] == len(X)
