# src/baselines.py
from typing import Callable, Dict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .features import APColumnSelector, RssiCleaner


def knn_room_pipeline(n_neighbors: int = 7) -> Pipeline:
    """kNN pipeline for room classification."""
    return Pipeline(
        [
            ("clean", RssiCleaner()),
            ("ap", APColumnSelector()),
            ("scale", RobustScaler(with_centering=True)),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors, metric="cosine", weights="distance"
                ),
            ),
        ]
    )


def mlp_room_pipeline(
    hidden_layer_sizes=(64, 32), max_iter: int = 400, random_state: int = 42
) -> Pipeline:
    """MLP pipeline for room classification."""
    return Pipeline(
        [
            ("clean", RssiCleaner()),
            ("ap", APColumnSelector()),
            ("scale", RobustScaler(with_centering=True)),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def knn_floor_pipeline(n_neighbors: int = 5) -> Pipeline:
    """kNN pipeline for floor classification."""
    return Pipeline(
        [
            ("clean", RssiCleaner()),
            ("ap", APColumnSelector()),
            ("scale", RobustScaler(with_centering=True)),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors, metric="cosine", weights="distance"
                ),
            ),
        ]
    )


def room_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    return {"knn": knn_room_pipeline, "mlp": mlp_room_pipeline}
