import argparse
import json
from pathlib import Path
from typing import Dict

from joblib import dump
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .baselines import knn_floor_pipeline, room_model_factories
from .data_io import BUILDING_ID, load_bldg10, sorted_ap_columns


def _classification_metrics(y_true, y_pred) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def train_bldg10_models(
    data_path: str,
    output_dir: str = "models/bldg10",
    room_model_name: str = "knn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train and persist Building 10 room and floor classifiers."""
    df = load_bldg10(data_path)
    ap_cols = sorted_ap_columns(df.columns)

    X = df[ap_cols]
    y_room = df["ROOMID"]
    y_floor = df["FLOOR"]

    (
        X_train,
        X_holdout,
        y_room_train,
        y_room_holdout,
        y_floor_train,
        y_floor_holdout,
    ) = train_test_split(
        X,
        y_room,
        y_floor,
        test_size=test_size,
        random_state=random_state,
        stratify=y_room,
    )

    room_factories = room_model_factories()
    if room_model_name not in room_factories:
        available = ", ".join(sorted(room_factories.keys()))
        raise ValueError(
            f"Unsupported room model: {room_model_name}. Available: {available}"
        )

    room_model = room_factories[room_model_name]().fit(X_train, y_room_train)
    floor_model = knn_floor_pipeline().fit(X_train, y_floor_train)

    room_pred = room_model.predict(X_holdout)
    floor_pred = floor_model.predict(X_holdout)

    room_metrics = _classification_metrics(y_room_holdout, room_pred)
    floor_metrics = _classification_metrics(y_floor_holdout, floor_pred)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    room_model_path = output_path / f"bldg10_room_{room_model_name}.joblib"
    floor_model_path = output_path / "bldg10_floor_knn.joblib"
    metadata_path = output_path / "metadata.json"

    dump(room_model, room_model_path)
    dump(floor_model, floor_model_path)

    metadata = {
        "dataset": "bldg10",
        "building_id": BUILDING_ID,
        "room_model": room_model_name,
        "floor_model": "knn",
        "ap_columns": ap_cols,
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "room_labels": [str(label) for label in room_model.classes_.tolist()],
        "floor_labels": [int(label) for label in floor_model.classes_.tolist()],
        "metrics_holdout": {"room": room_metrics, "floor": floor_metrics},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "room_model_path": str(room_model_path),
        "floor_model_path": str(floor_model_path),
        "metadata_path": str(metadata_path),
        "room_metrics": room_metrics,
        "floor_metrics": floor_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Building 10 indoor localization models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/bldg10/final_data.csv",
        help="Path to Building 10 dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/bldg10",
        help="Directory to store trained models and metadata",
    )
    parser.add_argument(
        "--room-model",
        choices=sorted(room_model_factories().keys()),
        default="knn",
        help="Model used for room classification",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for quick quality check",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for split"
    )
    args = parser.parse_args()

    result = train_bldg10_models(
        data_path=args.data,
        output_dir=args.output_dir,
        room_model_name=args.room_model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Building 10 training complete.")
    print(f"Room model: {result['room_model_path']}")
    print(f"Floor model: {result['floor_model_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Room holdout accuracy: {result['room_metrics']['accuracy']:.4f}")
    print(f"Floor holdout accuracy: {result['floor_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
