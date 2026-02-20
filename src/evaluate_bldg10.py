import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .baselines import knn_floor_pipeline, room_model_factories
from .data_io import load_bldg10, sorted_ap_columns


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


def _cross_validate_room_model(model_name: str, X, y, folds: int) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = []
    factory = room_model_factories()[model_name]
    for train_idx, test_idx in cv.split(X, y):
        model = factory()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics.append(_classification_metrics(y_test, pred))

    return {
        "accuracy_mean": float(np.mean([item["accuracy"] for item in metrics])),
        "accuracy_std": float(np.std([item["accuracy"] for item in metrics])),
        "f1_macro_mean": float(np.mean([item["f1_macro"] for item in metrics])),
        "f1_macro_std": float(np.std([item["f1_macro"] for item in metrics])),
    }


def _plot_confusion(cm, labels: List[str], title: str, out_path: Path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_distribution(counts, title: str, out_path: Path, max_labels: int = 30):
    plt.figure(figsize=(12, 6))
    sorted_counts = counts.sort_values(ascending=False)
    if len(sorted_counts) > max_labels:
        sorted_counts = sorted_counts.iloc[:max_labels]
    sns.barplot(x=sorted_counts.index, y=sorted_counts.values, color="#4c72b0")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_markdown_report(
    report_path: Path,
    dataset_info: Dict[str, object],
    room_cv: Dict[str, Dict[str, float]],
    room_holdout: Dict[str, Dict[str, float]],
    floor_holdout: Dict[str, float],
    best_room_model: str,
):
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Building 10 Evaluation Report\n\n")
        handle.write("## Dataset Summary\n\n")
        handle.write(f"- Samples: {dataset_info['samples']}\n")
        handle.write(f"- AP features: {dataset_info['ap_features']}\n")
        handle.write(f"- Room classes: {dataset_info['room_classes']}\n")
        handle.write(f"- Floor classes: {dataset_info['floor_classes']}\n\n")

        handle.write("## Room Classification (Cross-Validation)\n\n")
        handle.write("| Model | Accuracy (mean+/-std) | Macro F1 (mean+/-std) |\n")
        handle.write("|---|---|---|\n")
        for model_name, metrics in room_cv.items():
            acc = f"{metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}"
            f1 = f"{metrics['f1_macro_mean']:.4f} +/- {metrics['f1_macro_std']:.4f}"
            handle.write(f"| {model_name.upper()} | {acc} | {f1} |\n")
        handle.write("\n")

        handle.write("## Holdout Results\n\n")
        handle.write("### Room Classification\n\n")
        handle.write("| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
        handle.write("|---|---|---|---|---|\n")
        for model_name, metrics in room_holdout.items():
            handle.write(
                f"| {model_name.upper()} | {metrics['accuracy']:.4f} | "
                f"{metrics['precision_macro']:.4f} | {metrics['recall_macro']:.4f} | "
                f"{metrics['f1_macro']:.4f} |\n"
            )
        handle.write("\n")

        handle.write("### Floor Classification (KNN)\n\n")
        handle.write("| Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
        handle.write("|---|---|---|---|\n")
        handle.write(
            f"| {floor_holdout['accuracy']:.4f} | {floor_holdout['precision_macro']:.4f} | "
            f"{floor_holdout['recall_macro']:.4f} | {floor_holdout['f1_macro']:.4f} |\n\n"
        )

        handle.write("## Visualizations\n\n")
        handle.write("Room class distribution:\n\n")
        handle.write("![](plots/room_class_distribution.png)\n\n")
        handle.write("Room confusion matrix (normalized):\n\n")
        handle.write(f"![](plots/room_confusion_matrix_{best_room_model}.png)\n\n")
        handle.write("Floor confusion matrix (normalized):\n\n")
        handle.write("![](plots/floor_confusion_matrix_knn.png)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Building 10 room/floor localization models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/bldg10/final_data.csv",
        help="Path to final_data.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["knn", "mlp"],
        help="Room classification models to evaluate",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/bldg10",
        help="Directory to write report outputs",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="reports/bldg10/plots",
        help="Directory to write plots",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    plots_dir = Path(args.plots_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_bldg10(args.data)
    ap_cols = sorted_ap_columns(df.columns)
    X = df[ap_cols]
    y_room = df["ROOMID"]
    y_floor = df["FLOOR"]

    room_counts = y_room.value_counts()
    min_room_count = int(room_counts.min())
    cv_folds = min(5, min_room_count)
    if cv_folds < 2:
        raise ValueError("Not enough samples per room for cross-validation.")

    model_factories = room_model_factories()
    selected_models = [name for name in args.models if name in model_factories]
    if not selected_models:
        raise ValueError("No valid room models selected for evaluation.")

    (
        X_train,
        X_test,
        y_room_train,
        y_room_test,
        y_floor_train,
        y_floor_test,
    ) = train_test_split(
        X,
        y_room,
        y_floor,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_room,
    )

    room_cv = {}
    room_holdout = {}
    fitted_room_models = {}
    for model_name in selected_models:
        room_cv[model_name] = _cross_validate_room_model(
            model_name, X, y_room, folds=cv_folds
        )

        model = model_factories[model_name]()
        model.fit(X_train, y_room_train)
        pred = model.predict(X_test)
        room_holdout[model_name] = _classification_metrics(y_room_test, pred)
        fitted_room_models[model_name] = (model, pred)

    floor_model = knn_floor_pipeline().fit(X_train, y_floor_train)
    floor_pred = floor_model.predict(X_test)
    floor_holdout = _classification_metrics(y_floor_test, floor_pred)

    best_room_model = max(
        room_holdout.items(), key=lambda item: item[1]["f1_macro"]
    )[0]
    _, best_room_pred = fitted_room_models[best_room_model]

    room_labels = sorted(y_room.unique().tolist())
    floor_labels = sorted(y_floor.unique().tolist())
    room_cm = confusion_matrix(
        y_room_test, best_room_pred, labels=room_labels, normalize="true"
    )
    floor_cm = confusion_matrix(
        y_floor_test, floor_pred, labels=floor_labels, normalize="true"
    )

    _plot_distribution(
        room_counts,
        "Room Class Distribution (Top 30)",
        plots_dir / "room_class_distribution.png",
    )
    _plot_confusion(
        room_cm,
        room_labels,
        f"Room Confusion Matrix (Normalized) - {best_room_model.upper()}",
        plots_dir / f"room_confusion_matrix_{best_room_model}.png",
    )
    _plot_confusion(
        floor_cm,
        [str(label) for label in floor_labels],
        "Floor Confusion Matrix (Normalized) - KNN",
        plots_dir / "floor_confusion_matrix_knn.png",
    )

    results = {
        "dataset_info": {
            "samples": int(len(df)),
            "ap_features": int(len(ap_cols)),
            "room_classes": int(y_room.nunique()),
            "room_class_min_count": min_room_count,
            "room_class_counts": {str(k): int(v) for k, v in room_counts.items()},
            "floor_classes": int(y_floor.nunique()),
            "floor_class_counts": {
                str(k): int(v) for k, v in y_floor.value_counts().items()
            },
            "cv_folds_room": int(cv_folds),
        },
        "room_cv": room_cv,
        "room_holdout": room_holdout,
        "floor_holdout": floor_holdout,
        "best_room_model": best_room_model,
        "building_id": 10,
    }

    results_path = reports_dir / "bldg10_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report_path = reports_dir / "bldg10_evaluation_report.md"
    _write_markdown_report(
        report_path=report_path,
        dataset_info=results["dataset_info"],
        room_cv=room_cv,
        room_holdout=room_holdout,
        floor_holdout=floor_holdout,
        best_room_model=best_room_model,
    )

    print(f"Results saved to {results_path}")
    print(f"Report saved to {report_path}")
    print(f"Best room model: {best_room_model.upper()}")
    print(f"Room holdout accuracy: {room_holdout[best_room_model]['accuracy']:.4f}")
    print(f"Floor holdout accuracy: {floor_holdout['accuracy']:.4f}")


if __name__ == "__main__":
    main()
