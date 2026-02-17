# src/evaluate_bldg10.py
"""
Evaluation workflow for Building 10 dataset (final_data.csv).

This adapts the existing pipeline to:
- Use AP1..AP143 (renamed to WAP*)
- Predict ROOMID (room_id) instead of BUILDINGID
- Keep FLOOR classification
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

from .data_io import load_bldg10
from .baselines import (
    knn_building_pipeline,
    mlp_building_pipeline,
    xgb_building_pipeline,
    knn_floor_pipeline,
)


def _available_models():
    models = {
        "knn": knn_building_pipeline,
        "mlp": mlp_building_pipeline,
    }
    try:
        import xgboost  # noqa: F401
        models["xgb"] = xgb_building_pipeline
    except Exception:
        pass
    return models


def _cv_metrics(model_fn, X, y, cv, random_state=42):
    accs, precs, recs, f1s = [], [], [], []
    for train_idx, test_idx in cv.split(X, y):
        model = model_fn()
        if "mlp" in model_fn.__name__:
            model.set_params(clf__random_state=random_state)
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="macro", zero_division=0
        )
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision_macro_mean": float(np.mean(precs)),
        "precision_macro_std": float(np.std(precs)),
        "recall_macro_mean": float(np.mean(recs)),
        "recall_macro_std": float(np.std(recs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
    }


def _plot_confusion(cm, labels, title, out_path):
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


def _plot_distribution(counts, title, out_path, max_labels=30):
    plt.figure(figsize=(12, 6))
    counts = counts.sort_values(ascending=False)
    if len(counts) > max_labels:
        counts = counts.iloc[:max_labels]
    sns.barplot(x=counts.index, y=counts.values, color="#4c72b0")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_model_metrics(model_metrics, out_path):
    labels = list(model_metrics.keys())
    acc = [model_metrics[m]["accuracy"] for m in labels]
    f1 = [model_metrics[m]["f1_macro"] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, acc, width, label="Accuracy")
    plt.bar(x + width / 2, f1, width, label="Macro F1")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Room Classification (Holdout)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Building 10 dataset (room + floor classification)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/bldg10/final_data.csv",
        help="Path to final_data.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/bldg10",
        help="Directory to write reports and results",
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
        help="Holdout test size fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    reports_dir = Path(args.reports_dir)
    plots_dir = Path(args.plots_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_bldg10(str(data_path))
    wap_cols = [c for c in df.columns if c.startswith("WAP")]

    X = df[wap_cols]
    y_room = df["ROOMID"]
    y_floor = df["FLOOR"]

    # Holdout split for confusion matrices and plots
    X_train, X_test, y_room_train, y_room_test, y_floor_train, y_floor_test = (
        train_test_split(
            X,
            y_room,
            y_floor,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_room,
        )
    )

    models = _available_models()
    if not models.get("xgb"):
        notes = ["xgboost not available; xgb model skipped"]
    else:
        notes = []

    # Cross-validation for room classification
    room_counts = y_room.value_counts()
    min_room = int(room_counts.min())
    n_splits = min(5, min_room)
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=args.random_state
    )
    room_cv = {}
    for name, model_fn in models.items():
        room_cv[name] = _cv_metrics(
            model_fn, X, y_room, cv, random_state=args.random_state
        )

    # Holdout evaluation for room models
    room_holdout = {}
    for name, model_fn in models.items():
        model = model_fn()
        if name == "mlp":
            model.set_params(clf__random_state=args.random_state)
        model.fit(X_train, y_room_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_room_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_room_test, pred, average="macro", zero_division=0
        )
        room_holdout[name] = {
            "accuracy": float(acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    # Floor classification (kNN as pipeline default)
    floor_model = knn_floor_pipeline()
    floor_model.fit(X_train, y_floor_train)
    floor_pred = floor_model.predict(X_test)
    floor_acc = accuracy_score(y_floor_test, floor_pred)
    floor_precision, floor_recall, floor_f1, _ = precision_recall_fscore_support(
        y_floor_test, floor_pred, average="macro", zero_division=0
    )
    floor_holdout = {
        "accuracy": float(floor_acc),
        "precision_macro": float(floor_precision),
        "recall_macro": float(floor_recall),
        "f1_macro": float(floor_f1),
    }

    # Choose best room model by macro F1 on holdout
    best_room_model = max(room_holdout.items(), key=lambda x: x[1]["f1_macro"])[0]
    best_model = models[best_room_model]()
    if best_room_model == "mlp":
        best_model.set_params(clf__random_state=args.random_state)
    best_model.fit(X_train, y_room_train)
    best_pred = best_model.predict(X_test)

    # Confusion matrices (normalized)
    room_labels = sorted(y_room.unique().tolist())
    cm_room = confusion_matrix(
        y_room_test, best_pred, labels=room_labels, normalize="true"
    )
    cm_floor = confusion_matrix(
        y_floor_test,
        floor_pred,
        labels=sorted(y_floor.unique().tolist()),
        normalize="true",
    )

    _plot_distribution(
        room_counts,
        "Room Class Distribution (Top 30)",
        plots_dir / "room_class_distribution.png",
    )
    _plot_confusion(
        cm_room,
        room_labels,
        f"Room Confusion Matrix (Normalized) - {best_room_model.upper()}",
        plots_dir / f"room_confusion_matrix_{best_room_model}.png",
    )
    _plot_confusion(
        cm_floor,
        sorted(y_floor.unique().tolist()),
        "Floor Confusion Matrix (Normalized) - KNN",
        plots_dir / "floor_confusion_matrix_knn.png",
    )
    _plot_model_metrics(room_holdout, plots_dir / "room_model_metrics.png")

    results = {
        "dataset_info": {
            "samples": int(len(df)),
            "wap_features": int(len(wap_cols)),
            "room_classes": int(y_room.nunique()),
            "room_class_min_count": int(min_room),
            "room_class_counts": room_counts.to_dict(),
            "floor_classes": int(y_floor.nunique()),
            "floor_class_counts": y_floor.value_counts().to_dict(),
            "cv_folds_room": int(n_splits),
        },
        "room_cv": room_cv,
        "room_holdout": room_holdout,
        "floor_holdout": floor_holdout,
        "best_room_model": best_room_model,
        "notes": notes,
    }

    # Save results JSON
    results_path = reports_dir / "bldg10_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write markdown report
    report_path = reports_dir / "bldg10_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Building 10 Evaluation Report\n\n")
        f.write("## Dataset Summary\n\n")
        f.write(f"- Samples: {len(df)}\n")
        f.write(f"- WAP features: {len(wap_cols)}\n")
        f.write(f"- Rooms: {y_room.nunique()}\n")
        f.write(f"- Floors: {y_floor.nunique()}\n\n")

        f.write("## Room Classification (Cross-Validation)\n\n")
        f.write("| Model | Accuracy (mean+/-std) | Macro F1 (mean+/-std) |\n")
        f.write("|---|---|---|\n")
        for name, metrics in room_cv.items():
            acc = f"{metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}"
            f1 = f"{metrics['f1_macro_mean']:.4f} +/- {metrics['f1_macro_std']:.4f}"
            f.write(f"| {name.upper()} | {acc} | {f1} |\n")
        f.write("\n")

        f.write("## Holdout Results\n\n")
        f.write("### Room Classification\n\n")
        f.write("| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
        f.write("|---|---|---|---|---|\n")
        for name, metrics in room_holdout.items():
            f.write(
                f"| {name.upper()} | {metrics['accuracy']:.4f} | "
                f"{metrics['precision_macro']:.4f} | {metrics['recall_macro']:.4f} | "
                f"{metrics['f1_macro']:.4f} |\n"
            )
        f.write("\n")

        f.write("### Floor Classification (KNN)\n\n")
        f.write("| Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| {floor_holdout['accuracy']:.4f} | {floor_holdout['precision_macro']:.4f} | "
            f"{floor_holdout['recall_macro']:.4f} | {floor_holdout['f1_macro']:.4f} |\n"
        )
        f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("Room class distribution:\n\n")
        f.write("![](plots/room_class_distribution.png)\n\n")
        f.write("Room confusion matrix (normalized):\n\n")
        f.write(f"![](plots/room_confusion_matrix_{best_room_model}.png)\n\n")
        f.write("Floor confusion matrix (normalized):\n\n")
        f.write("![](plots/floor_confusion_matrix_knn.png)\n\n")
        f.write("Room model comparison (holdout):\n\n")
        f.write("![](plots/room_model_metrics.png)\n\n")

        if notes:
            f.write("## Notes\n\n")
            for note in notes:
                f.write(f"- {note}\n")

    print(f"Results saved to {results_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
