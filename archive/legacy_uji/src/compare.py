# src/compare.py
"""
Compare performance of different model types for indoor localization.
"""
import json
from pathlib import Path
import argparse
from time import time

from .data_io import load_uji
from .utils_geo import add_xy
from .baselines import (
    knn_building_pipeline, knn_floor_pipeline, knn_xy_pipeline,
    xgb_building_pipeline, xgb_xy_pipeline,
    mlp_building_pipeline, mlp_xy_pipeline
)
from .evaluate import evaluate_split


def train_and_evaluate_model(model_type, train_df, val_df):
    """Train and evaluate a specific model type."""
    print(f"\nTraining {model_type.upper()} model...")

    # Prepare features and targets
    wap_cols = [c for c in train_df.columns if c.startswith("WAP")]
    X_tr = train_df[wap_cols]
    X_va = val_df[wap_cols]

    y_b_tr = train_df['BUILDINGID']
    y_f_tr = train_df['FLOOR']
    xy_tr = train_df[['X_M', 'Y_M']].values

    y_b_va = val_df['BUILDINGID']
    y_f_va = val_df['FLOOR']
    xy_va = val_df[['X_M', 'Y_M']].values

    # Select pipelines based on model type
    start_time = time()
    if model_type == "knn":
        building_model = knn_building_pipeline().fit(X_tr, y_b_tr)
        floor_model = knn_floor_pipeline().fit(X_tr, y_f_tr)
        xy_model = knn_xy_pipeline().fit(X_tr, xy_tr)
    elif model_type == "xgb":
        building_model = xgb_building_pipeline().fit(X_tr, y_b_tr)
        floor_model = knn_floor_pipeline().fit(X_tr, y_f_tr)  # Keep kNN for floor
        xy_model = xgb_xy_pipeline().fit(X_tr, xy_tr)
    elif model_type == "mlp":
        building_model = mlp_building_pipeline().fit(X_tr, y_b_tr)
        floor_model = knn_floor_pipeline().fit(X_tr, y_f_tr)  # Keep kNN for floor
        xy_model = mlp_xy_pipeline().fit(X_tr, xy_tr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    training_time = time() - start_time

    # Evaluate
    print(f"Evaluating {model_type.upper()} model...")
    metrics = evaluate_split(building_model, floor_model, xy_model,
                             X_va, y_b_va, y_f_va, xy_va)

    return {
        'model_type': model_type,
        'training_time': training_time,
        'metrics': metrics,
        'predictions': {
            'building_true': y_b_va.tolist(),
            'building_pred': building_model.predict(X_va).tolist(),
            'floor_true': y_f_va.tolist(),
            'floor_pred': floor_model.predict(X_va).tolist(),
            'xy_true': xy_va.tolist(),
            'xy_pred': xy_model.predict(X_va).tolist()
        }
    }


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare indoor localization models"
    )
    parser.add_argument(
        "--models", nargs="+", choices=["knn", "xgb", "mlp"],
        default=["knn", "xgb", "mlp"],
        help="Model types to compare (default: all)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=10000,
        help="Sample size for training (default: 10000)"
    )
    parser.add_argument(
        "--output", type=str, default="comparison_results.json",
        help="Output file for results"
    )
    args = parser.parse_args()

    print("Indoor Localization - Model Comparison")
    print("=" * 50)
    print(f"Models to compare: {', '.join(args.models).upper()}")
    print(f"Sample size: {args.sample_size}")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "TrainingData.csv"
    val_path = data_dir / "ValidationData.csv"

    print("Loading data...")
    train, val = load_uji(str(train_path), str(val_path))

    # Sample for faster comparison
    if len(train) > args.sample_size:
        train = train.sample(args.sample_size, random_state=42)
    if len(val) > args.sample_size // 5:
        val = val.sample(args.sample_size // 5, random_state=42)

    print(f"Training: {len(train)} samples, Validation: {len(val)} samples")

    # Add coordinates
    train, center = add_xy(train)
    val, _ = add_xy(val)
    print(f"Reference point: lat={center[0]:.2f}, lon={center[1]:.2f}")

    # Compare models
    results = []
    for model_type in args.models:
        try:
            result = train_and_evaluate_model(model_type, train, val)
            results.append(result)
        except Exception as e:
            print(f"Error training {model_type.upper()} model: {e}")
            continue

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nComparison results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Model':<10} {'Building Acc':<12} {'Floor Acc':<10} " \
             f"{'Pos Error':<10} {'Time':<8}"
    print(header)
    print("-" * 80)

    for result in results:
        model = result['model_type'].upper()
        metrics = result['metrics']
        time_taken = result['training_time']

        building_acc = metrics.get('building_accuracy', 0)
        floor_acc = metrics.get('floor_accuracy', 0)
        pos_error = metrics.get('strict_2d_error_mean', float('inf'))

        row = f"{model:<10} {building_acc:<12.3f} {floor_acc:<10.3f} " \
              f"{pos_error:<10.1f} {time_taken:<8.1f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
