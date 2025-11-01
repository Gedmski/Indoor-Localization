# src/train.py
from pathlib import Path
from joblib import dump
import os
import argparse

from .data_io import load_uji
from .utils_geo import add_xy
from .baselines import (
    knn_building_pipeline, knn_floor_pipeline, knn_xy_pipeline,
    xgb_building_pipeline, xgb_xy_pipeline,
    mlp_building_pipeline, mlp_xy_pipeline
)
from .evaluate import evaluate_split, evaluate_hierarchical


def train_baseline_models(train_df, val_df, save_path="models", model_type="knn"):
    """Train baseline models and save them.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        save_path: Directory to save models
        model_type: Type of model to train ('knn', 'xgb', 'mlp')

    Returns:
        Dict with trained models
    """
    print(f"Training {model_type.upper()} models...")

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
    if model_type == "knn":
        building_pipeline = knn_building_pipeline
        xy_pipeline = knn_xy_pipeline
    elif model_type == "xgb":
        building_pipeline = xgb_building_pipeline
        xy_pipeline = xgb_xy_pipeline
    elif model_type == "mlp":
        building_pipeline = mlp_building_pipeline
        xy_pipeline = mlp_xy_pipeline
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train models
    building_model = building_pipeline().fit(X_tr, y_b_tr)
    floor_model = knn_floor_pipeline().fit(X_tr, y_f_tr)  # Keep kNN for floor
    xy_model = xy_pipeline().fit(X_tr, xy_tr)

    # Also train per-building position models
    xy_per_building = {}
    for building_id in train_df['BUILDINGID'].unique():
        mask = train_df['BUILDINGID'] == building_id
        if mask.sum() > 10:  # Only train if enough samples
            X_building = train_df.loc[mask, wap_cols]
            xy_building = train_df.loc[mask, ['X_M', 'Y_M']].values
            model = xy_pipeline().fit(X_building, xy_building)
            xy_per_building[building_id] = model

    models = {
        'building': building_model,
        'floor': floor_model,
        'xy_global': xy_model,
        'xy_per_building': xy_per_building
    }

    # Save models
    os.makedirs(save_path, exist_ok=True)
    dump(building_model, f"{save_path}/building_{model_type}.joblib")
    dump(floor_model, f"{save_path}/floor_{model_type}.joblib")
    dump(xy_model, f"{save_path}/xy_{model_type}_global.joblib")

    for building_id, model in xy_per_building.items():
        dump(model, f"{save_path}/xy_{model_type}_building_{building_id}.joblib")

    print(f"Models saved to {save_path}/")

    # Evaluate
    print("\nEvaluating global models...")
    metrics = evaluate_split(building_model, floor_model, xy_model,
                             X_va, y_b_va, y_f_va, xy_va)

    print("Global Model Results:")
    for k, v in metrics.items():
        if 'pct' in k:
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v:.3f}")

    # Evaluate hierarchical models
    if xy_per_building:
        print("\nEvaluating hierarchical models...")
        hierarchical_metrics = evaluate_hierarchical(
            building_model, {'global': floor_model}, xy_per_building,
            X_va, y_b_va, y_f_va, xy_va)

        print("Hierarchical Model Results:")
        for k, v in hierarchical_metrics.items():
            if 'pct' in k:
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v:.3f}")

    return models


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train indoor localization models")
    parser.add_argument("--model", choices=["knn", "xgb", "mlp"], default="knn",
                       help="Model type to train (default: knn)")
    args = parser.parse_args()

    print("Indoor Localization Training Pipeline")
    print("=" * 50)
    print(f"Model type: {args.model.upper()}")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "TrainingData.csv"
    val_path = data_dir / "ValidationData.csv"

    print(f"Loading data from {data_dir}")
    train, val = load_uji(str(train_path), str(val_path))
    train_samples = train.shape[0]
    val_samples = val.shape[0]
    print(f"Training: {train_samples} samples, "
          f"Validation: {val_samples} samples")

    # Add coordinate transformation
    print("Converting coordinates to local Cartesian system...")
    train, center = add_xy(train)
    val, _ = add_xy(val)
    print(f"Reference point: lat={center[0]:.2f}, lon={center[1]:.2f}")

    # Train models
    _ = train_baseline_models(train, val, model_type=args.model)

    print("\nTraining completed successfully!")
    print(f"Models saved in models/ directory")
    print(f"\nTo run inference, use: python src/serve.py --model {args.model}")


if __name__ == "__main__":
    main()
