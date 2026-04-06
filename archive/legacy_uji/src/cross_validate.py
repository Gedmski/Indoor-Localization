# src/cross_validate.py
"""
Cross-validation for indoor localization models.
"""
import json
from pathlib import Path
import argparse
from sklearn.model_selection import cross_validate
import pandas as pd

from .data_io import load_uji
from .utils_geo import add_xy
from .baselines import (
    knn_building_pipeline, knn_floor_pipeline, knn_xy_pipeline,
    xgb_building_pipeline, xgb_xy_pipeline,
    mlp_building_pipeline, mlp_xy_pipeline
)


def position_error_scorer(estimator, X, y):
    """Custom scorer for position error."""
    from .utils_geo import meter_error
    y_pred = estimator.predict(X)
    return -meter_error(y, y_pred).mean()  # Negative for minimization


def cross_validate_model(model_type, X, y_building, y_floor, xy, cv_folds=5):
    """Cross-validate a specific model type."""
    print(f"Cross-validating {model_type.upper()} model with "
          f"{cv_folds} folds...")

    # Select pipelines
    if model_type == "knn":
        building_pipe = knn_building_pipeline()
        xy_pipe = knn_xy_pipeline()
    elif model_type == "xgb":
        building_pipe = xgb_building_pipeline()
        xy_pipe = xgb_xy_pipeline()
    elif model_type == "mlp":
        building_pipe = mlp_building_pipeline()
        xy_pipe = mlp_xy_pipeline()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Cross-validation for building classification
    print("  Building classification CV...")
    building_scores = cross_validate(
        building_pipe, X, y_building,
        cv=cv_folds,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score=False, n_jobs=-1
    )

    # Cross-validation for position regression
    print("  Position regression CV...")
    xy_scores = cross_validate(
        xy_pipe, X, xy,
        cv=cv_folds, scoring={'pos_error': position_error_scorer},
        return_train_score=False, n_jobs=-1
    )

    # Floor classification (always kNN for consistency)
    print("  Floor classification CV...")
    floor_pipe = knn_floor_pipeline()
    floor_scores = cross_validate(
        floor_pipe, X, y_floor,
        cv=cv_folds,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score=False, n_jobs=-1
    )

    return {
        'building': {
            'accuracy': {
                'mean': building_scores['test_accuracy'].mean(),
                'std': building_scores['test_accuracy'].std(),
                'scores': building_scores['test_accuracy'].tolist()
            },
            'precision': {
                'mean': building_scores['test_precision_macro'].mean(),
                'std': building_scores['test_precision_macro'].std()
            },
            'recall': {
                'mean': building_scores['test_recall_macro'].mean(),
                'std': building_scores['test_recall_macro'].std()
            },
            'f1': {
                'mean': building_scores['test_f1_macro'].mean(),
                'std': building_scores['test_f1_macro'].std()
            }
        },
        'floor': {
            'accuracy': {
                'mean': floor_scores['test_accuracy'].mean(),
                'std': floor_scores['test_accuracy'].std(),
                'scores': floor_scores['test_accuracy'].tolist()
            },
            'precision': {
                'mean': floor_scores['test_precision_macro'].mean(),
                'std': floor_scores['test_precision_macro'].std()
            },
            'recall': {
                'mean': floor_scores['test_recall_macro'].mean(),
                'std': floor_scores['test_recall_macro'].std()
            },
            'f1': {
                'mean': floor_scores['test_f1_macro'].mean(),
                'std': floor_scores['test_f1_macro'].std()
            }
        },
        'position': {
            'error_mean': -xy_scores['test_pos_error'].mean(),
            'error_std': xy_scores['test_pos_error'].std(),
            'error_scores': (-xy_scores['test_pos_error']).tolist()
        }
    }


def main():
    """Main cross-validation function."""
    parser = argparse.ArgumentParser(
        description="Cross-validate indoor localization models"
    )
    parser.add_argument(
        "--model", choices=["knn", "xgb", "mlp"], required=True,
        help="Model type to cross-validate"
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=5000,
        help="Sample size for CV (default: 5000)"
    )
    parser.add_argument(
        "--output", type=str, default="cv_results.json",
        help="Output file for results"
    )
    args = parser.parse_args()

    print("Indoor Localization - Cross-Validation")
    print("=" * 50)
    print(f"Model type: {args.model.upper()}")
    print(f"CV folds: {args.folds}")
    print(f"Sample size: {args.sample_size}")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "TrainingData.csv"
    val_path = data_dir / "ValidationData.csv"

    print("Loading data...")
    train, val = load_uji(str(train_path), str(val_path))

    # Combine train and val for CV
    combined = pd.concat([train, val], ignore_index=True)

    # Sample for faster CV
    if len(combined) > args.sample_size:
        combined = combined.sample(args.sample_size, random_state=42)

    print(f"Dataset: {len(combined)} samples")

    # Add coordinates
    combined, center = add_xy(combined)
    print(f"Reference point: lat={center[0]:.2f}, lon={center[1]:.2f}")

    # Prepare features and targets
    wap_cols = [c for c in combined.columns if c.startswith("WAP")]
    X = combined[wap_cols]
    y_building = combined['BUILDINGID']
    y_floor = combined['FLOOR']
    xy = combined[['X_M', 'Y_M']].values

    # Cross-validate
    results = cross_validate_model(
        args.model, X, y_building, y_floor, xy, args.folds
    )

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, 'w') as f:
        json.dump({
            'model_type': args.model,
            'cv_folds': args.folds,
            'sample_size': args.sample_size,
            'results': results
        }, f, indent=2, default=str)

    print(f"\nCV results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    b_results = results['building']
    f_results = results['floor']
    p_results = results['position']

    print("Building Classification:")
    print(f"  Accuracy: {b_results['accuracy']['mean']:.3f} ± "
          f"{b_results['accuracy']['std']:.3f}")
    print(f"  F1-Score: {b_results['f1']['mean']:.3f} ± "
          f"{b_results['f1']['std']:.3f}")

    print("\nFloor Classification:")
    print(f"  Accuracy: {f_results['accuracy']['mean']:.3f} ± "
          f"{f_results['accuracy']['std']:.3f}")
    print(f"  F1-Score: {f_results['f1']['mean']:.3f} ± "
          f"{f_results['f1']['std']:.3f}")

    print("\nPosition Regression:")
    print(f"  Mean Error: {p_results['error_mean']:.2f} ± "
          f"{p_results['error_std']:.2f} meters")

    print("=" * 60)


if __name__ == "__main__":
    main()
