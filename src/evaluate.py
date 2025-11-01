# src/evaluate.py
"""
Comprehensive evaluation script for indoor localization models.
Combines cross-validation, model comparison, and ensemble evaluation.
"""
import json
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from .data_io import load_uji
from .utils_geo import add_xy
from .cross_validate import cross_validate_model
from .compare import compare_models
from .ensemble import evaluate_ensemble


def run_full_evaluation(model_types=None, cv_folds=5, sample_size=5000):
    """Run comprehensive evaluation of all models."""
    if model_types is None:
        model_types = ["knn", "xgb", "mlp"]

    print("Indoor Localization - Comprehensive Evaluation")
    print("=" * 60)
    print(f"Models: {', '.join(m.upper() for m in model_types)}")
    print(f"CV folds: {cv_folds}")
    print(f"Sample size: {sample_size}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load and prepare data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "TrainingData.csv"
    val_path = data_dir / "ValidationData.csv"

    print("Loading data...")
    train, val = load_uji(str(train_path), str(val_path))

    # Combine train and val for evaluation
    combined = pd.concat([train, val], ignore_index=True)

    # Sample for faster evaluation
    if len(combined) > sample_size:
        combined = combined.sample(sample_size, random_state=42)

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

    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(combined),
            'features': len(wap_cols),
            'buildings': len(y_building.unique()),
            'floors': len(y_floor.unique()),
            'reference_point': center.tolist()
        },
        'cross_validation': {},
        'model_comparison': {},
        'ensemble_evaluation': {}
    }

    # 1. Cross-validation for each model
    print("\n" + "=" * 60)
    print("PHASE 1: CROSS-VALIDATION")
    print("=" * 60)

    for model_type in model_types:
        print(f"\nCross-validating {model_type.upper()}...")
        cv_results = cross_validate_model(
            model_type, X, y_building, y_floor, xy, cv_folds
        )
        results['cross_validation'][model_type] = cv_results

        # Print summary
        b_res = cv_results['building']
        f_res = cv_results['floor']
        p_res = cv_results['position']

        print(f"  Building Acc: {b_res['accuracy']['mean']:.3f} ± "
              f"{b_res['accuracy']['std']:.3f}")
        print(f"  Floor Acc: {f_res['accuracy']['mean']:.3f} ± "
              f"{f_res['accuracy']['std']:.3f}")
        print(f"  Position Error: {p_res['error_mean']:.2f} ± "
              f"{p_res['error_std']:.2f} m")

    # 2. Model comparison
    print("\n" + "=" * 60)
    print("PHASE 2: MODEL COMPARISON")
    print("=" * 60)

    comparison_results = compare_models(
        X, y_building, y_floor, xy, model_types=model_types
    )
    results['model_comparison'] = comparison_results

    print("Model Comparison Results:")
    for model, metrics in comparison_results.items():
        print(f"\n{model.upper()}:")
        print(f"  Building Acc: {metrics['building_accuracy']:.3f}")
        print(f"  Floor Acc: {metrics['floor_accuracy']:.3f}")
        print(f"  Position Error: {metrics['position_error']:.2f} m")
        print(f"  Training Time: {metrics['training_time']:.2f} s")
        print(f"  Prediction Time: {metrics['prediction_time']:.4f} s")

    # 3. Ensemble evaluation
    print("\n" + "=" * 60)
    print("PHASE 3: ENSEMBLE EVALUATION")
    print("=" * 60)

    ensemble_results = evaluate_ensemble(
        X, y_building, y_floor, xy, model_types=model_types
    )
    results['ensemble_evaluation'] = ensemble_results

    print("Ensemble Results:")
    for task, metrics in ensemble_results.items():
        print(f"\n{task.replace('_', ' ').title()}:")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
        if 'error' in metrics:
            print(f"  Error: {metrics['error']:.2f} m")

    # 4. Summary and recommendations
    print("\n" + "=" * 60)
    print("PHASE 4: SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    # Find best models
    best_building = max(
        [(m, results['cross_validation'][m]['building']['accuracy']['mean'])
         for m in model_types],
        key=lambda x: x[1]
    )

    best_floor = max(
        [(m, results['cross_validation'][m]['floor']['accuracy']['mean'])
         for m in model_types],
        key=lambda x: x[1]
    )

    best_position = min(
        [(m, results['cross_validation'][m]['position']['error_mean'])
         for m in model_types],
        key=lambda x: x[1]
    )

    print("Best Models by Task:")
    print(f"  Building Classification: {best_building[0].upper()} "
          f"({best_building[1]:.3f})")
    print(f"  Floor Classification: {best_floor[0].upper()} "
          f"({best_floor[1]:.3f})")
    print(f"  Position Regression: {best_position[0].upper()} "
          f"({best_position[1]:.2f} m)")

    # Ensemble performance
    ensemble_building = ensemble_results['building_classification']['accuracy']
    ensemble_floor = ensemble_results['floor_classification']['accuracy']
    ensemble_position = ensemble_results['position_regression']['error']

    print("\nEnsemble Performance:")
    print(f"  Building Accuracy: {ensemble_building:.3f}")
    print(f"  Floor Accuracy: {ensemble_floor:.3f}")
    print(f"  Position Error: {ensemble_position:.2f} m")

    # Recommendations
    print("\nRecommendations:")
    if ensemble_building > best_building[1]:
        print("  ✓ Use ensemble for building classification")
    else:
        print(f"  ✓ Use {best_building[0].upper()} for building "
              f"classification")

    if ensemble_floor > best_floor[1]:
        print("  ✓ Use ensemble for floor classification")
    else:
        print(f"  ✓ Use {best_floor[0].upper()} for floor classification")

    if ensemble_position < best_position[1]:
        print("  ✓ Use ensemble for position regression")
    else:
        print(f"  ✓ Use {best_position[0].upper()} for position regression")

    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of indoor localization models"
    )
    parser.add_argument(
        "--models", nargs="+", choices=["knn", "xgb", "mlp"],
        default=["knn", "xgb", "mlp"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=5000,
        help="Sample size for evaluation (default: 5000)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Output file for results"
    )
    args = parser.parse_args()

    # Run evaluation
    results = run_full_evaluation(
        model_types=args.models,
        cv_folds=args.folds,
        sample_size=args.sample_size
    )

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nComplete evaluation results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
