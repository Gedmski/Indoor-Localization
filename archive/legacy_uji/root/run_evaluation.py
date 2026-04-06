#!/usr/bin/env python3
"""
Run comprehensive evaluation of indoor localization models.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Run evaluation and document results."""
    print("Indoor Localization - Model Evaluation & Documentation")
    print("=" * 60)

    try:
        # Import evaluation function
        from evaluate import run_full_evaluation

        # Run comprehensive evaluation
        print("Running comprehensive model evaluation...")
        results = run_full_evaluation(
            model_types=["knn", "xgb", "mlp"],
            cv_folds=3,  # Use fewer folds for faster evaluation
            sample_size=2000  # Use smaller sample for faster evaluation
        )

        # Save results
        import json
        output_file = Path(__file__).parent / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        # Best models
        cv_results = results['cross_validation']
        best_building = max(
            [(m, cv_results[m]['building']['accuracy']['mean'])
             for m in cv_results],
            key=lambda x: x[1]
        )
        best_position = min(
            [(m, cv_results[m]['position']['error_mean'])
             for m in cv_results],
            key=lambda x: x[1]
        )

        print(f"Best Building Model: {best_building[0].upper()} "
              f"({best_building[1]:.3f})")
        print(f"Best Position Model: {best_position[0].upper()} "
              f"({best_position[1]:.2f}m)")

        # Ensemble performance
        ensemble_results = results['ensemble_evaluation']
        print("\nEnsemble Performance:")
        b_acc = ensemble_results['building_classification']['accuracy']
        p_err = ensemble_results['position_regression']['error']
        print(f"  Building: {b_acc:.3f}")
        print(f"  Position: {p_err:.2f}m")

        return True

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
