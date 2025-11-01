# src/tune.py
"""
Hyperparameter tuning for indoor localization models.
"""
import os
import json
from pathlib import Path
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np

from .data_io import load_uji
from .utils_geo import add_xy
from .baselines import (
    knn_building_pipeline, knn_xy_pipeline,
    xgb_building_pipeline, xgb_xy_pipeline,
    mlp_building_pipeline, mlp_xy_pipeline
)
from .metrics import evaluate_position_metrics


def position_error_scorer(y_true, y_pred):
    """Scorer for position error (lower is better)."""
    from .utils_geo import meter_error
    return -meter_error(y_true, y_pred).mean()  # Negative for minimization


def tune_knn_models(X_train, y_building_train, y_floor_train, xy_train,
                   X_val, y_building_val, y_floor_val, xy_val):
    """Tune kNN hyperparameters."""
    print("Tuning kNN hyperparameters...")

    # kNN hyperparameter grid
    knn_params = {
        'building__clf__n_neighbors': [3, 5, 7, 9, 11],
        'xy__reg__n_neighbors': [3, 5, 7, 9, 11],
        'xy__reg__weights': ['uniform', 'distance']
    }

    # Create pipelines for tuning
    building_pipe = knn_building_pipeline()
    xy_pipe = knn_xy_pipeline()

    # Tune building classifier
    print("  Tuning building classifier...")
    building_grid = GridSearchCV(
        building_pipe, {k: v for k, v in knn_params.items() if 'building' in k},
        cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    building_grid.fit(X_train, y_building_train)

    # Tune XY regressor
    print("  Tuning XY regressor...")
    xy_grid = GridSearchCV(
        xy_pipe, {k: v for k, v in knn_params.items() if 'xy' in k},
        cv=3, scoring=make_scorer(position_error_scorer), n_jobs=-1, verbose=1
    )
    xy_grid.fit(X_train, xy_train)

    return {
        'building': {
            'best_params': building_grid.best_params_,
            'best_score': building_grid.best_score_,
            'val_score': building_grid.score(X_val, y_building_val)
        },
        'xy': {
            'best_params': xy_grid.best_params_,
            'best_score': xy_grid.best_score_,
            'val_score': xy_grid.score(X_val, xy_val)
        }
    }


def tune_xgb_models(X_train, y_building_train, y_floor_train, xy_train,
                   X_val, y_building_val, y_floor_val, xy_val):
    """Tune XGBoost hyperparameters."""
    print("Tuning XGBoost hyperparameters...")

    # XGBoost hyperparameter grid (conservative for speed)
    xgb_params = {
        'building__clf__n_estimators': [50, 100],
        'building__clf__max_depth': [3, 4, 5],
        'building__clf__learning_rate': [0.1, 0.2],
        'xy__reg__n_estimators': [50, 100],
        'xy__reg__max_depth': [3, 4, 5],
        'xy__reg__learning_rate': [0.1, 0.2]
    }

    # Create pipelines for tuning
    building_pipe = xgb_building_pipeline()
    xy_pipe = xgb_xy_pipeline()

    # Tune building classifier
    print("  Tuning building classifier...")
    building_grid = GridSearchCV(
        building_pipe, {k: v for k, v in xgb_params.items() if 'building' in k},
        cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    building_grid.fit(X_train, y_building_train)

    # Tune XY regressor
    print("  Tuning XY regressor...")
    xy_grid = GridSearchCV(
        xy_pipe, {k: v for k, v in xgb_params.items() if 'xy' in k},
        cv=3, scoring=make_scorer(position_error_scorer), n_jobs=-1, verbose=1
    )
    xy_grid.fit(X_train, xy_train)

    return {
        'building': {
            'best_params': building_grid.best_params_,
            'best_score': building_grid.best_score_,
            'val_score': building_grid.score(X_val, y_building_val)
        },
        'xy': {
            'best_params': xy_grid.best_params_,
            'best_score': xy_grid.best_score_,
            'val_score': xy_grid.score(X_val, xy_val)
        }
    }


def tune_mlp_models(X_train, y_building_train, y_floor_train, xy_train,
                   X_val, y_building_val, y_floor_val, xy_val):
    """Tune MLP hyperparameters."""
    print("Tuning MLP hyperparameters...")

    # MLP hyperparameter grid
    mlp_params = {
        'building__clf__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'building__clf__alpha': [0.0001, 0.001, 0.01],
        'xy__reg__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'xy__reg__alpha': [0.0001, 0.001, 0.01]
    }

    # Create pipelines for tuning
    building_pipe = mlp_building_pipeline()
    xy_pipe = mlp_xy_pipeline()

    # Tune building classifier
    print("  Tuning building classifier...")
    building_grid = GridSearchCV(
        building_pipe, {k: v for k, v in mlp_params.items() if 'building' in k},
        cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    building_grid.fit(X_train, y_building_train)

    # Tune XY regressor
    print("  Tuning XY regressor...")
    xy_grid = GridSearchCV(
        xy_pipe, {k: v for k, v in mlp_params.items() if 'xy' in k},
        cv=3, scoring=make_scorer(position_error_scorer), n_jobs=-1, verbose=1
    )
    xy_grid.fit(X_train, xy_train)

    return {
        'building': {
            'best_params': building_grid.best_params_,
            'best_score': building_grid.best_score_,
            'val_score': building_grid.score(X_val, y_building_val)
        },
        'xy': {
            'best_params': xy_grid.best_params_,
            'best_score': xy_grid.best_score_,
            'val_score': xy_grid.score(X_val, xy_val)
        }
    }


def main():
    """Main tuning function."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for indoor localization models")
    parser.add_argument("--model", choices=["knn", "xgb", "mlp"], required=True,
                       help="Model type to tune")
    parser.add_argument("--sample-size", type=int, default=5000,
                       help="Sample size for tuning (default: 5000)")
    parser.add_argument("--output", type=str, default="tuning_results.json",
                       help="Output file for results (default: tuning_results.json)")
    args = parser.parse_args()

    print("Indoor Localization - Hyperparameter Tuning")
    print("=" * 50)
    print(f"Model type: {args.model.upper()}")
    print(f"Sample size: {args.sample_size}")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "TrainingData.csv"
    val_path = data_dir / "ValidationData.csv"

    print("Loading data...")
    train, val = load_uji(str(train_path), str(val_path))

    # Sample for faster tuning
    if len(train) > args.sample_size:
        train = train.sample(args.sample_size, random_state=42)
    if len(val) > args.sample_size // 5:
        val = val.sample(args.sample_size // 5, random_state=42)

    print(f"Training: {len(train)} samples, Validation: {len(val)} samples")

    # Add coordinates
    train, _ = add_xy(train)
    val, _ = add_xy(val)

    # Prepare features
    wap_cols = [c for c in train.columns if c.startswith("WAP")]
    X_train = train[wap_cols]
    X_val = val[wap_cols]

    y_building_train = train['BUILDINGID']
    y_floor_train = train['FLOOR']
    xy_train = train[['X_M', 'Y_M']].values

    y_building_val = val['BUILDINGID']
    y_floor_val = val['FLOOR']
    xy_val = val[['X_M', 'Y_M']].values

    # Tune hyperparameters
    if args.model == "knn":
        results = tune_knn_models(X_train, y_building_train, y_floor_train, xy_train,
                                 X_val, y_building_val, y_floor_val, xy_val)
    elif args.model == "xgb":
        results = tune_xgb_models(X_train, y_building_train, y_floor_train, xy_train,
                                 X_val, y_building_val, y_floor_val, xy_val)
    elif args.model == "mlp":
        results = tune_mlp_models(X_train, y_building_train, y_floor_train, xy_train,
                                 X_val, y_building_val, y_floor_val, xy_val)

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, 'w') as f:
        json.dump({
            'model_type': args.model,
            'sample_size': args.sample_size,
            'results': results
        }, f, indent=2, default=str)

    print(f"\nTuning results saved to {output_path}")

    # Print summary
    print("\nBest Parameters Summary:")
    for component, result in results.items():
        print(f"  {component.upper()}:")
        print(f"    Best CV Score: {result['best_score']:.4f}")
        print(f"    Val Score: {result['val_score']:.4f}")
        print(f"    Best Params: {result['best_params']}")


if __name__ == "__main__":
    main()