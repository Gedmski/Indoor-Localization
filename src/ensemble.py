# src/ensemble.py
"""
Ensemble methods for indoor localization.
"""
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .features import RssiCleaner, APSelector
from .baselines import (
    knn_building_pipeline, knn_xy_pipeline,
    xgb_building_pipeline, xgb_xy_pipeline,
    mlp_building_pipeline, mlp_xy_pipeline
)


def ensemble_building_pipeline():
    """Ensemble pipeline for building classification using voting."""
    # Create base estimators
    estimators = [
        ('knn', knn_building_pipeline()),
        ('xgb', xgb_building_pipeline()),
        ('mlp', mlp_building_pipeline())
    ]

    # Create voting classifier
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')

    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("ensemble", voting_clf)
    ])


def ensemble_xy_pipeline():
    """Ensemble pipeline for XY regression using averaging."""
    # Create base estimators
    estimators = [
        ('knn', knn_xy_pipeline()),
        ('xgb', xgb_xy_pipeline()),
        ('mlp', mlp_xy_pipeline())
    ]

    # Create voting regressor (averaging)
    voting_reg = VotingRegressor(estimators=estimators)

    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("ensemble", voting_reg)
    ])


def weighted_ensemble_building_pipeline(weights=None):
    """Weighted ensemble for building classification."""
    if weights is None:
        weights = [0.5, 0.3, 0.2]  # Default weights favoring kNN

    estimators = [
        ('knn', knn_building_pipeline()),
        ('xgb', xgb_building_pipeline()),
        ('mlp', mlp_building_pipeline())
    ]

    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights
    )

    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("ensemble", voting_clf)
    ])


def weighted_ensemble_xy_pipeline(weights=None):
    """Weighted ensemble for XY regression."""
    if weights is None:
        weights = [0.5, 0.3, 0.2]  # Default weights favoring kNN

    estimators = [
        ('knn', knn_xy_pipeline()),
        ('xgb', xgb_xy_pipeline()),
        ('mlp', mlp_xy_pipeline())
    ]

    voting_reg = VotingRegressor(estimators=estimators, weights=weights)

    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("ensemble", voting_reg)
    ])


def hierarchical_ensemble_pipeline():
    """Hierarchical ensemble that uses building prediction to
    weight floor/position models."""
    # This is a more complex ensemble that could adapt weights
    # based on building confidence
    # For now, return a simple ensemble
    return {
        'building': ensemble_building_pipeline(),
        'xy': ensemble_xy_pipeline()
    }


if __name__ == "__main__":
    # Quick test
    from data_io import load_uji
    from utils_geo import add_xy

    # Load small sample
    train, val = load_uji('data/TrainingData.csv', 'data/ValidationData.csv')
    train_small = train.sample(1000, random_state=42)
    val_small = val.sample(200, random_state=42)

    train_small, _ = add_xy(train_small)
    val_small, _ = add_xy(val_small)

    wap_cols = [c for c in train_small.columns if c.startswith('WAP')]
    X_tr = train_small[wap_cols]
    X_va = val_small[wap_cols]

    y_b_tr = train_small['BUILDINGID']
    xy_tr = train_small[['X_M', 'Y_M']].values

    y_b_va = val_small['BUILDINGID']
    xy_va = val_small[['X_M', 'Y_M']].values

    # Test ensemble building classifier
    print("Testing ensemble building classifier...")
    building_ensemble = ensemble_building_pipeline()
    building_ensemble.fit(X_tr, y_b_tr)
    b_pred = building_ensemble.predict(X_va)
    b_acc = (b_pred == y_b_va).mean()
    print(f"Ensemble building accuracy: {b_acc:.3f}")

    # Test ensemble XY regressor
    print("Testing ensemble XY regressor...")
    xy_ensemble = ensemble_xy_pipeline()
    xy_ensemble.fit(X_tr, xy_tr)
    xy_pred = xy_ensemble.predict(X_va)

    from utils_geo import meter_error
    pos_errors = meter_error(xy_va, xy_pred)
    print(f"Ensemble position error: {pos_errors.mean():.2f} meters")
