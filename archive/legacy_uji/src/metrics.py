# src/metrics.py
import numpy as np
from .utils_geo import meter_error


def strict_position_error(y_true_b, y_pred_b, y_true_f, y_pred_f,
                         xy_true, xy_pred, penalty=100.0):
    """Calculate position error with penalty for wrong building/floor.

    Args:
        y_true_b: True building IDs
        y_pred_b: Predicted building IDs
        y_true_f: True floor IDs
        y_pred_f: Predicted floor IDs
        xy_true: True (x,y) coordinates, shape (n_samples, 2)
        xy_pred: Predicted (x,y) coordinates, shape (n_samples, 2)
        penalty: Penalty in meters for wrong building/floor

    Returns:
        Tuple of (strict_mean, overall_mean) where:
        - strict_mean: Mean error only on correct building+floor samples
        - overall_mean: Mean error with penalty for wrong predictions
    """
    ok = (y_true_b == y_pred_b) & (y_true_f == y_pred_f)
    err = meter_error(xy_true, xy_pred)
    strict = err[ok]
    overall = np.where(ok, err, penalty)
    return strict.mean() if len(strict) > 0 else np.nan, overall.mean()


def evaluate_position_metrics(y_true_b, y_pred_b, y_true_f, y_pred_f,
                              xy_true, xy_pred):
    """Comprehensive position evaluation metrics.

    Args:
        Same as strict_position_error

    Returns:
        Dict with various position metrics
    """
    strict_err, overall_err = strict_position_error(
        y_true_b, y_pred_b, y_true_f, y_pred_f, xy_true, xy_pred)

    # Additional metrics
    ok = (y_true_b == y_pred_b) & (y_true_f == y_pred_f)
    err = meter_error(xy_true, xy_pred)

    return {
        'strict_2d_error_mean': strict_err,
        'overall_2d_error_mean': overall_err,
        'correct_building_floor_pct': ok.mean() * 100,
        'median_error_all': np.median(err),
        'p95_error_all': np.percentile(err, 95),
        'max_error_all': err.max(),
    }


def evaluate_classification_metrics(y_true, y_pred, label_name=""):
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_name: Name for the metric (e.g., "building", "floor")

    Returns:
        Dict with accuracy and other metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)

    return {
        f'{label_name}_accuracy': acc,
        f'{label_name}_precision': precision,
        f'{label_name}_recall': recall,
        f'{label_name}_f1': f1,
    }


def print_evaluation_report(building_true, building_pred,
                           floor_true, floor_pred,
                           xy_true, xy_pred):
    """Print a comprehensive evaluation report.

    Args:
        building_true: True building IDs
        building_pred: Predicted building IDs
        floor_true: True floor IDs
        floor_pred: Predicted floor IDs
        xy_true: True (x,y) coordinates
        xy_pred: Predicted (x,y) coordinates
    """
    print("=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)

    # Building metrics
    building_metrics = evaluate_classification_metrics(
        building_true, building_pred, "building")
    print("Building Classification:")
    for k, v in building_metrics.items():
        print(f"  {k}: {v:.3f}")

    # Floor metrics
    floor_metrics = evaluate_classification_metrics(
        floor_true, floor_pred, "floor")
    print("\nFloor Classification:")
    for k, v in floor_metrics.items():
        print(f"  {k}: {v:.3f}")

    # Position metrics
    pos_metrics = evaluate_position_metrics(
        building_true, building_pred, floor_true, floor_pred,
        xy_true, xy_pred)
    print("\nPosition Metrics:")
    for k, v in pos_metrics.items():
        if 'pct' in k:
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v:.2f}")

    print("=" * 50)


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Synthetic predictions (some correct, some wrong)
    building_true = np.random.randint(0, 3, n_samples)
    building_pred = building_true.copy()
    wrong_mask = np.random.random(n_samples) < 0.1
    building_pred[wrong_mask] = np.random.randint(0, 3, wrong_mask.sum())

    floor_true = np.random.randint(0, 5, n_samples)
    floor_pred = floor_true.copy()
    wrong_mask = np.random.random(n_samples) < 0.15
    floor_pred[wrong_mask] = np.random.randint(0, 5, wrong_mask.sum())

    # Position data
    xy_true = np.random.normal(0, 50, (n_samples, 2))  # True positions
    xy_pred = xy_true + np.random.normal(0, 5, (n_samples, 2))  # Add noise

    print_evaluation_report(building_true, building_pred,
                            floor_true, floor_pred,
                            xy_true, xy_pred)
