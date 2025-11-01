# src/features.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

WAP_PREFIX = "WAP"


class RssiCleaner(BaseEstimator, TransformerMixin):
    """Clean and preprocess RSSI values."""

    def __init__(self, missing_value=-100, fill_value=-110, min_rssi=-110,
                 max_rssi=-30):
        self.missing_value = missing_value
        self.fill_value = fill_value
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi
        self.cols = None

    def fit(self, X, y=None):
        self.cols = [c for c in X.columns if c.startswith(WAP_PREFIX)]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = X[self.cols].replace(self.missing_value,
                                            self.fill_value)
        X[self.cols] = X[self.cols].clip(self.min_rssi, self.max_rssi)
        return X


class APSelector(BaseEstimator, TransformerMixin):
    """Select top-k access points by coverage and variance."""

    def __init__(self, coverage_min=0.02, top_k=200):
        self.coverage_min = coverage_min
        self.top_k = top_k
        self.keep_cols = None

    def fit(self, X, y=None):
        wap_cols = [c for c in X.columns if c.startswith(WAP_PREFIX)]
        coverage = (X[wap_cols] > -110).mean().sort_values(ascending=False)
        eligible = coverage[coverage >= self.coverage_min]
        self.keep_cols = list(eligible.index[:self.top_k])
        return self

    def transform(self, X):
        return X[self.keep_cols]


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering with signal patterns and statistics."""

    def __init__(self, rssi_threshold=-85, n_quantiles=5):
        self.rssi_threshold = rssi_threshold
        self.n_quantiles = n_quantiles
        self.wap_cols = None

    def fit(self, X, y=None):
        self.wap_cols = [c for c in X.columns if c.startswith(WAP_PREFIX)]
        return self

    def transform(self, X):
        X = X.copy()

        # Basic features
        rssi_values = X[self.wap_cols].replace(-110, np.nan)

        # Count features
        X['ap_count'] = (X[self.wap_cols] > self.rssi_threshold).sum(axis=1)
        X['ap_count_strong'] = (X[self.wap_cols] > -70).sum(axis=1)

        # RSSI statistics
        X['rssi_mean'] = rssi_values.mean(axis=1, skipna=True)
        X['rssi_median'] = rssi_values.median(axis=1, skipna=True)
        X['rssi_max'] = rssi_values.max(axis=1, skipna=True)
        X['rssi_min'] = rssi_values.min(axis=1, skipna=True)
        X['rssi_std'] = rssi_values.std(axis=1, skipna=True)
        X['rssi_range'] = X['rssi_max'] - X['rssi_min']

        # Signal strength distribution
        for i in range(self.n_quantiles):
            quantile = (i + 1) / self.n_quantiles
            X[f'rssi_q{int(quantile*100)}'] = rssi_values.quantile(
                quantile, axis=1, numeric_only=True)

        # Signal variability measures
        X['rssi_cv'] = X['rssi_std'] / X['rssi_mean'].abs()  # Coefficient of variation
        X['rssi_skew'] = rssi_values.skew(axis=1, skipna=True)
        X['rssi_kurtosis'] = rssi_values.kurtosis(axis=1, skipna=True)

        # AP density features
        strong_signals = (X[self.wap_cols] > -70).sum(axis=1)
        weak_signals = ((X[self.wap_cols] <= -70) &
                       (X[self.wap_cols] > -110)).sum(axis=1)
        X['signal_ratio'] = strong_signals / (weak_signals + 1)  # Avoid division by zero

        # Fill NaN values
        fill_cols = [c for c in X.columns if c.startswith(('rssi_', 'ap_count', 'signal_'))]
        X[fill_cols] = X[fill_cols].fillna(0)

        return X


class SignalPatternExtractor(BaseEstimator, TransformerMixin):
    """Extract signal patterns and temporal-like features."""

    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.wap_cols = None
        self.cluster_centers_ = None

    def fit(self, X, y=None):
        self.wap_cols = [c for c in X.columns if c.startswith(WAP_PREFIX)]

        # Simple clustering of RSSI patterns (could use K-means)
        # For now, just store the median pattern per "cluster"
        from sklearn.cluster import KMeans
        rssi_data = X[self.wap_cols].values
        kmeans = KMeans(n_clusters=self.n_clusters,
                       random_state=self.random_state, n_init=10)
        kmeans.fit(rssi_data)
        self.cluster_centers_ = kmeans.cluster_centers_

        return self

    def transform(self, X):
        X = X.copy()

        # Add cluster membership features (simplified)
        rssi_data = X[self.wap_cols].values

        # Compute distance to each cluster center
        for i in range(self.n_clusters):
            distances = np.linalg.norm(rssi_data - self.cluster_centers_[i], axis=1)
            X[f'pattern_dist_{i}'] = distances

        # Add the closest cluster
        min_distances = np.min([
            np.linalg.norm(rssi_data - center, axis=1)
            for center in self.cluster_centers_
        ], axis=0)
        X['pattern_closest_dist'] = min_distances

        return X


if __name__ == "__main__":
    # Quick test
    import os
    from data_io import load_uji

    train_path = os.path.join(os.path.dirname(__file__), "..", "data",
                              "TrainingData.csv")
    val_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "ValidationData.csv")
    train, _ = load_uji(train_path, val_path)

    # Test RSSI cleaner
    cleaner = RssiCleaner()
    train_clean = cleaner.fit_transform(train)
    print("RSSI cleaning:")
    missing_before = (train[cleaner.cols] == -100).sum().sum()
    print(f"Missing values before: {missing_before}")
    missing_after = (train_clean[cleaner.cols] == -100).sum().sum()
    print(f"Missing values after: {missing_after}")
    print(f"Min RSSI: {train_clean[cleaner.cols].min().min()}")
    print(f"Max RSSI: {train_clean[cleaner.cols].max().max()}")

    # Test AP selector
    selector = APSelector(coverage_min=0.01, top_k=50)
    train_selected = selector.fit_transform(train_clean)
    print(f"\nAP selection: {len(selector.keep_cols)} APs selected")
    print(f"Coverage of selected APs: {train_selected.mean().describe()}")

    # Test feature engineer
    engineer = FeatureEngineer()
    train_features = engineer.fit_transform(train_selected)
    print(f"\nFeature engineering: {train_features.shape[1]} features")
    print("New features:")
    features_desc = train_features[['ap_count', 'rssi_mean',
                                    'rssi_median', 'rssi_max']].describe()
    print(features_desc)
