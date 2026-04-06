# src/features.py
from typing import Iterable, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data_io import IMU_COLUMNS

AP_PREFIX = "AP"


def _sorted_ap_columns(columns: Iterable[str]) -> List[str]:
    ap_cols = [column for column in columns if column.startswith(AP_PREFIX)]

    def _sort_key(column_name: str):
        suffix = column_name[len(AP_PREFIX):]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)

    return sorted(ap_cols, key=_sort_key)


class RssiCleaner(BaseEstimator, TransformerMixin):
    """Normalize RSSI values for model use."""

    def __init__(
        self,
        missing_value: float = -100.0,
        fill_value: float = -110.0,
        min_rssi: float = -110.0,
        max_rssi: float = -30.0,
    ):
        self.missing_value = missing_value
        self.fill_value = fill_value
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi
        self.ap_cols_: Optional[List[str]] = None

    def fit(self, X, y=None):
        self.ap_cols_ = _sorted_ap_columns(X.columns)
        return self

    def transform(self, X):
        transformed = X.copy()
        if not self.ap_cols_:
            return transformed

        ap_values = transformed[self.ap_cols_].apply(pd.to_numeric, errors="coerce")
        ap_values = ap_values.fillna(self.missing_value)
        ap_values = ap_values.replace(self.missing_value, self.fill_value)
        ap_values = ap_values.clip(lower=self.min_rssi, upper=self.max_rssi)
        transformed[self.ap_cols_] = ap_values
        return transformed


class ImuCleaner(BaseEstimator, TransformerMixin):
    """Impute IMU columns using medians learned from training data."""

    def __init__(self, imu_columns: Optional[List[str]] = None, fill_value: float = 0.0):
        self.imu_columns = imu_columns
        self.fill_value = fill_value
        self.imu_cols_: Optional[List[str]] = None
        self.medians_: Optional[dict[str, float]] = None

    def fit(self, X, y=None):
        if self.imu_columns:
            self.imu_cols_ = list(self.imu_columns)
        else:
            self.imu_cols_ = [column for column in IMU_COLUMNS if column in X.columns]

        self.medians_ = {}
        if not self.imu_cols_:
            return self

        imu_values = X[self.imu_cols_].apply(pd.to_numeric, errors="coerce")
        medians = imu_values.median(numeric_only=True)

        for column in self.imu_cols_:
            median_value = medians.get(column, self.fill_value)
            if pd.isna(median_value):
                median_value = self.fill_value
            self.medians_[column] = float(median_value)
        return self

    def transform(self, X):
        transformed = X.copy()
        if not self.imu_cols_:
            return transformed

        for column in self.imu_cols_:
            if column not in transformed.columns:
                transformed[column] = self.fill_value

        imu_values = transformed[self.imu_cols_].apply(pd.to_numeric, errors="coerce")
        for column in self.imu_cols_:
            fallback = (
                self.medians_.get(column, self.fill_value) if self.medians_ else self.fill_value
            )
            imu_values[column] = imu_values[column].fillna(fallback)

        transformed[self.imu_cols_] = imu_values
        return transformed


class FeatureColumnSelector(BaseEstimator, TransformerMixin):
    """Select and order AP + IMU columns, filling missing values for inference."""

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        ap_missing_fill_value: float = -100.0,
        imu_missing_fill_value: float = 0.0,
    ):
        self.required_columns = required_columns
        self.ap_missing_fill_value = ap_missing_fill_value
        self.imu_missing_fill_value = imu_missing_fill_value
        self.keep_cols_: Optional[List[str]] = None

    def fit(self, X, y=None):
        if self.required_columns:
            self.keep_cols_ = list(self.required_columns)
        else:
            ap_cols = _sorted_ap_columns(X.columns)
            imu_cols = [column for column in IMU_COLUMNS if column in X.columns]
            self.keep_cols_ = ap_cols + imu_cols
        return self

    def transform(self, X):
        transformed = X.copy()
        keep_cols = self.keep_cols_ or []
        for column in keep_cols:
            if column not in transformed.columns:
                if column.startswith(AP_PREFIX):
                    transformed[column] = self.ap_missing_fill_value
                elif column in IMU_COLUMNS:
                    transformed[column] = self.imu_missing_fill_value
                else:
                    transformed[column] = 0.0

        ap_cols = [column for column in keep_cols if column.startswith(AP_PREFIX)]
        imu_cols = [column for column in keep_cols if column in IMU_COLUMNS]

        if ap_cols:
            transformed[ap_cols] = (
                transformed[ap_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(self.ap_missing_fill_value)
            )
        if imu_cols:
            transformed[imu_cols] = (
                transformed[imu_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(self.imu_missing_fill_value)
            )

        return transformed[keep_cols]
