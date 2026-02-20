# src/features.py
from typing import Iterable, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

        transformed[self.ap_cols_] = transformed[self.ap_cols_].replace(
            self.missing_value, self.fill_value
        )
        transformed[self.ap_cols_] = transformed[self.ap_cols_].clip(
            lower=self.min_rssi, upper=self.max_rssi
        )
        return transformed


class APColumnSelector(BaseEstimator, TransformerMixin):
    """Select and order AP columns, filling missing APs for inference."""

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        missing_fill_value: float = -100.0,
    ):
        self.required_columns = required_columns
        self.missing_fill_value = missing_fill_value
        self.keep_cols_: Optional[List[str]] = None

    def fit(self, X, y=None):
        if self.required_columns:
            self.keep_cols_ = list(self.required_columns)
        else:
            self.keep_cols_ = _sorted_ap_columns(X.columns)
        return self

    def transform(self, X):
        transformed = X.copy()
        for ap_name in self.keep_cols_ or []:
            if ap_name not in transformed.columns:
                transformed[ap_name] = self.missing_fill_value
        return transformed[self.keep_cols_]
