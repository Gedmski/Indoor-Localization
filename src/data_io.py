from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


AP_PREFIX = "AP"
DEFAULT_BLDG10_PATH = Path("data") / "bldg10" / "final_data.csv"
BUILDING_ID = 10


def sorted_ap_columns(columns: Iterable[str]) -> List[str]:
    """Return AP columns in numeric order: AP1, AP2, ..., AP143."""
    ap_cols = [column for column in columns if column.startswith(AP_PREFIX)]

    def _ap_sort_key(column_name: str):
        suffix = column_name[len(AP_PREFIX):]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)

    return sorted(ap_cols, key=_ap_sort_key)


def load_bldg10(final_data_path: str | Path = DEFAULT_BLDG10_PATH) -> pd.DataFrame:
    """Load and standardize the Building 10 dataset."""
    data_path = Path(final_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Building 10 dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    required_columns = {"floor", "room_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in dataset: {missing_list}")

    ap_cols = sorted_ap_columns(df.columns)
    if not ap_cols:
        raise ValueError("No AP columns were found (expected AP1..AP143).")

    normalized = df.copy()
    normalized.loc[:, ap_cols] = (
        normalized.loc[:, ap_cols].apply(pd.to_numeric, errors="coerce").fillna(-100.0)
    )
    normalized = normalized.assign(
        FLOOR=normalized["floor"].astype(int),
        ROOMID=normalized["room_id"].astype(str),
        BUILDINGID=BUILDING_ID,
    )
    return normalized


def build_inference_frame(
    rssi_scan: Dict[str, float],
    ap_columns: List[str],
    missing_value: float = -100.0,
) -> pd.DataFrame:
    """Create a one-row feature frame from a raw scan."""
    row = {ap_name: float(rssi_scan.get(ap_name, missing_value)) for ap_name in ap_columns}
    return pd.DataFrame([row], columns=ap_columns)
