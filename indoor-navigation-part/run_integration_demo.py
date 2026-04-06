import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure `src` is importable when this file is run as:
# `python indoor-navigation-part/run_integration_demo.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.navigation.localization_client import LocalizationClient
from src.navigation.pipeline import NavigationSession
from src.navigation.router_core import B10Router


IMU_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "mag_heading",
]


def build_scan_from_csv(csv_path: Path, row_index: int = 0) -> Dict[str, Dict[str, float]]:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise RuntimeError(f"Input CSV has no rows: {csv_path}")

    idx = max(0, min(row_index, len(frame) - 1))
    row = frame.iloc[idx]
    rssi = {
        str(column): float(row[column])
        for column in frame.columns
        if str(column).startswith("AP")
    }
    imu = {}
    for column in IMU_COLUMNS:
        imu[column] = float(row[column]) if column in frame.columns else 0.0

    return {"rssi": rssi, "imu": imu, "top_k": 3}


def main():
    parser = argparse.ArgumentParser(description="Run localization + navigation integration demo.")
    parser.add_argument(
        "--navigation-dir",
        default="indoor-navigation-part",
        help="Directory containing floor_data/manifests/walkable images.",
    )
    parser.add_argument(
        "--localization-url",
        default="http://localhost:8080",
        help="Base URL for localization API service.",
    )
    parser.add_argument(
        "--scan-csv",
        default="data/bldg10/final_data.csv",
        help="CSV used to source a demo AP+IMU scan row.",
    )
    parser.add_argument("--scan-row", type=int, default=0, help="CSV row index to use as input scan.")
    parser.add_argument("--goal-room-id", required=True, help="Destination room id (for example 10.2.63).")
    parser.add_argument("--top-k", type=int, default=3, help="Candidate count requested from localization.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Room confidence acceptance threshold.")
    parser.add_argument(
        "--transition-hits",
        type=int,
        default=2,
        help="Consecutive hits required before switching to a new location.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render route overlays if floor PDFs are available and visualization deps are installed.",
    )
    args = parser.parse_args()

    navigation_dir = Path(args.navigation_dir).resolve()
    scan_csv = Path(args.scan_csv).resolve()

    scan = build_scan_from_csv(scan_csv, row_index=args.scan_row)
    scan["top_k"] = int(args.top_k)

    router = B10Router(navigation_dir)
    client = LocalizationClient(base_url=args.localization_url)
    session = NavigationSession(
        router=router,
        localization_client=client,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
        transition_hits_required=args.transition_hits,
    )

    decision = session.update_and_route(
        scan=scan,
        goal_room_id=args.goal_room_id,
        accessible=True,
        visualize=args.visualize,
    )
    print(json.dumps(decision.as_dict(), indent=2))


if __name__ == "__main__":
    main()
