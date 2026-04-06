import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

try:
    from pydantic import model_validator
except ImportError:  # pragma: no cover - Pydantic v1 fallback
    model_validator = None
    from pydantic import root_validator

from .data_io import BUILDING_ID, build_inference_frame


class WifiSample(BaseModel):
    """Input model for flexible RSSI and IMU scans."""

    rssi: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional RSSI dictionary keyed by AP names (for example AP1..AP178). "
            "Omitted APs are filled with the default missing RSSI value."
        ),
    )
    imu: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional motion sensor dictionary. Recognized keys include accel_x, accel_y, "
            "accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, and mag_heading. "
            "Omitted IMU values are filled with 0.0."
        ),
    )
    top_k: int = Field(3, ge=1, le=10, description="Number of room candidates to return.")

    if model_validator is not None:

        @model_validator(mode="after")
        def validate_modalities(self):
            if not self.rssi and not self.imu:
                raise ValueError("At least one of 'rssi' or 'imu' must be provided.")
            return self

    else:

        @root_validator
        def validate_modalities(cls, values):
            if not values.get("rssi") and not values.get("imu"):
                raise ValueError("At least one of 'rssi' or 'imu' must be provided.")
            return values


class RoomCandidate(BaseModel):
    room_id: str
    probability: float


class LocationPrediction(BaseModel):
    building: int
    floor: int
    room_id: str
    confidence_room: Optional[float] = None
    confidence_floor: Optional[float] = None
    candidates: List[RoomCandidate] = Field(default_factory=list)


app = FastAPI(
    title="Building 10 Indoor Localization API",
    description="Room and floor inference for the Building 10 dataset.",
    version="2.0.0",
)

room_model = None
floor_model = None
model_metadata: Dict[str, object] = {}
ap_columns: List[str] = []
imu_columns: List[str] = []
model_dir = Path(__file__).parent.parent / "models" / "bldg10"


def _resolve_scan_value(scan: Dict[str, float], ap_name: str, default_value: float = -100.0):
    """Support AP and WAP aliases for easier integration."""
    if ap_name in scan:
        return scan[ap_name]
    suffix = ap_name[2:]
    if suffix.isdigit():
        wap_alias = f"WAP{int(suffix):03d}"
        if wap_alias in scan:
            return scan[wap_alias]
    return default_value


def load_models(model_path: str | Path = model_dir):
    """Load Building 10 models and metadata from disk."""
    global room_model, floor_model, model_metadata, ap_columns, imu_columns, model_dir

    model_dir = Path(model_path)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    room_model_name = model_metadata.get("room_model", "knn")

    room_model_path = model_dir / f"bldg10_room_{room_model_name}.joblib"
    floor_model_path = model_dir / "bldg10_floor_knn.joblib"
    if not room_model_path.exists() or not floor_model_path.exists():
        raise FileNotFoundError("Room or floor model file is missing.")

    room_model = load(room_model_path)
    floor_model = load(floor_model_path)
    ap_columns = [str(column) for column in model_metadata.get("ap_columns", [])]
    if not ap_columns:
        raise ValueError("No AP columns found in metadata.")
    imu_columns = [str(column) for column in model_metadata.get("imu_columns", [])]
    if not imu_columns:
        raise ValueError(
            "No IMU columns found in metadata. Retrain models with feature_schema_version=2."
        )


@app.on_event("startup")
async def startup_event():
    load_models(model_dir)


@app.post("/predict", response_model=LocationPrediction)
async def predict_location(sample: WifiSample):
    if room_model is None or floor_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    try:
        normalized_scan = {
            ap_name: float(_resolve_scan_value(sample.rssi, ap_name)) for ap_name in ap_columns
        }
        normalized_imu = {imu_name: float(sample.imu.get(imu_name, 0.0)) for imu_name in imu_columns}
        X = build_inference_frame(normalized_scan, normalized_imu, ap_columns, imu_columns)

        predicted_room = str(room_model.predict(X)[0])
        predicted_floor = int(floor_model.predict(X)[0])

        room_confidence = None
        floor_confidence = None
        candidates: List[RoomCandidate] = []

        if hasattr(room_model, "predict_proba"):
            room_probs = room_model.predict_proba(X)[0]
            room_labels = [str(label) for label in room_model.classes_]
            ranked = sorted(
                zip(room_labels, room_probs), key=lambda item: item[1], reverse=True
            )
            room_confidence = float(ranked[0][1])
            candidates = [
                RoomCandidate(room_id=room_id, probability=float(probability))
                for room_id, probability in ranked[: sample.top_k]
            ]

        if hasattr(floor_model, "predict_proba"):
            floor_probs = floor_model.predict_proba(X)[0]
            floor_confidence = float(max(floor_probs))

        return LocationPrediction(
            building=int(model_metadata.get("building_id", BUILDING_ID)),
            floor=predicted_floor,
            room_id=predicted_room,
            confidence_room=room_confidence,
            confidence_floor=floor_confidence,
            candidates=candidates,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}") from exc


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": room_model is not None and floor_model is not None,
        "building_id": model_metadata.get("building_id", BUILDING_ID),
        "room_model": model_metadata.get("room_model"),
        "ap_features": len(ap_columns),
        "imu_features": len(imu_columns),
        "total_features": len(ap_columns) + len(imu_columns),
    }


@app.get("/")
async def root():
    return {
        "service": "Building 10 Indoor Localization API",
        "version": "2.0.0",
        "supported_input_modes": ["rssi", "imu", "rssi+imu"],
        "predict_contract": {
            "request": {
                "examples": {
                    "rssi_only": {
                        "rssi": {"AP1": -70, "AP2": -100},
                        "top_k": 3,
                    },
                    "imu_only": {
                        "imu": {"accel_x": 0.01, "gyro_z": 0.0, "mag_heading": 170.0},
                        "top_k": 3,
                    },
                    "combined": {
                        "rssi": {"AP1": -70, "AP2": -100},
                        "imu": {
                            "accel_x": 0.01,
                            "accel_y": 0.02,
                            "accel_z": 1.01,
                            "gyro_x": 0.1,
                            "gyro_y": -0.1,
                            "gyro_z": 0.0,
                            "mag_x": -40.0,
                            "mag_y": 5.0,
                            "mag_z": -6.0,
                            "mag_heading": 170.0,
                        },
                        "top_k": 3,
                    },
                }
            },
            "response_keys": [
                "building",
                "floor",
                "room_id",
                "confidence_room",
                "confidence_floor",
                "candidates",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Building 10 API server.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/bldg10",
        help="Directory containing Building 10 model files and metadata.json",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    load_models(args.model_dir)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
