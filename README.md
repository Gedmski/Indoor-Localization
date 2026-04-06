# Building 10 Indoor Localization

This repository is now scoped to **Building 10 only** using:

- dataset: `data/bldg10/final_data.csv`
- features: `AP1..AP178` + IMU (`accel_*`, `gyro_*`, `mag_*`, `mag_heading`)
- targets: `ROOMID` (room), `FLOOR` (floor)
- building id: fixed to `10`
- inference inputs: RSSI only, motion sensors only, or both

Legacy multi-building/UJI files were moved to `archive/legacy_uji/`.

## Active workflow

1. Train Building 10 models
   - `python run_bldg10_train.py`
2. Evaluate models and generate report/plots
   - `python run_bldg10_evaluation.py`
3. Serve prediction API
   - `python run_bldg10_server.py`

## API contract

`POST /predict`

Request:

```json
{
  "rssi": {
    "AP1": -65,
    "AP2": -78,
    "AP3": -100
  },
  "top_k": 3
}
```

You can also send:

- RSSI only: include `rssi`
- Motion only: include `imu`
- Combined: include both `rssi` and `imu`

At least one modality is required. Missing APs are filled with `-100`, and missing IMU values are filled with `0.0`.

Response:

```json
{
  "building": 10,
  "floor": 2,
  "room_id": "2F-Hall-03",
  "confidence_room": 0.91,
  "confidence_floor": 0.99,
  "candidates": [
    { "room_id": "2F-Hall-03", "probability": 0.91 },
    { "room_id": "2F-Hall-02", "probability": 0.06 },
    { "room_id": "2F-Hall-04", "probability": 0.03 }
  ]
}
```

## Active files

- `src/data_io.py`: dataset loading, AP ordering, IMU schema, inference frame builder
- `src/features.py`: RSSI + IMU cleaning and feature column selection
- `src/baselines.py`: room/floor model pipelines
- `src/train.py`: model training and model artifact export
- `src/evaluate_bldg10.py`: evaluation report and confusion plots
- `src/serve.py`: FastAPI prediction service
- `run_bldg10_train.py`: training entrypoint
- `run_bldg10_evaluation.py`: evaluation entrypoint
- `run_bldg10_server.py`: API entrypoint

## Model artifacts

Training writes to `models/bldg10/`:

- `bldg10_room_<model>.joblib`
- `bldg10_floor_knn.joblib`
- `metadata.json`

The API loads `metadata.json` to discover AP/IMU feature order and model filenames.

## Docs

Implementation details and integration guidance:

- `docs/BUILDING10_IMPLEMENTATION.md`
- `docs/NAV_LOCALIZATION_INTEGRATION.md`
- `archive/legacy_uji/README.md`
