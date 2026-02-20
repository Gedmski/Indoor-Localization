# Building 10 Implementation Guide

## Scope

The active system supports one dataset and one building:

- input data: `data/bldg10/final_data.csv`
- AP features: `AP1..AP143`
- prediction outputs: `room_id`, `floor`, fixed `building=10`

No UJI multi-building training or XY regression is in the active runtime path.

## Runtime architecture

1. Data standardization
   - `src/data_io.py`
   - loads CSV
   - enforces required columns: `floor`, `room_id`, `AP*`
   - creates normalized labels: `FLOOR`, `ROOMID`, `BUILDINGID`
2. Feature processing
   - `src/features.py`
   - replaces missing RSSI (`-100`) with fill value (`-110`)
   - clips RSSI values to stable bounds
   - enforces AP column ordering for train/inference consistency
3. Model pipelines
   - `src/baselines.py`
   - room models: `knn`, `mlp`
   - floor model: `knn`
4. Training
   - `src/train.py`
   - trains room and floor models
   - writes models and metadata to `models/bldg10/`
5. Serving
   - `src/serve.py`
   - loads artifacts from `models/bldg10/`
   - exposes `POST /predict` and `GET /health`

## Commands

Train:

```bash
python run_bldg10_train.py --room-model knn
```

Evaluate:

```bash
python run_bldg10_evaluation.py
```

Serve API:

```bash
python run_bldg10_server.py --model-dir models/bldg10 --port 8080
```

## Files produced by training

`models/bldg10/metadata.json` stores:

- building id
- AP feature list and order
- selected room model type
- holdout metrics
- class labels

The API relies on this metadata to keep inference aligned with training.

## Navigation system integration

Use this project as a localization microservice:

1. Collect live Wi-Fi scan from app/infrastructure.
2. Convert scan to payload format: `{"rssi": {"AP1": ..., "AP2": ...}, "top_k": 3}`.
3. Call `POST /predict`.
4. Use response fields in navigation layer:
   - `building` -> building selector (always 10)
   - `floor` -> floor routing graph selection
   - `room_id` -> nearest node/anchor lookup in your map graph
   - `candidates` -> fallback for uncertain localization
5. Apply your existing smoothing/map-matching and rerouting logic.

## Expected active repository layout

- `data/bldg10/final_data.csv`
- `models/bldg10/` (generated artifacts)
- `src/data_io.py`
- `src/features.py`
- `src/baselines.py`
- `src/train.py`
- `src/evaluate_bldg10.py`
- `src/serve.py`
- `run_bldg10_train.py`
- `run_bldg10_evaluation.py`
- `run_bldg10_server.py`
- `README.md`
- `docs/BUILDING10_IMPLEMENTATION.md`

## Archived content

Legacy assets are preserved under:

- `archive/legacy_uji/`

This keeps historical work accessible while keeping the active surface clean.
