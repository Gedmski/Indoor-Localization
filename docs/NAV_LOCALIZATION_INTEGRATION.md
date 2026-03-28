# Building 10 Navigation + Localization Integration

## Overview

This repository now supports a module-based integration between:

- Localization API (`src/serve.py`, `POST /predict`)
- Navigation router (`src/navigation/router_core.py`)
- Stateful orchestration (`src/navigation/pipeline.py`)

The runtime behavior is:

1. Send RSSI, IMU, or combined scan to localization service.
2. Parse prediction (`room_id`, `floor`, confidence, candidates).
3. Apply confidence threshold (`0.60`) and top-k candidate fallback.
4. Map predicted room labels to navigation labels (`E1 -> elevator1`, `E2 -> elevator2`).
5. Require 2 consecutive detections before switching to a new location.
6. Route from selected room to destination `goal_room_id`.

## Startup Sequence

1. Train models (if needed):

```bash
python run_bldg10_train.py
```

2. Start localization API:

```bash
python run_bldg10_server.py --model-dir models/bldg10 --port 8080
```

3. Run integration demo:

```bash
python indoor-navigation-part/run_integration_demo.py --goal-room-id 10.2.63
```

## Python Module Usage

```python
from pathlib import Path
from src.navigation.localization_client import LocalizationClient
from src.navigation.pipeline import NavigationSession
from src.navigation.router_core import B10Router

router = B10Router(Path("indoor-navigation-part"))
client = LocalizationClient(base_url="http://localhost:8080", timeout_s=5.0)
session = NavigationSession(
    router=router,
    localization_client=client,
    confidence_threshold=0.60,
    top_k=3,
    transition_hits_required=2,
)

scan = {
    "rssi": {"AP1": -65.0, "AP2": -79.0},
    "top_k": 3,
}

decision = session.update_and_route(scan=scan, goal_room_id="10.2.63", accessible=True)
print(decision.as_dict())
```

You can supply `scan["rssi"]`, `scan["imu"]`, or both. The service rejects completely empty scans. Missing APs are filled with `-100`, and missing IMU values are filled with `0.0`.

## RouteDecision Contract

`NavigationSession.update_and_route(...)` returns a `RouteDecision` dataclass:

- `status`: `ok | hold | error`
- `reason`: `rerouted | awaiting_confirmation | low_confidence | no_mapped_candidate | localization_error | routing_error`
- `selected_room_id`: mapped room used for routing
- `selected_floor`: floor resolved from the selected room
- `route_payload`: router response payload when a route is available
- `rerouted`: true when a new route was computed in this call

## Confidence and Stability Rules

- Confidence threshold: `0.60`
- Candidate fallback: first mapped candidate from top-k with probability >= threshold
- Transition filter: 2-hit confirm before accepting a new room/floor
- Event-driven updates: routes are recomputed when location changes are confirmed

## Known Label Mapping

Localization room labels and navigation room IDs are mostly aligned. Current explicit aliases:

- `E1 -> elevator1`
- `E2 -> elevator2`

This mapping is hardcoded in `src/navigation/pipeline.py` and can be extended by passing `room_aliases` to `NavigationSession`.

## Asset Discovery Behavior

Navigation assets are discovered by patterns in `src/navigation/assets.py`:

- `navigation_masks_manifest*.json`
- `floor_data*.json` (floor 1)
- `floor2_data*.json` (floor 2)
- floor-specific walkable images matched by hints

The loader fails fast with explicit errors if mandatory floor data or walkable masks are missing.
