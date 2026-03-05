from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

from src.navigation.assets import discover_navigation_assets
from src.navigation.localization_client import (
    LocalizationClient,
    LocalizationHTTPError,
    LocalizationPayloadError,
    LocalizationResult,
    LocalizationTimeoutError,
    RoomCandidatePrediction,
)
from src.navigation.pipeline import NavigationSession
from src.navigation.router_core import B10Router


ROOT_DIR = Path(__file__).resolve().parents[1]
NAV_DIR = ROOT_DIR / "indoor-navigation-part"


def _result(
    room_id: str,
    floor: int = 1,
    confidence_room: float | None = 0.9,
    candidates: List[tuple[str, float]] | None = None,
) -> LocalizationResult:
    candidate_objs = [
        RoomCandidatePrediction(room_id=rid, probability=prob)
        for rid, prob in (candidates or [])
    ]
    return LocalizationResult(
        building=10,
        floor=floor,
        room_id=room_id,
        confidence_room=confidence_room,
        confidence_floor=0.95,
        candidates=candidate_objs,
    )


class _StubLocalizationClient:
    def __init__(self, responses: List[Any]):
        self.responses = list(responses)
        self.index = 0

    def predict(self, scan: Dict[str, Any], top_k: int = 3):
        if self.index >= len(self.responses):
            raise RuntimeError("No more stub responses.")
        current = self.responses[self.index]
        self.index += 1
        if isinstance(current, Exception):
            raise current
        return current


class _MockResponse:
    def __init__(self, status_code: int, payload: Any, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self._text = text or str(payload)

    @property
    def text(self) -> str:
        return self._text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture(scope="module")
def router():
    return B10Router(NAV_DIR)


def test_asset_discovery_handles_variant_filenames():
    assets = discover_navigation_assets(NAV_DIR)
    assert assets.floor_data_paths[1].name == "floor_data.json"
    assert assets.floor_data_paths[2].name == "floor2_data (1).json"
    manifest_names = {path.name for path in assets.manifest_paths}
    assert "navigation_masks_manifest (2).json" in manifest_names
    assert "navigation_masks_manifest (1) (1).json" in manifest_names


def test_localization_client_success(monkeypatch):
    payload = {
        "building": 10,
        "floor": 1,
        "room_id": "10.1.17",
        "confidence_room": 0.92,
        "confidence_floor": 0.99,
        "candidates": [
            {"room_id": "10.1.17", "probability": 0.92},
            {"room_id": "10.1.18", "probability": 0.08},
        ],
    }

    def _fake_post(*args, **kwargs):
        return _MockResponse(200, payload)

    monkeypatch.setattr(requests, "post", _fake_post)
    client = LocalizationClient(base_url="http://localhost:8080")
    result = client.predict(scan={"rssi": {}, "imu": {}, "top_k": 3}, top_k=3)
    assert result.room_id == "10.1.17"
    assert result.floor == 1
    assert len(result.candidates) == 2


def test_localization_client_timeout(monkeypatch):
    def _fake_post(*args, **kwargs):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(requests, "post", _fake_post)
    client = LocalizationClient(base_url="http://localhost:8080")
    with pytest.raises(LocalizationTimeoutError):
        client.predict(scan={"rssi": {}, "imu": {}, "top_k": 3}, top_k=3)


def test_localization_client_non_200(monkeypatch):
    def _fake_post(*args, **kwargs):
        return _MockResponse(503, {"detail": "service unavailable"})

    monkeypatch.setattr(requests, "post", _fake_post)
    client = LocalizationClient(base_url="http://localhost:8080")
    with pytest.raises(LocalizationHTTPError):
        client.predict(scan={"rssi": {}, "imu": {}, "top_k": 3}, top_k=3)


def test_localization_client_malformed_payload(monkeypatch):
    def _fake_post(*args, **kwargs):
        return _MockResponse(200, {"floor": 1, "room_id": "10.1.17"})

    monkeypatch.setattr(requests, "post", _fake_post)
    client = LocalizationClient(base_url="http://localhost:8080")
    with pytest.raises(LocalizationPayloadError):
        client.predict(scan={"rssi": {}, "imu": {}, "top_k": 3}, top_k=3)


def test_alias_mapping_e1_to_elevator(router):
    stub = _StubLocalizationClient([_result("E1", floor=1, confidence_room=0.9)])
    session = NavigationSession(router=router, localization_client=stub)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.17")
    assert decision.status == "ok"
    assert decision.reason == "rerouted"
    assert decision.selected_room_id == "elevator1"


def test_top_k_fallback_selects_first_mapped_candidate(router):
    stub = _StubLocalizationClient(
        [
            _result(
                room_id="unknown",
                floor=1,
                confidence_room=0.95,
                candidates=[("unknown", 0.95), ("E1", 0.84), ("10.1.17", 0.80)],
            )
        ]
    )
    session = NavigationSession(router=router, localization_client=stub)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.17")
    assert decision.status == "ok"
    assert decision.selected_room_id == "elevator1"


def test_confidence_gate_returns_hold_on_low_confidence(router):
    stub = _StubLocalizationClient(
        [_result(room_id="10.1.17", floor=1, confidence_room=0.35, candidates=[("10.1.17", 0.35)])]
    )
    session = NavigationSession(router=router, localization_client=stub, confidence_threshold=0.60)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.18")
    assert decision.status == "hold"
    assert decision.reason == "low_confidence"


def test_two_hit_confirmation_before_switch(router):
    stub = _StubLocalizationClient(
        [
            _result("10.1.17", floor=1, confidence_room=0.9),
            _result("10.1.18", floor=1, confidence_room=0.9),
            _result("10.1.18", floor=1, confidence_room=0.9),
        ]
    )
    session = NavigationSession(router=router, localization_client=stub, transition_hits_required=2)

    first = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.31")
    second = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.31")
    third = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.31")

    assert first.status == "ok"
    assert second.status == "hold"
    assert second.reason == "awaiting_confirmation"
    assert third.status == "ok"
    assert third.selected_room_id == "10.1.18"


def test_pipeline_no_mapped_candidate(router):
    stub = _StubLocalizationClient(
        [
            _result(
                room_id="unknown",
                floor=1,
                confidence_room=0.9,
                candidates=[("unknown2", 0.85)],
            )
        ]
    )
    session = NavigationSession(router=router, localization_client=stub)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.17")
    assert decision.status == "hold"
    assert decision.reason == "no_mapped_candidate"


def test_pipeline_localization_error(router):
    stub = _StubLocalizationClient([LocalizationHTTPError("unavailable")])
    session = NavigationSession(router=router, localization_client=stub)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.1.17")
    assert decision.status == "error"
    assert decision.reason == "localization_error"


def test_pipeline_routing_error(router):
    stub = _StubLocalizationClient([_result("10.1.17", floor=1, confidence_room=0.9)])
    session = NavigationSession(router=router, localization_client=stub)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.9.99")
    assert decision.status == "error"
    assert decision.reason == "routing_error"


def test_multi_floor_route_success(router):
    stub = _StubLocalizationClient([_result("10.1.17", floor=1, confidence_room=0.9)])
    session = NavigationSession(router=router, localization_client=stub)
    decision = session.update_and_route(scan={"rssi": {}, "imu": {}}, goal_room_id="10.2.63")
    assert decision.status == "ok"
    assert decision.reason == "rerouted"
    assert decision.route_payload is not None
    assert "path_floor_start" in decision.route_payload
    assert "path_floor_goal" in decision.route_payload
