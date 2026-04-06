import asyncio

import pytest
from pydantic import ValidationError

from src import serve


class _StubRoomModel:
    def __init__(self):
        self.classes_ = ["10.1.17", "10.1.18", "10.1.31"]
        self.last_frame = None

    def predict(self, X):
        self.last_frame = X.copy()
        return ["10.1.17"]

    def predict_proba(self, X):
        self.last_frame = X.copy()
        return [[0.82, 0.12, 0.06]]


class _StubFloorModel:
    def __init__(self):
        self.classes_ = [1, 2]
        self.last_frame = None

    def predict(self, X):
        self.last_frame = X.copy()
        return [1]

    def predict_proba(self, X):
        self.last_frame = X.copy()
        return [[0.9, 0.1]]


@pytest.fixture
def stubbed_models(monkeypatch):
    room_model = _StubRoomModel()
    floor_model = _StubFloorModel()

    monkeypatch.setattr(serve, "room_model", room_model)
    monkeypatch.setattr(serve, "floor_model", floor_model)
    monkeypatch.setattr(serve, "model_metadata", {"building_id": 10})
    monkeypatch.setattr(serve, "ap_columns", ["AP1", "AP2", "AP3"])
    monkeypatch.setattr(serve, "imu_columns", ["accel_x", "gyro_z", "mag_heading"])

    return room_model, floor_model


def test_wifi_sample_requires_at_least_one_modality():
    with pytest.raises(ValidationError):
        serve.WifiSample()

    with pytest.raises(ValidationError):
        serve.WifiSample(rssi={}, imu={})


def test_predict_location_accepts_rssi_only(stubbed_models):
    room_model, floor_model = stubbed_models

    prediction = asyncio.run(
        serve.predict_location(serve.WifiSample(rssi={"AP1": -61.0, "AP3": -78.0}, top_k=2))
    )

    frame = room_model.last_frame
    assert frame is not None
    assert floor_model.last_frame is not None
    assert list(frame.columns) == ["AP1", "AP2", "AP3", "accel_x", "gyro_z", "mag_heading"]
    assert frame.iloc[0]["AP1"] == -61.0
    assert frame.iloc[0]["AP2"] == -100.0
    assert frame.iloc[0]["AP3"] == -78.0
    assert frame.iloc[0]["accel_x"] == 0.0
    assert frame.iloc[0]["gyro_z"] == 0.0
    assert frame.iloc[0]["mag_heading"] == 0.0
    assert prediction.room_id == "10.1.17"
    assert prediction.floor == 1
    assert len(prediction.candidates) == 2


def test_predict_location_accepts_imu_only(stubbed_models):
    room_model, _ = stubbed_models

    prediction = asyncio.run(
        serve.predict_location(
            serve.WifiSample(imu={"accel_x": 0.02, "mag_heading": 182.0}, top_k=3)
        )
    )

    frame = room_model.last_frame
    assert frame is not None
    assert frame.iloc[0]["AP1"] == -100.0
    assert frame.iloc[0]["AP2"] == -100.0
    assert frame.iloc[0]["AP3"] == -100.0
    assert frame.iloc[0]["accel_x"] == 0.02
    assert frame.iloc[0]["gyro_z"] == 0.0
    assert frame.iloc[0]["mag_heading"] == 182.0
    assert prediction.room_id == "10.1.17"
    assert len(prediction.candidates) == 3


def test_predict_location_accepts_combined_modalities_and_wap_aliases(stubbed_models):
    room_model, _ = stubbed_models

    prediction = asyncio.run(
        serve.predict_location(
            serve.WifiSample(
                rssi={"WAP001": -66.0, "AP2": -72.0},
                imu={"gyro_z": 0.4},
                top_k=1,
            )
        )
    )

    frame = room_model.last_frame
    assert frame is not None
    assert frame.iloc[0]["AP1"] == -66.0
    assert frame.iloc[0]["AP2"] == -72.0
    assert frame.iloc[0]["AP3"] == -100.0
    assert frame.iloc[0]["accel_x"] == 0.0
    assert frame.iloc[0]["gyro_z"] == 0.4
    assert frame.iloc[0]["mag_heading"] == 0.0
    assert prediction.room_id == "10.1.17"
    assert len(prediction.candidates) == 1
