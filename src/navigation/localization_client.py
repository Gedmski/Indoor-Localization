from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, TypedDict

import requests


class LiveScan(TypedDict):
    rssi: Dict[str, float]
    imu: Dict[str, float]
    top_k: int


@dataclass(frozen=True)
class RoomCandidatePrediction:
    room_id: str
    probability: float


@dataclass(frozen=True)
class LocalizationResult:
    building: Optional[int]
    floor: int
    room_id: str
    confidence_room: Optional[float]
    confidence_floor: Optional[float]
    candidates: List[RoomCandidatePrediction]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [asdict(item) for item in self.candidates]
        return payload


class LocalizationClientError(RuntimeError):
    code = "localization_error"


class LocalizationTimeoutError(LocalizationClientError):
    code = "timeout"


class LocalizationHTTPError(LocalizationClientError):
    code = "http_error"


class LocalizationPayloadError(LocalizationClientError):
    code = "payload_error"


class LocalizationClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _predict_url(self) -> str:
        return f"{self.base_url}/predict"

    def predict(self, scan: Dict[str, Any], top_k: int = 3) -> LocalizationResult:
        payload = {
            "rssi": scan.get("rssi", {}),
            "imu": scan.get("imu", {}),
            "top_k": int(scan.get("top_k", top_k)),
        }

        try:
            response = requests.post(self._predict_url(), json=payload, timeout=self.timeout_s)
        except requests.Timeout as exc:
            raise LocalizationTimeoutError("Localization request timed out.") from exc
        except requests.RequestException as exc:
            raise LocalizationHTTPError(f"Localization request failed: {exc}") from exc

        if response.status_code != 200:
            detail = response.text
            try:
                body = response.json()
                if isinstance(body, dict) and body.get("detail"):
                    detail = str(body.get("detail"))
            except Exception:
                pass
            raise LocalizationHTTPError(
                f"Localization service returned status {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LocalizationPayloadError("Localization response is not valid JSON.") from exc

        return self._parse_response(data)

    def _parse_response(self, data: Any) -> LocalizationResult:
        if not isinstance(data, dict):
            raise LocalizationPayloadError("Localization response must be a JSON object.")

        required_keys = ["floor", "room_id", "candidates"]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise LocalizationPayloadError(
                f"Localization response missing required keys: {', '.join(missing)}"
            )

        try:
            floor = int(data["floor"])
        except (TypeError, ValueError) as exc:
            raise LocalizationPayloadError("Localization response 'floor' must be an integer.") from exc

        room_id = str(data["room_id"]).strip()
        if not room_id:
            raise LocalizationPayloadError("Localization response 'room_id' is empty.")

        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list):
            raise LocalizationPayloadError("Localization response 'candidates' must be a list.")

        candidates: List[RoomCandidatePrediction] = []
        for idx, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, dict):
                raise LocalizationPayloadError(f"Candidate at index {idx} must be an object.")
            if "room_id" not in candidate or "probability" not in candidate:
                raise LocalizationPayloadError(
                    f"Candidate at index {idx} must contain 'room_id' and 'probability'."
                )
            candidate_room = str(candidate["room_id"]).strip()
            if not candidate_room:
                raise LocalizationPayloadError(f"Candidate at index {idx} has empty 'room_id'.")
            try:
                probability = float(candidate["probability"])
            except (TypeError, ValueError) as exc:
                raise LocalizationPayloadError(
                    f"Candidate at index {idx} has invalid 'probability'."
                ) from exc
            candidates.append(
                RoomCandidatePrediction(room_id=candidate_room, probability=probability)
            )

        building = data.get("building")
        if building is not None:
            try:
                building = int(building)
            except (TypeError, ValueError) as exc:
                raise LocalizationPayloadError("Localization response 'building' must be an integer.") from exc

        confidence_room = data.get("confidence_room")
        if confidence_room is not None:
            try:
                confidence_room = float(confidence_room)
            except (TypeError, ValueError) as exc:
                raise LocalizationPayloadError(
                    "Localization response 'confidence_room' must be a number."
                ) from exc

        confidence_floor = data.get("confidence_floor")
        if confidence_floor is not None:
            try:
                confidence_floor = float(confidence_floor)
            except (TypeError, ValueError) as exc:
                raise LocalizationPayloadError(
                    "Localization response 'confidence_floor' must be a number."
                ) from exc

        return LocalizationResult(
            building=building,
            floor=floor,
            room_id=room_id,
            confidence_room=confidence_room,
            confidence_floor=confidence_floor,
            candidates=candidates,
        )
