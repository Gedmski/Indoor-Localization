from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .localization_client import (
    LocalizationClient,
    LocalizationClientError,
    LocalizationResult,
    RoomCandidatePrediction,
)
from .router_core import B10Router, Pose


DEFAULT_ROOM_ALIASES = {
    "E1": "elevator1",
    "E2": "elevator2",
}


@dataclass
class RouteDecision:
    status: str
    reason: str
    selected_room_id: Optional[str] = None
    selected_floor: Optional[int] = None
    route_payload: Optional[Dict[str, Any]] = None
    rerouted: bool = False
    localization: Optional[Dict[str, Any]] = None
    details: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Candidate:
    room_id: str
    probability: Optional[float]


class NavigationSession:
    def __init__(
        self,
        router: B10Router,
        localization_client: LocalizationClient,
        confidence_threshold: float = 0.60,
        top_k: int = 3,
        transition_hits_required: int = 2,
        room_aliases: Optional[Dict[str, str]] = None,
    ):
        if transition_hits_required < 1:
            raise ValueError("transition_hits_required must be at least 1")
        self.router = router
        self.localization_client = localization_client
        self.confidence_threshold = float(confidence_threshold)
        self.top_k = int(top_k)
        self.transition_hits_required = int(transition_hits_required)
        self.room_aliases = dict(DEFAULT_ROOM_ALIASES)
        if room_aliases:
            self.room_aliases.update(room_aliases)

        self.active_room_id: Optional[str] = None
        self.active_floor: Optional[int] = None
        self.pending_room_id: Optional[str] = None
        self.pending_floor: Optional[int] = None
        self.pending_hits: int = 0
        self.last_route: Optional[Dict[str, Any]] = None
        self.last_goal_room_id: Optional[str] = None

    def _map_room_id(self, room_id: str) -> str:
        clean = str(room_id).strip()
        if not clean:
            return clean
        if clean in self.room_aliases:
            return self.room_aliases[clean]
        upper = clean.upper()
        if upper in self.room_aliases:
            return self.room_aliases[upper]
        return clean

    def _candidate_rows(self, result: LocalizationResult) -> List[_Candidate]:
        rows: List[_Candidate] = []
        rows.append(_Candidate(room_id=result.room_id, probability=result.confidence_room))
        for item in result.candidates:
            rows.append(_Candidate(room_id=item.room_id, probability=item.probability))
        deduped: List[_Candidate] = []
        seen = set()
        for row in rows:
            mapped = self._map_room_id(row.room_id)
            key = (mapped, row.probability)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(_Candidate(room_id=mapped, probability=row.probability))
        return deduped

    def _select_candidate(self, result: LocalizationResult) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        candidates = self._candidate_rows(result)
        mapped = [item for item in candidates if item.room_id and self.router.has_room_id(item.room_id)]
        if not mapped:
            return None, "no_mapped_candidate", None

        for item in mapped:
            prob = item.probability
            if prob is not None and prob >= self.confidence_threshold:
                return item.room_id, None, prob
        return None, "low_confidence", None

    def _route_from_room(
        self,
        selected_room_id: str,
        goal_room_id: str,
        accessible: bool,
        visualize: bool,
    ) -> RouteDecision:
        try:
            floor, cell = self.router.room_id_to_cell(selected_room_id)
            pose = Pose(x=float(cell[1]), y=float(cell[0]), floor=floor)
            route_payload = self.router.route_pose_to_room(
                start_pose=pose,
                goal_room_id=goal_room_id,
                accessible=accessible,
                visualize=visualize,
            )
        except Exception as exc:
            return RouteDecision(
                status="error",
                reason="routing_error",
                selected_room_id=selected_room_id,
                details=str(exc),
            )

        self.active_room_id = selected_room_id
        self.active_floor = floor
        self.last_route = route_payload
        self.last_goal_room_id = goal_room_id
        self.pending_room_id = None
        self.pending_floor = None
        self.pending_hits = 0

        return RouteDecision(
            status="ok",
            reason="rerouted",
            selected_room_id=selected_room_id,
            selected_floor=floor,
            route_payload=route_payload,
            rerouted=True,
        )

    def update_and_route(
        self,
        scan: Dict[str, Any],
        goal_room_id: str,
        accessible: bool = True,
        visualize: bool = False,
    ) -> RouteDecision:
        try:
            localization = self.localization_client.predict(scan=scan, top_k=self.top_k)
        except LocalizationClientError as exc:
            return RouteDecision(
                status="error",
                reason="localization_error",
                details=f"{exc.__class__.__name__}: {exc}",
            )

        selected_room_id, reject_reason, selected_prob = self._select_candidate(localization)
        if selected_room_id is None:
            return RouteDecision(
                status="hold",
                reason=str(reject_reason),
                localization={
                    "room_id": localization.room_id,
                    "floor": localization.floor,
                    "confidence_room": localization.confidence_room,
                },
            )

        localization_payload = {
            "room_id": localization.room_id,
            "floor": localization.floor,
            "confidence_room": localization.confidence_room,
            "selected_probability": selected_prob,
            "candidates": [asdict(item) for item in localization.candidates],
        }

        try:
            floor, _ = self.router.room_id_to_cell(selected_room_id)
        except Exception as exc:
            return RouteDecision(
                status="error",
                reason="routing_error",
                selected_room_id=selected_room_id,
                localization=localization_payload,
                details=str(exc),
            )

        if self.active_room_id is None:
            decision = self._route_from_room(
                selected_room_id=selected_room_id,
                goal_room_id=goal_room_id,
                accessible=accessible,
                visualize=visualize,
            )
            decision.localization = localization_payload
            return decision

        if selected_room_id == self.active_room_id:
            self.pending_room_id = None
            self.pending_floor = None
            self.pending_hits = 0

            if self.last_route is not None and self.last_goal_room_id == goal_room_id:
                return RouteDecision(
                    status="ok",
                    reason="rerouted",
                    selected_room_id=self.active_room_id,
                    selected_floor=self.active_floor,
                    route_payload=self.last_route,
                    rerouted=False,
                    localization=localization_payload,
                )

            decision = self._route_from_room(
                selected_room_id=selected_room_id,
                goal_room_id=goal_room_id,
                accessible=accessible,
                visualize=visualize,
            )
            decision.localization = localization_payload
            return decision

        if self.pending_room_id == selected_room_id and self.pending_floor == floor:
            self.pending_hits += 1
        else:
            self.pending_room_id = selected_room_id
            self.pending_floor = floor
            self.pending_hits = 1

        if self.pending_hits < self.transition_hits_required:
            return RouteDecision(
                status="hold",
                reason="awaiting_confirmation",
                selected_room_id=selected_room_id,
                selected_floor=floor,
                localization=localization_payload,
                details=(
                    f"Pending location change {self.pending_hits}/{self.transition_hits_required}: "
                    f"{self.active_room_id} -> {selected_room_id}"
                ),
            )

        decision = self._route_from_room(
            selected_room_id=selected_room_id,
            goal_room_id=goal_room_id,
            accessible=accessible,
            visualize=visualize,
        )
        decision.localization = localization_payload
        return decision
