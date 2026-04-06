import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
PDF_EXTENSIONS = (".pdf",)

WALKABLE_HINTS = {
    1: ["inv_walkable_route_floor_1.jpeg", "walkable routes floor1.jpeg", "floor_1", "level1"],
    2: ["inv_walkable_route_floor_2.png", "floor_2", "level2"],
}

PDF_HINTS = {
    1: ["B10  Level 1 Floor_Plan.pdf", "level 1", "level1"],
    2: ["B10  Level 2 Floor_Plan.pdf", "level 2", "level2"],
}


@dataclass(frozen=True)
class NavigationAssets:
    base_dir: Path
    manifest_paths: List[Path]
    floor_data_paths: Dict[int, Path]
    walkable_paths: Dict[int, Path]
    pdf_paths: Dict[int, Path]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sorted_matches(base_dir: Path, pattern: str) -> List[Path]:
    return sorted(base_dir.glob(pattern), key=lambda item: item.name.lower())


def _ranked_path_choice(candidates: Iterable[Path], preferred_names: List[str], hints: List[str]) -> Optional[Path]:
    preferred_lookup = {name.lower() for name in preferred_names}
    hints_lookup = [hint.lower() for hint in hints]

    best_path = None
    best_score = -1
    for path in candidates:
        lower = path.name.lower()
        score = 0
        if lower in preferred_lookup:
            score += 1000
        for hint in hints_lookup:
            if hint and hint in lower:
                score += 10
        if score > best_score:
            best_score = score
            best_path = path
        elif score == best_score and best_path is not None and lower < best_path.name.lower():
            best_path = path
    return best_path


def _pick_floor_data_path(base_dir: Path, floor: int) -> Optional[Path]:
    if floor == 1:
        strict = _sorted_matches(base_dir, "floor_data*.json")
        strict = [path for path in strict if "floor2" not in path.name.lower()]
        if strict:
            return strict[0]
        fallback = [path for path in _sorted_matches(base_dir, "*floor*.json") if "floor2" not in path.name.lower()]
        return fallback[0] if fallback else None

    strict = _sorted_matches(base_dir, "floor2_data*.json")
    if strict:
        return strict[0]
    fallback = [path for path in _sorted_matches(base_dir, "*floor*.json") if "floor2" in path.name.lower()]
    return fallback[0] if fallback else None


def _pick_image_by_hints(base_dir: Path, floor: int) -> Optional[Path]:
    candidates: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(_sorted_matches(base_dir, f"*{ext}"))
    return _ranked_path_choice(
        candidates,
        preferred_names=[WALKABLE_HINTS[floor][0]],
        hints=WALKABLE_HINTS[floor],
    )


def _pick_pdf_by_hints(base_dir: Path, floor: int) -> Optional[Path]:
    candidates: List[Path] = []
    for ext in PDF_EXTENSIONS:
        candidates.extend(_sorted_matches(base_dir, f"*{ext}"))
    return _ranked_path_choice(
        candidates,
        preferred_names=[PDF_HINTS[floor][0]],
        hints=PDF_HINTS[floor],
    )


def discover_navigation_assets(base_dir: str | Path) -> NavigationAssets:
    resolved_base = Path(base_dir).resolve()
    if not resolved_base.exists():
        raise FileNotFoundError(f"Navigation base directory does not exist: {resolved_base}")

    manifest_paths = _sorted_matches(resolved_base, "navigation_masks_manifest*.json")
    floor_data_paths: Dict[int, Path] = {}
    walkable_paths: Dict[int, Path] = {}
    pdf_paths: Dict[int, Path] = {}

    missing: List[str] = []
    for floor in (1, 2):
        floor_path = _pick_floor_data_path(resolved_base, floor)
        if floor_path is None:
            missing.append(f"missing floor{floor} data json (expected floor_data*.json / floor2_data*.json)")
        else:
            floor_data_paths[floor] = floor_path

        walkable = _pick_image_by_hints(resolved_base, floor)
        if walkable is None:
            missing.append(f"missing walkable image for floor {floor}")
        else:
            walkable_paths[floor] = walkable

        pdf = _pick_pdf_by_hints(resolved_base, floor)
        if pdf is not None:
            pdf_paths[floor] = pdf

    if missing:
        raise FileNotFoundError(
            "Navigation asset discovery failed:\n"
            + "\n".join(f"- {entry}" for entry in missing)
            + f"\nBase directory: {resolved_base}"
        )

    return NavigationAssets(
        base_dir=resolved_base,
        manifest_paths=manifest_paths,
        floor_data_paths=floor_data_paths,
        walkable_paths=walkable_paths,
        pdf_paths=pdf_paths,
    )


def load_floor_data_records(floor_data_paths: Dict[int, Path]) -> List[dict]:
    records: List[dict] = []
    for floor in sorted(floor_data_paths.keys()):
        path = floor_data_paths[floor]
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Floor data file must contain a list: {path}")
        for row in payload:
            if isinstance(row, dict):
                normalized = dict(row)
                if normalized.get("floor") is not None:
                    try:
                        normalized["floor"] = int(str(normalized["floor"]).strip())
                    except ValueError:
                        pass
                if normalized.get("room_id") is not None:
                    normalized["room_id"] = str(normalized["room_id"]).strip()
                records.append(normalized)
    return records


def build_room_index(records: Iterable[dict]) -> Dict[str, dict]:
    room_by_id: Dict[str, dict] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        room_id = row.get("room_id")
        if room_id is None:
            continue
        rid = str(room_id).strip()
        if not rid:
            continue
        room_by_id[rid] = dict(row)
    return room_by_id


def parse_floor_from_room_id(room_id: str) -> Optional[int]:
    match = re.match(r"^\d+\.(\d+)\.", room_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
