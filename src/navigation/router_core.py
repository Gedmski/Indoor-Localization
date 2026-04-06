from __future__ import annotations

import heapq
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .assets import NavigationAssets, build_room_index, discover_navigation_assets, load_floor_data_records, load_json

try:
    import fitz  # type: ignore
    import matplotlib.pyplot as plt

    HAS_VIZ = True
except Exception:
    HAS_VIZ = False


METERS_PER_GRID_STEP = 0.8
WALK_SPEED_M_PER_S = 1.3
TRANSFER_TIME_S = {
    "elevator": 20.0,
    "stairway": 35.0,
}


def merge_manifests(manifest_paths: Iterable[Path]) -> List[dict]:
    all_entries: List[dict] = []
    for path in manifest_paths:
        payload = load_json(path)
        if isinstance(payload, list):
            all_entries.extend([item for item in payload if isinstance(item, dict)])
            continue

        if not isinstance(payload, dict):
            continue

        floors = payload.get("floors", payload)
        if isinstance(floors, list):
            all_entries.extend([item for item in floors if isinstance(item, dict)])
        elif isinstance(floors, dict):
            for value in floors.values():
                if isinstance(value, dict):
                    all_entries.append(value)
    return all_entries


def floor_from_manifest_item(item: dict) -> Optional[int]:
    if "floor" in item:
        try:
            return int(item["floor"])
        except Exception:
            pass
    floor_plan = str(item.get("floor_plan", "")).lower()
    if "level1" in floor_plan or "level 1" in floor_plan:
        return 1
    if "level2" in floor_plan or "level 2" in floor_plan:
        return 2
    return None


def pick_floor_meta(manifest_entries: List[dict], floor: int) -> Optional[dict]:
    for item in manifest_entries:
        parsed_floor = floor_from_manifest_item(item)
        if parsed_floor == floor:
            return item
    fallback_key = f"level{floor}"
    for item in manifest_entries:
        if fallback_key in str(item).lower():
            return item
    return None


def meta_origin_and_spacing(meta: dict, mask_w: int, mask_h: int) -> Tuple[float, float, float]:
    raw_w = float(meta.get("width", mask_w))
    raw_h = float(meta.get("height", mask_h))
    raw_spacing = float(meta.get("grid_spacing", 40))

    origin = meta.get("origin", {}) if isinstance(meta.get("origin"), dict) else {}
    raw_ox = float(origin.get("x", meta.get("origin_x", raw_w / 2)))
    raw_oy = float(origin.get("y", meta.get("origin_y", raw_h / 2)))

    sx = mask_w / raw_w if raw_w else 1.0
    sy = mask_h / raw_h if raw_h else 1.0
    scale = (sx + sy) / 2.0

    origin_x = raw_ox * sx
    origin_y = raw_oy * sy
    spacing = raw_spacing * scale
    return origin_x, origin_y, spacing


def load_walkable_boolean(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    frac_white = float((arr > 200).mean())
    if frac_white > 0.60:
        return arr < 128
    return arr > 127


def build_grid_from_walkable(walkable_mask: np.ndarray, grid_spacing_px: float) -> np.ndarray:
    height, width = walkable_mask.shape
    grid_spacing = max(1, int(round(grid_spacing_px)))
    grid_w = int(math.ceil(width / grid_spacing))
    grid_h = int(math.ceil(height / grid_spacing))

    grid = np.zeros((grid_h, grid_w), dtype=bool)
    for gy in range(grid_h):
        for gx in range(grid_w):
            px = min(width - 1, int(gx * grid_spacing + grid_spacing / 2))
            py = min(height - 1, int(gy * grid_spacing + grid_spacing / 2))
            grid[gy, gx] = bool(walkable_mask[py, px])
    return grid


def nearest_free_cell(grid: np.ndarray, gy: int, gx: int, r: int = 60) -> Optional[Tuple[int, int]]:
    if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1] and grid[gy, gx]:
        return gy, gx
    for radius in range(1, r + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx]:
                    return ny, nx
    return None


def neighbors_4(grid: np.ndarray, state: Tuple[int, int]):
    y, x = state
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx]:
            yield ny, nx


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if start is None or goal is None:
        return None
    if not grid[start] or not grid[goal]:
        return None

    open_heap = [(0, start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in neighbors_4(grid, current):
            tentative = g_score[current] + 1
            if neighbor not in g_score or tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                f_score = tentative + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))
                came_from[neighbor] = current

    return None


DIRS = {(0, 1): "Right", (0, -1): "Left", (1, 0): "Down", (-1, 0): "Up"}


def path_to_instructions(path: List[Tuple[int, int]]) -> List[str]:
    if not path or len(path) < 2:
        return ["You are already at the destination."]

    instructions: List[str] = []
    run_dir = None
    run_len = 0

    def flush():
        nonlocal run_dir, run_len
        if run_dir is not None and run_len > 0:
            instructions.append(f"Go {run_dir} for {run_len} steps")
        run_dir = None
        run_len = 0

    for i in range(1, len(path)):
        y0, x0 = path[i - 1]
        y1, x1 = path[i]
        dy, dx = (y1 - y0, x1 - x0)
        dname = DIRS.get((dy, dx), "Forward")
        if run_dir is None:
            run_dir = dname
            run_len = 1
        elif dname == run_dir:
            run_len += 1
        else:
            flush()
            run_dir = dname
            run_len = 1
    flush()
    return instructions


@dataclass
class Pose:
    x: float
    y: float
    floor: int


class B10Router:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        self.assets: NavigationAssets = discover_navigation_assets(self.base_dir)
        self.manifest_entries = merge_manifests(self.assets.manifest_paths)
        self.floor_data: List[dict] = load_floor_data_records(self.assets.floor_data_paths)
        self.room_by_id: Dict[str, dict] = build_room_index(self.floor_data)

        self._floor_loaded: Dict[int, bool] = {}
        self._walk_mask: Dict[int, np.ndarray] = {}
        self._grid: Dict[int, np.ndarray] = {}
        self._meta: Dict[int, dict] = {}
        self._origin_px: Dict[int, Tuple[float, float]] = {}
        self._spacing_px: Dict[int, float] = {}

    def has_room_id(self, room_id: str) -> bool:
        return str(room_id).strip() in self.room_by_id

    def ensure_floor_loaded(self, floor: int):
        if self._floor_loaded.get(floor):
            return
        if floor not in self.assets.walkable_paths:
            raise RuntimeError(f"No walkable mask found for floor {floor}.")

        walkable_path = self.assets.walkable_paths[floor]
        walkable_bool = load_walkable_boolean(walkable_path)
        mask_h, mask_w = walkable_bool.shape

        meta = pick_floor_meta(self.manifest_entries, floor)
        if meta is None:
            sample = None
            for room in self.floor_data:
                if (
                    str(room.get("floor")) == str(floor)
                    and room.get("center_x") is not None
                    and room.get("center_y") is not None
                ):
                    sample = room
                    break
            if sample is None:
                raise RuntimeError(
                    f"Missing manifest metadata for floor {floor} and no center_x/center_y fallback."
                )
            meta = {
                "width": mask_w,
                "height": mask_h,
                "grid_spacing": float(sample.get("grid_spacing", 40)),
                "origin": {"x": float(sample["center_x"]), "y": float(sample["center_y"])},
                "floor_plan": str(sample.get("floor_plan", f"floor{floor}")),
            }

        origin_x, origin_y, spacing = meta_origin_and_spacing(meta, mask_w, mask_h)
        grid = build_grid_from_walkable(walkable_bool, spacing)

        self._walk_mask[floor] = walkable_bool
        self._grid[floor] = grid
        self._meta[floor] = meta
        self._origin_px[floor] = (origin_x, origin_y)
        self._spacing_px[floor] = spacing
        self._floor_loaded[floor] = True

    def steps_to_distance_time(self, steps: int) -> Tuple[float, float]:
        distance_m = steps * METERS_PER_GRID_STEP
        time_s = distance_m / WALK_SPEED_M_PER_S if WALK_SPEED_M_PER_S > 0 else 0.0
        return distance_m, time_s

    def room_id_to_cell(self, room_id: str) -> Tuple[int, Tuple[int, int]]:
        rid = str(room_id).strip()
        if rid not in self.room_by_id:
            raise RuntimeError(f"Room id not found: {rid}")

        room = self.room_by_id[rid]
        floor = int(room["floor"])
        self.ensure_floor_loaded(floor)

        x_grid = float(room.get("x", 0.0))
        y_grid = float(room.get("y", 0.0))
        origin_x, origin_y = self._origin_px[floor]
        spacing = self._spacing_px[floor]

        px = origin_x + (x_grid * spacing)
        py = origin_y + (y_grid * spacing)
        gx = int(px // spacing)
        gy = int(py // spacing)

        snapped = nearest_free_cell(self._grid[floor], gy, gx, r=80)
        if snapped is None:
            raise RuntimeError(f"Room {rid} maps to blocked area on floor {floor}.")
        return floor, snapped

    def pose_to_cell(self, pose: Pose) -> Tuple[int, Tuple[int, int]]:
        floor = int(pose.floor)
        self.ensure_floor_loaded(floor)

        gy = int(round(pose.y))
        gx = int(round(pose.x))
        snapped = nearest_free_cell(self._grid[floor], gy, gx, r=100)
        if snapped is None:
            raise RuntimeError(f"Start pose maps to blocked area on floor {floor}.")
        return floor, snapped

    def connectors_on_floor(self, floor: int, accessible_only: bool) -> List[dict]:
        connectors = []
        for room in self.floor_data:
            if not isinstance(room, dict):
                continue
            if str(room.get("floor")) != str(floor):
                continue
            room_type = str(room.get("type", "")).lower()
            if room_type not in ["elevator", "stairway"]:
                continue
            if accessible_only:
                if room_type == "elevator" and bool(room.get("accessible", False)):
                    connectors.append(room)
            else:
                connectors.append(room)
        return connectors

    def connector_key(self, connector: dict) -> str:
        rid = str(connector.get("room_id", "")).lower()
        conn_type = str(connector.get("type", "")).lower()
        digits = "".join(re.findall(r"\d+", rid))
        return f"{conn_type}:{digits}" if digits else f"{conn_type}:{rid}"

    def visualize_path_on_floor(self, floor: int, path: List[Tuple[int, int]], title: str = ""):
        if not HAS_VIZ:
            print("[viz skipped] Install pymupdf + matplotlib to visualize.")
            return
        if floor not in self.assets.pdf_paths:
            print(f"[viz skipped] PDF not found for floor {floor}")
            return

        pdf_path = self.assets.pdf_paths[floor]
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()

        walk = self._walk_mask[floor]
        mask_h, mask_w = walk.shape
        img_h, img_w = img.shape[0], img.shape[1]
        sx = img_w / mask_w
        sy = img_h / mask_h

        spacing = self._spacing_px[floor]
        xs = []
        ys = []
        for gy, gx in path:
            px = (gx * spacing + spacing / 2) * sx
            py = (gy * spacing + spacing / 2) * sy
            xs.append(px)
            ys.append(py)

        plt.figure(figsize=(14, 8))
        plt.imshow(img)
        plt.title(title or f"Floor {floor} route")
        plt.axis("off")
        plt.plot(xs, ys, linewidth=4)
        plt.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=160)
        plt.show()

    def route_pose_to_room(
        self,
        start_pose: Pose,
        goal_room_id: str,
        accessible: bool = True,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        start_floor, start_cell = self.pose_to_cell(start_pose)
        goal_floor, goal_cell = self.room_id_to_cell(goal_room_id)

        if start_floor == goal_floor:
            t0 = time.perf_counter()
            path = astar(self._grid[start_floor], start_cell, goal_cell)
            t1 = time.perf_counter()
            if path is None:
                raise RuntimeError("No path found on same floor.")

            steps = len(path) - 1
            distance_m, walk_time_s = self.steps_to_distance_time(steps)
            instructions = path_to_instructions(path)

            if visualize:
                self.visualize_path_on_floor(start_floor, path, title=f"Floor {start_floor}: {goal_room_id}")

            return {
                "start_floor": start_floor,
                "goal_floor": goal_floor,
                "path": path,
                "steps": steps,
                "distance_m": distance_m,
                "time_s": walk_time_s,
                "instructions": instructions,
                "compute_time_s": t1 - t0,
            }

        start_connectors = self.connectors_on_floor(start_floor, accessible_only=accessible)
        goal_connectors = self.connectors_on_floor(goal_floor, accessible_only=accessible)
        if not start_connectors or not goal_connectors:
            raise RuntimeError("No suitable connectors found (elevators/stairs).")

        start_map: Dict[str, List[dict]] = {}
        goal_map: Dict[str, List[dict]] = {}
        for connector in start_connectors:
            start_map.setdefault(self.connector_key(connector), []).append(connector)
        for connector in goal_connectors:
            goal_map.setdefault(self.connector_key(connector), []).append(connector)

        candidate_pairs: List[Tuple[dict, dict]] = []
        for key in start_map.keys():
            if key in goal_map:
                for source in start_map[key]:
                    for target in goal_map[key]:
                        candidate_pairs.append((source, target))

        if not candidate_pairs:
            for source in start_connectors:
                for target in goal_connectors:
                    if str(source.get("type", "")).lower() == str(target.get("type", "")).lower():
                        candidate_pairs.append((source, target))

        best = None
        best_eta = float("inf")
        best_distance = float("inf")

        for source, target in candidate_pairs:
            _, source_cell = self.room_id_to_cell(str(source["room_id"]))
            _, target_cell = self.room_id_to_cell(str(target["room_id"]))

            path_start = astar(self._grid[start_floor], start_cell, source_cell)
            path_goal = astar(self._grid[goal_floor], target_cell, goal_cell)
            if path_start is None or path_goal is None:
                continue

            steps = (len(path_start) - 1) + (len(path_goal) - 1)
            distance_m, walk_time_s = self.steps_to_distance_time(steps)
            connector_type = str(source.get("type", "")).lower()
            transfer_time_s = TRANSFER_TIME_S.get(connector_type, 30.0)
            eta_time_s = walk_time_s + transfer_time_s

            if (eta_time_s < best_eta) or (
                abs(eta_time_s - best_eta) < 1e-6 and distance_m < best_distance
            ):
                best_eta = eta_time_s
                best_distance = distance_m
                best = {
                    "connector_start": source,
                    "connector_goal": target,
                    "path_floor_start": path_start,
                    "path_floor_goal": path_goal,
                    "steps": steps,
                    "distance_m": distance_m,
                    "walk_time_s": walk_time_s,
                    "transfer_time_s": transfer_time_s,
                    "eta_time_s": eta_time_s,
                }

        if best is None:
            raise RuntimeError("No multi-floor route found via available connectors.")

        if visualize:
            self.visualize_path_on_floor(
                start_floor,
                best["path_floor_start"],
                title=f"Floor {start_floor}: to {best['connector_start'].get('room_id')}",
            )
            self.visualize_path_on_floor(
                goal_floor,
                best["path_floor_goal"],
                title=f"Floor {goal_floor}: to {goal_room_id}",
            )

        return {
            "start_floor": start_floor,
            "goal_floor": goal_floor,
            "connector_start": best["connector_start"].get("room_id"),
            "connector_goal": best["connector_goal"].get("room_id"),
            "path_floor_start": best["path_floor_start"],
            "path_floor_goal": best["path_floor_goal"],
            "steps": best["steps"],
            "distance_m": best["distance_m"],
            "walk_time_s": best["walk_time_s"],
            "transfer_time_s": best["transfer_time_s"],
            "eta_time_s": best["eta_time_s"],
            "instructions_floor_start": path_to_instructions(best["path_floor_start"]),
            "instructions_floor_goal": path_to_instructions(best["path_floor_goal"]),
        }
