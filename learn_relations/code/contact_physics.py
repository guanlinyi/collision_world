from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from teacher_inputs import choose_actor_ids

FEATURE_NAMES = ('actor_a_present', 'actor_b_present', 'pair_present', 'center_distance_xy_m', 'aabb_gap_3d_m', 'oriented_box_gap_3d_m', 'closing_speed_mps', 'relative_speed_mps', 'actor_a_speed_mps', 'actor_b_speed_mps', 'actor_a_acceleration_mps2', 'actor_b_acceleration_mps2', 'actor_a_angular_speed_dps', 'actor_b_angular_speed_dps', 'max_abs_roll_deg', 'max_abs_pitch_deg', 'sin_yaw_difference', 'cos_yaw_difference', 'actor_a_abs_yaw_change_deg', 'actor_b_abs_yaw_change_deg', 'max_abs_roll_change_deg', 'sum_abs_speed_change_mps', 'max_abs_vertical_speed_mps')

def load_static_geometry(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mask2world-carla-static-geometry-0.1":
        raise ValueError(f"unsupported static geometry schema: {path}")
    return list(payload.get("objects", []))


def vec(mapping: dict[str, Any] | None) -> np.ndarray:
    mapping = mapping or {}
    return np.asarray(
        [float(mapping.get(axis, 0.0)) for axis in ("x", "y", "z")],
        dtype=np.float32,
    )


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def angle_delta(current: float, previous: float) -> float:
    return float((current - previous + 180.0) % 360.0 - 180.0)


def object_state(obj: dict[str, Any] | None) -> dict[str, Any]:
    if obj is None:
        return {
            "present": 0.0,
            "location": np.zeros(3, dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
            "acceleration": np.zeros(3, dtype=np.float32),
            "angular": np.zeros(3, dtype=np.float32),
            "rotation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "aabb_min": np.zeros(3, dtype=np.float32),
            "aabb_max": np.zeros(3, dtype=np.float32),
            "corners": np.empty((0, 3), dtype=np.float32),
        }
    transform = obj.get("actor_transform", {})
    corners = np.asarray(obj.get("bbox_3d_world", []), dtype=np.float32)
    location = vec(transform.get("location"))
    if corners.shape == (8, 3):
        aabb_min, aabb_max = corners.min(axis=0), corners.max(axis=0)
    else:
        aabb_min = aabb_max = location
    rotation = transform.get("rotation", {})
    return {
        "present": 1.0,
        "location": location,
        "velocity": vec(obj.get("velocity_world_mps")),
        "acceleration": vec(obj.get("acceleration_world_mps2")),
        "angular": vec(obj.get("angular_velocity_world_dps")),
        "rotation": {
            key: float(rotation.get(key, 0.0)) for key in ("yaw", "pitch", "roll")
        },
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "corners": corners,
    }


def aabb_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    delta = np.maximum(0.0, np.maximum(a["aabb_min"] - b["aabb_max"], b["aabb_min"] - a["aabb_max"]))
    return norm(delta)


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points[:, :2]})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float32)

    def cross(origin, left, right):
        return ((left[0] - origin[0]) * (right[1] - origin[1])
                - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    amount = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + amount * delta)))


def _polygons_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    for polygon in (left, right):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float32)
            length = float(np.linalg.norm(axis))
            if length <= 1e-12:
                continue
            axis /= length
            left_projection = left @ axis
            right_projection = right @ axis
            if left_projection.max() < right_projection.min() or right_projection.max() < left_projection.min():
                return False
    return True


def oriented_box_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Historical XY-hull/z-gap target retained for reproducibility, not a general exact 3D box or collider distance."""
    left_corners = np.asarray(a.get("corners", []), dtype=np.float32)
    right_corners = np.asarray(b.get("corners", []), dtype=np.float32)
    if left_corners.shape != (8, 3) or right_corners.shape != (8, 3):
        return aabb_gap(a, b)
    left, right = _convex_hull_xy(left_corners), _convex_hull_xy(right_corners)
    if len(left) < 3 or len(right) < 3:
        return aabb_gap(a, b)
    if _polygons_overlap(left, right):
        xy_gap = 0.0
    else:
        distances = []
        for points, edges in ((left, right), (right, left)):
            for point in points:
                distances.extend(
                    _point_segment_distance(point, edges[index], edges[(index + 1) % len(edges)])
                    for index in range(len(edges))
                )
        xy_gap = min(distances)
    z_gap = max(
        0.0,
        float(left_corners[:, 2].min() - right_corners[:, 2].max()),
        float(right_corners[:, 2].min() - left_corners[:, 2].max()),
    )
    return float(math.hypot(xy_gap, z_gap))


def frame_features(
    current: dict[int, dict[str, Any]],
    previous: dict[int, dict[str, Any]],
    actor_ids: list[int],
) -> np.ndarray:
    objects = [object_state(current.get(actor_id)) for actor_id in actor_ids[:2]]
    prev_objects = [object_state(previous.get(actor_id)) for actor_id in actor_ids[:2]]
    while len(objects) < 2:
        objects.append(object_state(None))
        prev_objects.append(object_state(None))
    a, b = objects
    pa, pb = prev_objects
    pair_present = a["present"] * b["present"]
    relative_position = b["location"] - a["location"]
    relative_velocity = b["velocity"] - a["velocity"]
    distance_xy = norm(relative_position[:2]) if pair_present else 50.0
    closing = (
        -float(np.dot(relative_position[:2], relative_velocity[:2])) / max(distance_xy, 1e-6)
        if pair_present
        else 0.0
    )
    yaw_delta = math.radians(angle_delta(b["rotation"]["yaw"], a["rotation"]["yaw"]))
    speed_a, speed_b = norm(a["velocity"]), norm(b["velocity"])
    previous_speed_a, previous_speed_b = norm(pa["velocity"]), norm(pb["velocity"])
    return np.asarray(
        [
            a["present"], b["present"], pair_present,
            distance_xy,
            aabb_gap(a, b) if pair_present else 50.0,
            oriented_box_gap(a, b) if pair_present else 50.0,
            closing,
            norm(relative_velocity) if pair_present else 0.0,
            speed_a, speed_b,
            norm(a["acceleration"]), norm(b["acceleration"]),
            norm(a["angular"]), norm(b["angular"]),
            max(abs(a["rotation"]["roll"]), abs(b["rotation"]["roll"])),
            max(abs(a["rotation"]["pitch"]), abs(b["rotation"]["pitch"])),
            math.sin(yaw_delta), math.cos(yaw_delta),
            abs(angle_delta(a["rotation"]["yaw"], pa["rotation"]["yaw"])) if a["present"] * pa["present"] else 0.0,
            abs(angle_delta(b["rotation"]["yaw"], pb["rotation"]["yaw"])) if b["present"] * pb["present"] else 0.0,
            max(
                abs(angle_delta(a["rotation"]["roll"], pa["rotation"]["roll"])) if a["present"] * pa["present"] else 0.0,
                abs(angle_delta(b["rotation"]["roll"], pb["rotation"]["roll"])) if b["present"] * pb["present"] else 0.0,
            ),
            abs(speed_a - previous_speed_a) + abs(speed_b - previous_speed_b),
            max(abs(float(a["velocity"][2])), abs(float(b["velocity"][2]))),
        ],
        dtype=np.float32,
    )


def recording_features(
    manifest: dict[str, Any],
    annotation: dict[str, Any],
    static_geometry: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    scenes = sorted(annotation.get("scene_state", []), key=lambda item: int(item["iteration"]))
    actor_ids = choose_actor_ids(manifest, scenes)
    use_static = (
        manifest.get("collision_mechanism") == "single_vehicle_static"
        and bool(static_geometry)
    )
    if use_static:
        actor_ids = actor_ids[:1] + [-1]
    static_candidates = []
    for item in static_geometry or []:
        box = item.get("bbox_world", {})
        location = vec(box.get("location"))
        corners = [
            [float(point.get(axis, 0.0)) for axis in ("x", "y", "z")]
            for point in box.get("corners", [])
        ]
        static_candidates.append((item, location, corners))
    previous: dict[int, dict[str, Any]] = {}
    features, iterations = [], []
    for scene in scenes:
        current = {int(obj["id"]): obj for obj in scene.get("objects", [])}
        if use_static and actor_ids and actor_ids[0] in current and static_candidates:
            dynamic_location = vec(current[actor_ids[0]].get("actor_transform", {}).get("location"))
            item, _, corners = min(
                static_candidates,
                key=lambda row: float(np.linalg.norm((row[1] - dynamic_location)[:2])),
            )
            box = item["bbox_world"]
            current[-1] = {
                "id": -1,
                "type_id": f"static_map.{item.get('semantic_label', 'unknown')}",
                "actor_transform": {
                    "location": box.get("location", {}),
                    "rotation": box.get("rotation", {}),
                },
                "bbox_3d_world": corners,
                "velocity_world_mps": {"x": 0.0, "y": 0.0, "z": 0.0},
                "acceleration_world_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular_velocity_world_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        features.append(frame_features(current, previous, actor_ids))
        iterations.append(
            len(iterations) + 1
            if manifest.get("capture_mode") == "async_rgb_only"
            else int(scene["iteration"])
        )
        previous = current
    return np.stack(features), iterations, actor_ids


