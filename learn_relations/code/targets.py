"""Build RGB-aligned 2.5D patch targets from CARLA training-only truth."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def actor_depth_m(obj: dict[str, Any]) -> float | None:
    corners = np.asarray(obj.get("bbox_3d_sensor", []), dtype=np.float32)
    if corners.ndim != 2 or corners.shape[1] < 1:
        return None
    positive = corners[:, 0][corners[:, 0] > 0]
    return None if not len(positive) else float(np.median(positive))


def actor_speed_mps(obj: dict[str, Any]) -> float:
    velocity = obj.get("velocity_world_mps") or {}
    return math.sqrt(sum(float(velocity.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z")))


def vector_in_sensor_axes(
    vector: dict[str, Any] | None,
    world_to_camera_matrix: list[list[float]] | np.ndarray | None,
) -> np.ndarray:
    """Rotate a world vector into CARLA camera axes without applying translation."""
    world = np.asarray([
        float((vector or {}).get(axis, 0.0)) for axis in ("x", "y", "z")
    ], dtype=np.float32)
    if world_to_camera_matrix is None:
        return world
    matrix = np.asarray(world_to_camera_matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"world_to_camera_matrix must be 4x4, got {matrix.shape}")
    return matrix[:3, :3] @ world


def actor_class_id(obj: dict[str, Any]) -> int:
    type_id = str(obj.get("type_id", "")).lower()
    if type_id.startswith("walker."):
        return 2
    if type_id.startswith("vehicle.") and any(
        word in type_id for word in ("ninja", "motorcycle", "bike", "bicycle", "scooter")
    ):
        return 3
    return 1 if type_id.startswith("vehicle.") else 0


def rasterize_patch_targets(
    frame: dict[str, Any],
    image_width: int,
    image_height: int,
    grid_size: int = 24,
    input_size: int = 384,
    world_to_camera_matrix: list[list[float]] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    presence = np.zeros((grid_size, grid_size), dtype=np.float32)
    depth = np.zeros_like(presence)
    speed = np.zeros_like(presence)
    velocity_sensor = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    angular_speed = np.zeros_like(presence)
    class_id = np.zeros((grid_size, grid_size), dtype=np.uint8)
    owner_depth = np.full_like(presence, np.inf)
    scale = input_size / max(image_width, image_height)
    resized_width = round(image_width * scale)
    resized_height = round(image_height * scale)
    pad_x = (input_size - resized_width) / 2.0
    pad_y = (input_size - resized_height) / 2.0
    patch = input_size / grid_size

    for obj in frame.get("objects", []):
        bbox = obj.get("2d_bbox")
        object_depth = actor_depth_m(obj)
        if not bbox or object_depth is None:
            continue
        (x0, y0), (x1, y1) = bbox
        x0, x1 = sorted((float(x0) * scale + pad_x, float(x1) * scale + pad_x))
        y0, y1 = sorted((float(y0) * scale + pad_y, float(y1) * scale + pad_y))
        candidates = []
        for gy in range(grid_size):
            cy = (gy + 0.5) * patch
            if y0 <= cy <= y1:
                for gx in range(grid_size):
                    cx = (gx + 0.5) * patch
                    if x0 <= cx <= x1:
                        candidates.append((gy, gx))
        if not candidates:
            gx = int(np.clip(round(((x0 + x1) / 2.0) / patch - 0.5), 0, grid_size - 1))
            gy = int(np.clip(round(((y0 + y1) / 2.0) / patch - 0.5), 0, grid_size - 1))
            candidates = [(gy, gx)]
        object_speed = actor_speed_mps(obj)
        object_velocity_sensor = vector_in_sensor_axes(
            obj.get("velocity_world_mps"), world_to_camera_matrix
        )
        angular = obj.get("angular_velocity_world_dps") or {}
        object_angular_speed = math.sqrt(
            sum(float(angular.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z"))
        )
        for gy, gx in candidates:
            if object_depth < owner_depth[gy, gx]:
                owner_depth[gy, gx] = object_depth
                presence[gy, gx] = 1.0
                depth[gy, gx] = object_depth
                speed[gy, gx] = object_speed
                velocity_sensor[:, gy, gx] = object_velocity_sensor
                angular_speed[gy, gx] = object_angular_speed
                class_id[gy, gx] = actor_class_id(obj)
    return {
        "presence": presence,
        "depth_m": depth,
        "speed_mps": speed,
        "velocity_sensor_mps": velocity_sensor,
        "angular_speed_dps": angular_speed,
        "class_id": class_id,
    }


def patch_center_in_original(
    gy: int,
    gx: int,
    image_width: int,
    image_height: int,
    grid_size: int = 24,
    input_size: int = 384,
) -> tuple[float, float]:
    scale = input_size / max(image_width, image_height)
    pad_x = (input_size - round(image_width * scale)) / 2.0
    pad_y = (input_size - round(image_height * scale)) / 2.0
    patch = input_size / grid_size
    return (((gx + 0.5) * patch - pad_x) / scale,
            ((gy + 0.5) * patch - pad_y) / scale)
