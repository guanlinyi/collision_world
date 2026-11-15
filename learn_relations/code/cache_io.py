from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import gzip

def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def load_record(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        record = {
            "manifest": str((path.parent / str(data["manifest"])).resolve()) if not Path(str(data["manifest"])).is_absolute() else str(data["manifest"]),
            "collision_mechanism": str(data["collision_mechanism"]),
            "tokens": data["tokens"].copy(),
            "temporal_delta_tokens": (
                data["temporal_delta_tokens"].copy()
                if "temporal_delta_tokens" in data.files
                else None
            ),
            "presence": data["presence"].copy(),
            "depth_m": data["depth_m"].copy(),
            "speed_mps": data["speed_mps"].copy(),
            "class_id": (
                data["class_id"].copy() if "class_id" in data.files
                else np.zeros_like(data["presence"], dtype=np.uint8)
            ),
            "ends": data["ends"].copy(),
            "grid_size": int(data["grid_size"]),
        }
        for key in ("motion_change", "motion_valid", "motion_names", "motion_history_frames"):
            if key in data.files:
                record[key] = data[key].copy()
        return record


def manifest_metadata(record: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(record["manifest"]).read_text(encoding="utf-8"))
    annotation_path = Path(manifest["source_annotation"])
    if not annotation_path.is_absolute():
        annotation_path = Path(record["manifest"]).parent / annotation_path
    annotation = load_json(annotation_path)
    positive = set()
    for incident in manifest.get("incidents", []):
        start, end = incident["training_intervals"]["timely_alarm_allowed"]
        positive.update(range(int(start), int(end) + 1))
    labels = np.asarray([int(end) in positive for end in record["ends"]], dtype=np.float32)
    scenes = sorted(annotation["scene_state"], key=lambda row: int(row["iteration"]))
    if manifest.get("capture_mode") == "async_rgb_only":
        time_by_iteration = {
            index: row.get("simulation_time_seconds")
            for index, row in enumerate(scenes, start=1)
        }
    else:
        time_by_iteration = {
            int(row["iteration"]): row.get("simulation_time_seconds") for row in scenes
        }
    onset = min(
        (int(row["onset_logical_iteration"]) for row in manifest.get("incidents", [])),
        default=None,
    )
    onset_time = min(
        (
            float(row["onset_simulation_time_seconds"])
            for row in manifest.get("incidents", [])
            if row.get("onset_simulation_time_seconds") is not None
        ),
        default=None,
    )
    return {
        **record,
        "features": record["tokens"],
        "labels": labels,
        "scenario_intent": manifest["scenario_intent"],
        # Reviewed batch04+ manifests store the height explicitly.  Earlier
        # eligible manifests predate that field, but their privileged sensor
        # calibration contains the exact same camera z coordinate.
        "camera_height_m": float(
            manifest.get("camera_height_m", annotation["sensor"]["location"]["z"])
        ),
        "fps": float(manifest["fps"]),
        "onset": onset,
        "onset_time_seconds": onset_time,
        "end_times_seconds": np.asarray([
            np.nan if time_by_iteration.get(int(end)) is None else time_by_iteration[int(end)]
            for end in record["ends"]
        ], dtype=np.float64),
    }


