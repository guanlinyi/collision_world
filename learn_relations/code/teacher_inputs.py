from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import gzip
from collections import Counter
MANIFEST_SCHEMA = 'mask2world-carla-collision-episodes-0.2'

def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def load_eligible_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {path}")
    policy = payload.get("training_use_policy", {})
    if policy.get("status") != "eligible" or policy.get("eligible") is not True:
        issues = policy.get("blocking_quality_issues", [])
        raise ValueError(f"training gate rejected {path}: {issues}")
    if not policy.get("allowed_logical_iteration_ranges"):
        raise ValueError(f"training gate has no allowed frame range: {path}")
    annotation = Path(payload["source_annotation"])
    if not annotation.is_absolute():
        payload["source_annotation"] = str((path.parent / annotation).resolve())
    return payload


def allowed(iteration: int, ranges: list[list[int]]) -> bool:
    return any(int(start) <= iteration <= int(end) for start, end in ranges)


def make_windows(
    manifest: dict[str, Any],
    features: np.ndarray,
    iterations: list[int],
    clip_length: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    policy = manifest["training_use_policy"]
    ranges = policy["allowed_logical_iteration_ranges"]
    positive_iterations: set[int] = set()
    excluded_late: set[int] = set()
    for incident in manifest.get("incidents", []):
        intervals = incident["training_intervals"]
        start, end = intervals["timely_alarm_allowed"]
        positive_iterations.update(range(int(start), int(end) + 1))
        late = intervals.get("late_after_deadline")
        if late:
            excluded_late.update(range(int(late[0]), int(late[1]) + 1))
    clips, labels, ends = [], [], []
    for end_index in range(clip_length - 1, len(iterations)):
        clip_iterations = iterations[end_index - clip_length + 1 : end_index + 1]
        end_iteration = iterations[end_index]
        if not all(allowed(value, ranges) for value in clip_iterations):
            continue
        if end_iteration in excluded_late:
            continue
        clips.append(features[end_index - clip_length + 1 : end_index + 1])
        labels.append(1.0 if end_iteration in positive_iterations else 0.0)
        ends.append(end_iteration)
    if not clips:
        raise ValueError("recording produced no training windows")
    return np.stack(clips), np.asarray(labels, dtype=np.float32), ends


def choose_actor_ids(
    manifest: dict[str, Any], scenes: list[dict[str, Any]]
) -> list[int]:
    incident_ids = {
        int(actor_id)
        for incident in manifest.get("incidents", [])
        for actor_id in incident.get("actor_ids", [])
        if int(actor_id) != 0
    }
    counts: Counter[int] = Counter()
    type_ids: dict[int, str] = {}
    for scene in scenes:
        for obj in scene.get("objects", []):
            actor_id = int(obj["id"])
            counts[actor_id] += 1
            type_ids[actor_id] = str(obj.get("type_id", ""))
    candidates = incident_ids or set(counts)
    ordered = sorted(candidates, key=lambda actor_id: (type_ids.get(actor_id, ""), actor_id))
    return sorted(ordered[:2], key=lambda actor_id: (type_ids.get(actor_id, ""), actor_id))


