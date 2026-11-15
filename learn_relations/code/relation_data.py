from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from cache_io import load_record, manifest_metadata

def load_records(cache_dirs: list[Path], feature_mode: str) -> list[dict[str, Any]]:
    records = []
    for cache_dir in cache_dirs:
        for path in sorted(cache_dir.glob("*.npz")):
            with np.load(path) as data:
                required = (
                    "contact_gap_m", "contact_speed_change_mps",
                    "relation_participant_masks", "relation_participant_valid",
                )
                if any(key not in data.files for key in required):
                    raise ValueError(f"relation/contact targets are absent: {path}")
                if feature_mode == "last":
                    features = data["tokens"].copy()
                elif feature_mode == "temporal_delta":
                    features = data["temporal_delta_tokens"].copy()
                else:
                    features = np.concatenate(
                        (data["tokens"], data["temporal_delta_tokens"]), axis=2
                    )
                extra = {
                    "features": features,
                    "gap_m": data["contact_gap_m"].astype(np.float32),
                    "speed_change_mps": data["contact_speed_change_mps"].astype(np.float32),
                    "participant_masks": data["relation_participant_masks"].astype(np.float32),
                    "participant_valid": data["relation_participant_valid"].astype(np.float32),
                    "static_second": data["relation_static_second"].astype(np.float32),
                }
            base = load_record(path)
            record = manifest_metadata(base)
            manifest = json.loads(Path(record["manifest"]).read_text(encoding="utf-8"))
            record.update(extra)
            record.update({
                "presence": base["presence"].astype(np.float32),
                "depth_m": base["depth_m"].astype(np.float32),
                "speed_mps": base["speed_mps"].astype(np.float32),
                "name": Path(record["manifest"]).stem,
                "collision_mechanism": manifest["collision_mechanism"],
            })
            records.append(record)
    return records


def concatenate(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([record[key] for record in records])
        for key in (
            "features", "gap_m", "speed_change_mps", "participant_masks",
            "participant_valid", "presence", "depth_m", "speed_mps",
        )
    }


def participant_mask_metrics(records: list[dict[str, Any]], predictions: list[np.ndarray]) -> dict[str, Any]:
    truth = np.concatenate([record["participant_masks"] for record in records]).astype(bool)
    valid = np.concatenate([record["participant_valid"] for record in records]).astype(bool)
    predicted = np.concatenate(predictions) >= .5
    ious = []
    for window in range(len(truth)):
        for slot in range(2):
            if not valid[window, slot]:
                continue
            intersection = np.sum(truth[window, slot] & predicted[window, slot])
            union = np.sum(truth[window, slot] | predicted[window, slot])
            ious.append(intersection / max(union, 1))
    return {
        "visible_slot_count": len(ious),
        "mean_visible_participant_mask_iou": float(np.mean(ious)) if ious else None,
    }


