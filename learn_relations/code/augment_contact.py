"""Recover the original privileged CARLA contact-label generation, preserving historical target definitions.
Company-phase source, local organization only pending rights/licensing review; not published or relicensed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from teacher_inputs import load_eligible_manifest, load_json
from contact_physics import FEATURE_NAMES, load_static_geometry, recording_features
FEATURE_INDEX = {name:index for index,name in enumerate(FEATURE_NAMES)}

def values_at_ends(
    features: np.ndarray, iterations: list[int], ends: np.ndarray,
) -> dict[str, np.ndarray]:
    index_by_iteration = {int(iteration): index for index, iteration in enumerate(iterations)}
    missing = [int(end) for end in ends if int(end) not in index_by_iteration]
    if missing:
        raise ValueError(f"cache ends absent from privileged frames: {missing[:10]}")
    selected = features[[index_by_iteration[int(end)] for end in ends]]
    return {
        "contact_pair_present": selected[:, FEATURE_INDEX["pair_present"]].astype(np.float16),
        "contact_gap_m": selected[:, FEATURE_INDEX["oriented_box_gap_3d_m"]].astype(np.float16),
        "contact_relative_speed_mps": selected[:, FEATURE_INDEX["relative_speed_mps"]].astype(np.float16),
        "contact_speed_change_mps": selected[:, FEATURE_INDEX["sum_abs_speed_change_mps"]].astype(np.float16),
    }


def augment_record(cache_path: Path, static_geometry: list[dict[str, Any]]) -> dict[str, Any]:
    with np.load(cache_path) as data:
        payload = {key: data[key].copy() for key in data.files}
    manifest_path = Path(str(payload["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = (cache_path.parent / manifest_path).resolve()
    payload["manifest"] = np.asarray(str(manifest_path))
    manifest = load_eligible_manifest(manifest_path)
    annotation = load_json(Path(manifest["source_annotation"]))
    features, iterations, _ = recording_features(manifest, annotation, static_geometry)
    payload.update(values_at_ends(features, iterations, payload["ends"]))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--static-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    paths = sorted(args.cache_dir.glob("*.npz"))
    if not paths:
        raise ValueError("no source NPZ caches")
    static = load_static_geometry(args.static_geometry)
    args.output_dir.mkdir(parents=True)
    rows = []
    for path in paths:
        payload = augment_record(path, static)
        payload["manifest"] = np.asarray(os.path.relpath(str(payload["manifest"]),args.output_dir))
        payload["contact_label_source"] = np.asarray("original recording_features: XY-hull/z-gap, nearest-center static target, sum of per-recorded-frame absolute speed-norm changes; float16")
        destination = args.output_dir / path.name
        with destination.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        rows.append({"source_cache":str(path), "output_cache":str(destination), "windows":int(len(payload["ends"]))})
    report = {"schema_version":"recovered-carla-contact-teacher-1", "records":rows,
        "scope":"privileged training/evaluation targets only; historical box-gap approximation and static association retained; no accident recognition claim"}
    with (args.output_dir/"augmentation_report.json").open("x",encoding="utf-8") as handle:
        json.dump(report,handle,indent=2)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
