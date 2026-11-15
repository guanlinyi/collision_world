"""Join reviewed external contact-teacher labels to one token cache without inventing targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("schema_version") != "reviewed-contact-teacher-1" or not labels.get("provenance") or not labels.get("target_definitions"):
        raise ValueError("labels require schema_version, provenance and target_definitions")
    with np.load(args.cache, allow_pickle=False) as data:
        payload = {key:data[key].copy() for key in data.files}
    rows = labels["by_end_iteration"]
    gap = np.asarray([rows[str(int(end))]["gap_m"] for end in payload["ends"]], dtype=np.float32)
    speed = np.asarray([rows[str(int(end))]["speed_change_mps"] for end in payload["ends"]], dtype=np.float32)
    if not np.isfinite(gap).all() or not np.isfinite(speed).all() or np.any(gap < 0) or np.any(speed < 0):
        raise ValueError("contact targets must be finite and nonnegative")
    payload["contact_gap_m"] = gap
    payload["contact_speed_change_mps"] = speed
    # Keep portable manifest references valid after moving the cache into a new directory.
    old_manifest = Path(str(payload["manifest"]))
    if not old_manifest.is_absolute():
        old_manifest = (args.cache.parent / old_manifest).resolve()
    import os
    payload["manifest"] = np.asarray(os.path.relpath(old_manifest, args.output.parent))
    payload["contact_teacher_provenance_json"] = np.asarray(json.dumps({k:labels[k] for k in ("provenance","target_definitions")}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    print(args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
