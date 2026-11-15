#!/usr/bin/env python3
"""Export CARLA map static-object 3D boxes as a reusable JSON sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable



def vector(value) -> dict[str, float]:
    return {axis: float(getattr(value, axis)) for axis in ("x", "y", "z")}


def rotation(value) -> dict[str, float]:
    return {axis: float(getattr(value, axis)) for axis in ("pitch", "yaw", "roll")}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Walls", "Fences", "Poles", "TrafficSigns"],
        help="CARLA CityObjectLabel names",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output = args.output_dir / "static_geometry.json"
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    rows = []
    label_counts = {}
    for label_name in args.labels:
        try:
            label = getattr(carla.CityObjectLabel, label_name)
        except AttributeError as error:
            raise ValueError(f"unknown CARLA CityObjectLabel: {label_name}") from error
        objects = world.get_environment_objects(label)
        label_counts[label_name] = len(objects)
        for obj in objects:
            bbox = obj.bounding_box
            corners = bbox.get_world_vertices(carla.Transform())
            rows.append(
                {
                    "id": int(obj.id),
                    "semantic_label": label_name,
                    "name": str(getattr(obj, "name", "")),
                    "bbox_world": {
                        "location": vector(bbox.location),
                        "extent": vector(bbox.extent),
                        "rotation": rotation(bbox.rotation),
                        "corners": [vector(point) for point in corners],
                    },
                }
            )
    payload = {
        "schema_version": "mask2world-carla-static-geometry-0.1",
        "map_name": world.get_map().name,
        "coordinate_system": "CARLA_world_meters",
        "label_counts": label_counts,
        "objects": rows,
    }
    args.output_dir.mkdir(parents=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "map_name": payload["map_name"],
        "label_counts": label_counts,
        "object_count": len(rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
