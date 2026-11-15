from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import argparse
import hashlib
import math
import torch
from torch import nn
from encoder import build_encoder
from teacher_inputs import load_eligible_manifest, load_json, choose_actor_ids
from rgb_inputs import selected_windows, base_by_training_iteration, load_rgb_tensor
from targets import rasterize_patch_targets

def cache_name(manifest_path: Path) -> str:
    digest = hashlib.sha1(str(manifest_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{manifest_path.stem}_{digest}.npz"


@torch.inference_mode()
def extract_record(
    encoder: nn.Module,
    manifest_path: Path,
    clip_length: int,
    stride: int,
    batch_size: int,
) -> dict[str, Any]:
    manifest = load_eligible_manifest(manifest_path)
    annotation = load_json(Path(manifest["source_annotation"]))
    clips, _, ends = selected_windows(manifest, annotation, clip_length, stride)
    base_by_iteration = base_by_training_iteration(manifest, annotation)
    size = annotation["sensor"]["image_size"]
    width, height = int(size["width"]), int(size["height"])
    token_batches = []
    temporal_delta_batches = []
    for start in range(0, len(clips), batch_size):
        tensors = []
        for paths in clips[start:start + batch_size]:
            tensors.append(torch.stack([load_rgb_tensor(path) for path in paths], dim=1))
        inputs = torch.stack(tensors).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = encoder(inputs)
        temporal = tokens.reshape(tokens.shape[0], clip_length // 2, -1, tokens.shape[-1])
        token_batches.append(temporal[:, -1].float().cpu().numpy().astype(np.float16))
        # CARLA surveillance cameras are static.  The difference between the
        # first and last temporal token in the same clip suppresses fixed
        # scene appearance while retaining motion/change evidence.  Keep it
        # alongside the original final token so experiments can compare the
        # representations without changing labels or sampling.
        temporal_delta_batches.append(
            (temporal[:, -1] - temporal[:, 0]).float().cpu().numpy().astype(np.float16)
        )
    spatial_tokens = np.concatenate(token_batches)
    temporal_delta_tokens = np.concatenate(temporal_delta_batches)
    grid_size = int(round(math.sqrt(spatial_tokens.shape[1])))
    if grid_size * grid_size != spatial_tokens.shape[1]:
        raise ValueError(f"non-square V-JEPA patch layout: {spatial_tokens.shape}")
    presence, depth, speed, class_id = [], [], [], []
    for end in ends:
        target = rasterize_patch_targets(
            base_by_iteration[int(end)], width, height, grid_size=grid_size
        )
        presence.append(target["presence"].reshape(-1))
        depth.append(target["depth_m"].reshape(-1))
        speed.append(target["speed_mps"].reshape(-1))
        class_id.append(target["class_id"].reshape(-1))
    return {
        "manifest": str(manifest_path.resolve()),
        "collision_mechanism": manifest["collision_mechanism"],
        "tokens": spatial_tokens,
        "temporal_delta_tokens": temporal_delta_tokens,
        "presence": np.asarray(presence, dtype=np.uint8),
        "depth_m": np.asarray(depth, dtype=np.float16),
        "speed_mps": np.asarray(speed, dtype=np.float16),
        "class_id": np.asarray(class_id, dtype=np.uint8),
        "ends": ends.astype(np.int32),
        "grid_size": grid_size,
    }


def participant_ids(manifest: dict[str, Any], annotation: dict[str, Any]) -> tuple[list[int], bool]:
    scenes = sorted(annotation.get("scene_state", []), key=lambda item: int(item["iteration"]))
    actor_ids = choose_actor_ids(manifest, scenes)
    static_second = manifest.get("collision_mechanism") == "single_vehicle_static"
    if static_second:
        actor_ids = actor_ids[:1]
    return actor_ids[:2], static_second


def actor_patch_target(
    frame: dict[str, Any], actor_id: int, width: int, height: int, grid_size: int,
) -> dict[str, np.ndarray]:
    selected = [obj for obj in frame.get("objects", []) if int(obj.get("id", -999999)) == actor_id]
    return rasterize_patch_targets(
        {"objects": selected}, width, height, grid_size=grid_size
    )


def participant_targets(
    manifest: dict[str, Any], annotation: dict[str, Any], ends: np.ndarray, grid_size: int,
) -> dict[str, np.ndarray]:
    actor_ids, static_second = participant_ids(manifest, annotation)
    image_size = annotation["sensor"]["image_size"]
    width, height = int(image_size["width"]), int(image_size["height"])
    frames = base_by_training_iteration(manifest, annotation)
    masks, depths, speeds, valid = [], [], [], []
    for end in ends:
        frame = frames[int(end)]
        window_masks = np.zeros((2, grid_size * grid_size), dtype=np.uint8)
        window_depths = np.zeros(2, dtype=np.float32)
        window_speeds = np.zeros(2, dtype=np.float32)
        window_valid = np.zeros(2, dtype=np.uint8)
        for slot, actor_id in enumerate(actor_ids):
            target = actor_patch_target(frame, actor_id, width, height, grid_size)
            mask = target["presence"].reshape(-1) > 0
            if np.any(mask):
                window_masks[slot] = mask.astype(np.uint8)
                window_depths[slot] = float(np.median(target["depth_m"].reshape(-1)[mask]))
                window_speeds[slot] = float(np.median(target["speed_mps"].reshape(-1)[mask]))
                window_valid[slot] = 1
        masks.append(window_masks)
        depths.append(window_depths)
        speeds.append(window_speeds)
        valid.append(window_valid)
    return {
        "relation_participant_masks": np.asarray(masks, dtype=np.uint8),
        "relation_participant_depth_m": np.asarray(depths, dtype=np.float16),
        "relation_participant_speed_mps": np.asarray(speeds, dtype=np.float16),
        "relation_participant_valid": np.asarray(valid, dtype=np.uint8),
        "relation_static_second": np.full(len(ends), static_second, dtype=np.uint8),
        "relation_actor_ids": np.asarray(actor_ids + [-1] * (2 - len(actor_ids)), dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract frozen RGB spatial tokens plus CARLA training-only dense/participant labels.")
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.stride < 1 or args.batch_size < 1:
        raise ValueError("stride and batch size must be positive")
    encoder = build_encoder(str(args.checkpoint), args.clip_length, args.vjepa_root)
    args.output_dir.mkdir(parents=True)
    for manifest_path in args.manifests:
        record = extract_record(encoder, manifest_path, args.clip_length, args.stride, args.batch_size)
        manifest = load_eligible_manifest(manifest_path)
        annotation = load_json(Path(manifest["source_annotation"]))
        record.update(participant_targets(manifest, annotation, record["ends"], record["grid_size"]))
        # Contact targets are deliberately not invented: attach reviewed external teacher labels separately.
        destination = args.output_dir / cache_name(manifest_path)
        with destination.open("xb") as handle:
            np.savez_compressed(handle, **record)
        print(destination)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
