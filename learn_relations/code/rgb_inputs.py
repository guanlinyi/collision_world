from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import cv2
import torch
from teacher_inputs import make_windows
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

def load_rgb_tensor(path: Path, size: int = 384) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read RGB frame: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    canvas = np.broadcast_to(IMAGE_MEAN, (size, size, 3)).copy()
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    canvas = (canvas - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(canvas).permute(2, 0, 1)


def selected_windows(
    manifest: dict[str, Any], annotation: dict[str, Any], clip_length: int, stride: int
) -> tuple[list[list[Path]], np.ndarray, np.ndarray]:
    base = sorted(annotation["base"], key=lambda item: int(item["iteration"]))
    iterations = (
        list(range(1, len(base) + 1))
        if manifest.get("capture_mode") == "async_rgb_only"
        else [int(item["iteration"]) for item in base]
    )
    _, labels, ends = make_windows(
        manifest, np.zeros((len(base), 1), dtype=np.float32), iterations, clip_length
    )
    end_to_label = {int(end): float(label) for end, label in zip(ends, labels)}
    selected_ends = set(ends[::stride])
    for incident in manifest.get("incidents", []):
        onset = int(incident["onset_logical_iteration"])
        if onset in end_to_label:
            selected_ends.add(onset)
    selected_ends = sorted(selected_ends)
    iteration_to_index = {iteration: index for index, iteration in enumerate(iterations)}
    annotation_path = Path(manifest["source_annotation"])
    image_dir = Path(manifest["image_dir"]) if manifest.get("image_dir") else annotation_path.parents[2] / "images" / "train"
    if not image_dir.is_absolute():
        image_dir = annotation_path.parent / image_dir
    clips: list[list[Path]] = []
    selected_labels = []
    for end in selected_ends:
        end_index = iteration_to_index[end]
        rows = base[end_index - clip_length + 1:end_index + 1]
        if len(rows) != clip_length:
            raise ValueError(f"incomplete RGB window ending at iteration {end}")
        paths = [image_dir / str(row["file_name"]) for row in rows]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing RGB frames: {missing[:3]}")
        clips.append(paths)
        selected_labels.append(end_to_label[end])
    return clips, np.asarray(selected_labels, dtype=np.float32), np.asarray(selected_ends)


def base_by_training_iteration(
    manifest: dict[str, Any], annotation: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Index source frames by the timebase used in the episode manifest."""
    base = sorted(annotation["base"], key=lambda item: int(item["iteration"]))
    if manifest.get("capture_mode") == "async_rgb_only":
        return {index: frame for index, frame in enumerate(base, start=1)}
    return {int(frame["iteration"]): frame for frame in base}


