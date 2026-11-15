"""Infer saved relation/event heads from RGB-derived token arrays; never read teacher targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train import RelationContactEventStudent, debounce


def infer(checkpoint_path, tokens_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "mask2world-relation-contact-event-student-checkpoint-0.1":
        raise ValueError("unsupported checkpoint schema")
    model = RelationContactEventStudent(int(checkpoint["input_width"]), int(checkpoint["grid_size"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    with np.load(tokens_path, allow_pickle=False) as data:
        features = data["tokens"].astype(np.float32)
        mode = checkpoint["feature_mode"]
        if mode == "temporal_delta":
            features = data["temporal_delta_tokens"].astype(np.float32)
        elif mode == "fused":
            features = np.concatenate((features, data["temporal_delta_tokens"].astype(np.float32)), axis=2)
    expected = (int(checkpoint["grid_size"]) ** 2, int(checkpoint["input_width"]))
    if features.ndim != 3 or features.shape[1:] != expected or not len(features) or not np.isfinite(features).all():
        raise ValueError(f"expected finite [windows, {expected[0]}, {expected[1]}] tokens")
    contacts, masks, events = [], [], []
    with torch.inference_mode():
        for start in range(0, len(features), 32):
            result = model(torch.from_numpy(features[start:start+32]).to(device))
            mean = checkpoint["target_mean"].to(device)
            scale = checkpoint["target_scale"].to(device)
            contacts.append(torch.expm1(torch.clamp_min(result["contact"] * scale + mean, 0)).cpu().numpy())
            masks.append(result["slot_mask_logits"].sigmoid().cpu().numpy())
            events.append(result["event_logit"].sigmoid().cpu().numpy())
    events = np.concatenate(events)
    policy = checkpoint["selected_event_policy_from_source_calibration"]
    return {"contact_gap_and_speed_change": np.concatenate(contacts),
            "participant_mask_probability": np.concatenate(masks),
            "event_probability": events,
            "event_alarm": debounce(events, float(policy["event_probability_threshold"]), int(policy["minimum_consecutive_windows"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    values = infer(args.checkpoint, args.tokens, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **values)
    print(json.dumps({"windows": len(values["event_probability"]), "output": str(args.output)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
