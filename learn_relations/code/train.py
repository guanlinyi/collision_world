#!/usr/bin/env python3
"""Train an RGB-only physical relation student with a post-impact event head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metrics import (
    alarm_summary,
    regression_metrics,
)
from relation_model import RelationContactStudent
from relation_data import (
    concatenate,
    load_records,
    participant_mask_metrics,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optional_mean(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None

class RelationContactEventStudent(RelationContactStudent):
    """Retain physical auxiliary tasks while learning persistent post-impact evidence."""

    def __init__(self, width: int, grid_size: int, hidden: int = 64):
        super().__init__(width, grid_size, hidden)
        relation_width = hidden * 6 + 8
        self.event_head = nn.Sequential(
            nn.LayerNorm(relation_width),
            nn.Linear(relation_width, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(tokens)
        output["event_logit"] = self.event_head(output["relation_embedding"]).squeeze(1)
        return output


def debounce(probabilities: np.ndarray, threshold: float, consecutive: int) -> np.ndarray:
    raw = np.asarray(probabilities) >= threshold
    if consecutive < 1:
        raise ValueError("consecutive must be at least one")
    confirmed = np.zeros_like(raw, dtype=bool)
    for index in range(consecutive - 1, len(raw)):
        confirmed[index] = bool(np.all(raw[index - consecutive + 1:index + 1]))
    return confirmed


def evaluate_event_alarm(
    record: dict[str, Any], probabilities: np.ndarray, threshold: float,
    consecutive: int,
) -> dict[str, Any]:
    alarms = debounce(probabilities, threshold, consecutive)
    alarm_indices = np.flatnonzero(alarms)
    onset = record["onset"]
    early = [index for index in alarm_indices if onset is None or int(record["ends"][index]) < onset]
    timely: list[tuple[int, float]] = []
    for index in alarm_indices:
        if onset is None or int(record["ends"][index]) < onset:
            continue
        end_time = float(record["end_times_seconds"][index])
        onset_time = record.get("onset_time_seconds")
        delay = (
            max(0.0, end_time - float(onset_time))
            if onset_time is not None and np.isfinite(end_time)
            else (int(record["ends"][index]) - int(onset)) / float(record["fps"])
        )
        if delay <= 3.0:
            timely.append((int(index), float(delay)))
    positive = record["scenario_intent"] == "collision"
    strict = bool(timely and not early) if positive else not bool(early)
    before = [
        float(probabilities[index]) for index in range(len(probabilities))
        if onset is None or int(record["ends"][index]) < onset
    ]
    after = [
        float(probabilities[index]) for index in range(len(probabilities))
        if onset is not None and int(record["ends"][index]) >= onset
    ]
    return {
        "manifest": record["manifest"],
        "name": record["name"],
        "scenario_intent": record["scenario_intent"],
        "collision_mechanism": record["collision_mechanism"],
        "false_alarm": bool(early),
        "timely_alarm": bool(timely),
        "first_timely_alarm_delay_seconds": timely[0][1] if timely else None,
        "strict_pass": strict,
        "alarm_count": int(alarms.sum()),
        "maximum_probability_before_onset": max(before) if before else None,
        "maximum_probability_after_onset": max(after) if after else None,
    }


def select_event_policy(
    records: list[dict[str, Any]], predictions: list[np.ndarray],
) -> tuple[dict[str, float | int], dict[str, Any]]:
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 19):
        for consecutive in (1, 2, 3):
            rows = [
                evaluate_event_alarm(record, prediction, float(threshold), consecutive)
                for record, prediction in zip(records, predictions)
            ]
            summary = alarm_summary(rows)
            rank = (
                summary["strict_pass_recording_count"],
                summary["positive_timely_alarm_count"],
                -summary["positive_early_false_alarm_count"],
                -summary["normal_false_alarm_count"],
                float(threshold),
                consecutive,
            )
            candidates.append((rank, float(threshold), consecutive, summary))
    _, threshold, consecutive, summary = max(candidates, key=lambda item: item[0])
    return {
        "event_probability_threshold": threshold,
        "minimum_consecutive_windows": consecutive,
    }, summary


def predict(
    model: nn.Module, features: np.ndarray, target_mean: np.ndarray,
    target_scale: np.ndarray, batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contact, masks, event = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start:start + batch_size]).float().to(DEVICE)
            output = model(x)
            contact.append(output["contact"].cpu().numpy())
            masks.append(torch.sigmoid(output["slot_mask_logits"]).cpu().numpy())
            event.append(torch.sigmoid(output["event_logit"]).cpu().numpy())
    normalized = np.concatenate(contact)
    log_values = normalized * target_scale + target_mean
    return (
        np.expm1(np.maximum(log_values, 0.0)),
        np.concatenate(masks),
        np.concatenate(event),
    )


def event_window_metrics(
    records: list[dict[str, Any]], predictions: list[np.ndarray], threshold: float,
) -> dict[str, Any]:
    truth = np.concatenate([record["labels"] for record in records]).astype(bool)
    probability = np.concatenate(predictions)
    predicted = probability >= threshold
    positive_recall = float(predicted[truth].mean()) if np.any(truth) else None
    negative_specificity = float((~predicted[~truth]).mean()) if np.any(~truth) else None
    balanced = (
        float((positive_recall + negative_specificity) / 2.0)
        if positive_recall is not None and negative_specificity is not None else None
    )
    return {
        "positive_window_count": int(truth.sum()),
        "negative_window_count": int((~truth).sum()),
        "positive_window_recall": positive_recall,
        "negative_window_specificity": negative_specificity,
        "balanced_window_accuracy": balanced,
        "brier_score": float(np.mean((probability - truth.astype(np.float32)) ** 2)),
    }


def train_one(
    training: list[dict[str, Any]], calibration: list[dict[str, Any]],
    validation: list[dict[str, Any]], epochs: int, seed: int,
) -> dict[str, Any]:
    values = concatenate(training)
    event_targets = np.concatenate([record["labels"] for record in training]).astype(np.float32)
    contact_targets = np.stack((
        np.log1p(values["gap_m"]), np.log1p(values["speed_change_mps"]),
    ), axis=1).astype(np.float32)
    target_mean = contact_targets.mean(axis=0)
    target_scale = np.maximum(contact_targets.std(axis=0), 0.1)
    normalized = (contact_targets - target_mean) / target_scale
    weights = 1.0 + 4.0 * (values["gap_m"] <= 1.0) + 4.0 * (values["speed_change_mps"] >= 1.0)
    positive_count = max(float(event_targets.sum()), 1.0)
    negative_count = max(float(len(event_targets) - event_targets.sum()), 1.0)
    event_positive_weight = min(negative_count / positive_count, 8.0)

    torch.manual_seed(seed)
    model = RelationContactEventStudent(
        values["features"].shape[-1], training[0]["grid_size"]
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(values["features"]))
    for _ in range(epochs):
        rng.shuffle(indices)
        model.train()
        for start in range(0, len(indices), 16):
            index = indices[start:start + 16]
            x = torch.from_numpy(values["features"][index]).float().to(DEVICE)
            contact_y = torch.from_numpy(normalized[index]).float().to(DEVICE)
            sample_weight = torch.from_numpy(weights[index].astype(np.float32)).to(DEVICE)
            event_y = torch.from_numpy(event_targets[index]).float().to(DEVICE)
            slot_y = torch.from_numpy(values["participant_masks"][index]).float().to(DEVICE)
            slot_valid = torch.from_numpy(values["participant_valid"][index]).float().to(DEVICE)
            presence_y = torch.from_numpy(values["presence"][index]).float().to(DEVICE)
            depth_y = torch.from_numpy(np.log1p(values["depth_m"][index])).float().to(DEVICE)
            speed_y = torch.from_numpy(np.log1p(values["speed_mps"][index])).float().to(DEVICE)
            output = model(x)

            per_contact = F.smooth_l1_loss(output["contact"], contact_y, reduction="none").mean(1)
            contact_loss = (per_contact * sample_weight).sum() / sample_weight.sum()
            event_loss = F.binary_cross_entropy_with_logits(
                output["event_logit"], event_y,
                pos_weight=torch.as_tensor(event_positive_weight, device=x.device),
            )
            slot_bce = F.binary_cross_entropy_with_logits(
                output["slot_mask_logits"], slot_y, reduction="none"
            )
            slot_weight = 1.0 + 5.0 * slot_y
            valid_weight = slot_valid.unsqueeze(-1)
            slot_loss = (slot_bce * slot_weight * valid_weight).sum() / (
                (slot_weight * valid_weight).sum().clamp_min(1.0)
            )
            dense = output["dense"]
            presence_loss = F.binary_cross_entropy_with_logits(
                dense[:, :, 0], presence_y
            )
            occupied = presence_y > 0.5
            dense_physics_loss = (
                F.smooth_l1_loss(dense[:, :, 1][occupied], depth_y[occupied])
                + 0.5 * F.smooth_l1_loss(dense[:, :, 2][occupied], speed_y[occupied])
                if torch.any(occupied) else dense[:, :, 1:].sum() * 0
            )
            loss = (
                contact_loss + event_loss + 0.5 * slot_loss
                + 0.25 * presence_loss + 0.25 * dense_physics_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    calibration_outputs = [
        predict(model, record["features"], target_mean, target_scale)
        for record in calibration
    ]
    calibration_events = [item[2] for item in calibration_outputs]
    policy, calibration_alarm = select_event_policy(calibration, calibration_events)
    validation_outputs = [
        predict(model, record["features"], target_mean, target_scale)
        for record in validation
    ]
    validation_contact = [item[0] for item in validation_outputs]
    validation_masks = [item[1] for item in validation_outputs]
    validation_events = [item[2] for item in validation_outputs]
    rows = [
        evaluate_event_alarm(
            record, probability,
            float(policy["event_probability_threshold"]),
            int(policy["minimum_consecutive_windows"]),
        )
        for record, probability in zip(validation, validation_events)
    ]
    metrics = regression_metrics(validation, validation_contact)
    metrics.update(participant_mask_metrics(validation, validation_masks))
    metrics.update(event_window_metrics(
        validation, validation_events,
        float(policy["event_probability_threshold"]),
    ))
    return {
        "seed": seed,
        "event_positive_weight": event_positive_weight,
        "selected_event_policy_from_source_calibration": policy,
        "calibration_alarm_summary": calibration_alarm,
        "validation_metrics": metrics,
        "validation_alarm_summary": alarm_summary(rows),
        "validation_recordings": rows,
        "checkpoint": {
            "schema_version": "mask2world-relation-contact-event-student-checkpoint-0.1",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "input_width": int(values["features"].shape[-1]),
            "grid_size": int(training[0]["grid_size"]),
            "target_mean": torch.from_numpy(target_mean.astype(np.float32)),
            "target_scale": torch.from_numpy(target_scale.astype(np.float32)),
            "selected_event_policy_from_source_calibration": policy,
            "seed": seed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=("last", "temporal_delta", "fused"), default="fused")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--calibration-height", type=float, default=3.0,
                        help="source cameras below this height train; exact height calibrates the alarm policy")
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[20260907, 20260908, 20260909, 20260910, 20260911],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.checkpoint_dir and args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()):
        raise FileExistsError(f"checkpoint directory is not empty: {args.checkpoint_dir}")
    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source = load_records(args.source_cache_dir, args.feature_mode)
    validation = load_records([args.validation_cache_dir], args.feature_mode)
    training = [record for record in source if record["camera_height_m"] < args.calibration_height]
    calibration = [record for record in source if record["camera_height_m"] == args.calibration_height]
    if not training or not calibration or not validation:
        raise ValueError("training, source calibration and validation must all be non-empty")
    source_manifests = {record["manifest"] for record in training + calibration}
    if source_manifests.intersection(record["manifest"] for record in validation):
        raise ValueError("source and validation must contain disjoint recordings")

    runs = []
    for seed in args.seeds:
        run = train_one(training, calibration, validation, args.epochs, seed)
        checkpoint = run.pop("checkpoint")
        checkpoint["feature_mode"] = args.feature_mode
        if args.checkpoint_dir:
            checkpoint_path = args.checkpoint_dir / f"relation_contact_event_seed_{seed}.pt"
            with checkpoint_path.open("xb") as handle:
                torch.save(checkpoint, handle)
            run["checkpoint"] = str(checkpoint_path.resolve())
        runs.append(run)
        print(json.dumps({
            "seed": seed,
            "policy": run["selected_event_policy_from_source_calibration"],
            "metrics": run["validation_metrics"],
            "alarm": run["validation_alarm_summary"],
        }), flush=True)
        torch.cuda.empty_cache()

    aggregate = {
        "seed_count": len(runs),
        "mean_gap_mae_m_at_truth_gap_le_2m": optional_mean([
            run["validation_metrics"]["gap_mae_m_at_truth_gap_le_2m"] for run in runs
        ]),
        "mean_speed_change_mae_mps_at_truth_change_ge_2mps": optional_mean([
            run["validation_metrics"]["speed_change_mae_mps_at_truth_change_ge_2mps"]
            for run in runs
        ]),
        "mean_visible_participant_mask_iou": optional_mean([
            run["validation_metrics"]["mean_visible_participant_mask_iou"] for run in runs
        ]),
        "mean_event_balanced_window_accuracy": optional_mean([
            run["validation_metrics"]["balanced_window_accuracy"] for run in runs
        ]),
        "strict_business_pass_seed_count": sum(
            run["validation_alarm_summary"]["strict_business_pass"] for run in runs
        ),
        "mean_positive_timely_alarm_count": float(np.mean([
            run["validation_alarm_summary"]["positive_timely_alarm_count"] for run in runs
        ])),
        "mean_normal_false_alarm_count": float(np.mean([
            run["validation_alarm_summary"]["normal_false_alarm_count"] for run in runs
        ])),
    }
    payload = {
        "schema_version": "mask2world-carla-relation-contact-event-student-cross-view-0.1",
        "feature_mode": args.feature_mode,
        "teacher_warning": (
            "CARLA participant masks, 2.5D values and collision timing are training labels only. "
            "Validation model input is frozen RGB-derived V-JEPA tokens."
        ),
        "controlled_split": (
            f"Source records below {args.calibration_height} m train; source records at "
            f"{args.calibration_height} m calibrate event threshold/debounce; "
            "the explicit validation-cache-dir is evaluation-only."
        ),
        "training_recording_count": len(training),
        "calibration_recording_count": len(calibration),
        "validation_recording_count": len(validation),
        "epochs": args.epochs,
        "aggregate": aggregate,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"aggregate": aggregate, "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
