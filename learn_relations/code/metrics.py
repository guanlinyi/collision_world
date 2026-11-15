from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
def regression_metrics(records: list[dict[str, Any]], predictions: list[np.ndarray]) -> dict[str, Any]:
    gap = np.concatenate([record["gap_m"] for record in records])
    change = np.concatenate([record["speed_change_mps"] for record in records])
    predicted = np.concatenate(predictions)
    near = gap <= 2.0
    impact = change >= 2.0
    return {
        "window_count": int(len(gap)),
        "gap_mae_m_all": float(np.mean(np.abs(predicted[:, 0] - gap))),
        "gap_mae_m_at_truth_gap_le_2m": float(np.mean(np.abs(predicted[near, 0] - gap[near]))) if np.any(near) else None,
        "near_window_count": int(near.sum()),
        "speed_change_mae_mps_all": float(np.mean(np.abs(predicted[:, 1] - change))),
        "speed_change_mae_mps_at_truth_change_ge_2mps": float(
            np.mean(np.abs(predicted[impact, 1] - change[impact]))
        ) if np.any(impact) else None,
        "impact_window_count": int(impact.sum()),
    }


def alarm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["scenario_intent"] == "collision"]
    normals = [row for row in rows if row["scenario_intent"] != "collision"]
    return {
        "strict_pass_recording_count": sum(row["strict_pass"] for row in rows),
        "recording_count": len(rows),
        "positive_timely_alarm_count": sum(row["timely_alarm"] for row in positives),
        "positive_recording_count": len(positives),
        "positive_early_false_alarm_count": sum(row["false_alarm"] for row in positives),
        "normal_false_alarm_count": sum(row["false_alarm"] for row in normals),
        "normal_recording_count": len(normals),
        "strict_business_pass": all(row["strict_pass"] for row in rows),
    }


