"""Exercise real student training, checkpoint serialization and label-free inference on fabricated tiny tokens.
This is an implementation check, not a CARLA experiment or a model-quality evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    rng = np.random.default_rng(20261115)
    records = []
    for name, height, split in (("train", 2., "source"), ("calibration", 3., "source"), ("heldout", 5., "heldout")):
        annotation = root / "input_json" / (name + "_annotation.json")
        manifest = root / "input_json" / (name + "_manifest.json")
        cache = root / "input_npz" / split / (name + ".npz")
        write_json(annotation, {"sensor":{"location":{"z":height}}, "scene_state":[{"iteration":i,"simulation_time_seconds":i/10.} for i in range(1,9)]})
        write_json(manifest, {"source_annotation":annotation.name, "scenario_intent":"collision", "camera_height_m":height,
            "collision_mechanism":"synthetic_smoke_only", "fps":10., "incidents":[{"onset_logical_iteration":5, "onset_simulation_time_seconds":.5,
            "training_intervals":{"timely_alarm_allowed":[5,8]}}]})
        cache.parent.mkdir(parents=True, exist_ok=True)
        masks = np.zeros((8,2,4),np.uint8); masks[:,0,:2]=1; masks[:,1,2:]=1
        with cache.open("xb") as handle:
            np.savez_compressed(handle, manifest=np.asarray(os.path.relpath(manifest,cache.parent)),
                collision_mechanism=np.asarray("synthetic_smoke_only"), tokens=rng.normal(size=(8,4,8)).astype(np.float32),
                temporal_delta_tokens=rng.normal(size=(8,4,8)).astype(np.float32), grid_size=np.asarray(2),
                presence=np.ones((8,4),np.uint8), depth_m=np.full((8,4),5.,np.float32), speed_mps=np.full((8,4),3.,np.float32),
                ends=np.arange(1,9), contact_gap_m=np.linspace(3,0,8,dtype=np.float32), contact_speed_change_mps=np.linspace(0,4,8,dtype=np.float32),
                relation_participant_masks=masks, relation_participant_valid=np.ones((8,2),np.uint8), relation_static_second=np.zeros(8,np.uint8))
        records.append(cache)
    code = Path(__file__).resolve().parent
    commands = [
        [sys.executable,"-B",str(code/"train.py"),"--source-cache-dir",str(root/"input_npz/source"),"--validation-cache-dir",str(root/"input_npz/heldout"),
         "--output",str(root/"output_json/training.json"),"--checkpoint-dir",str(root/"output_pt"),"--epochs","2","--seeds","20261115","--device","cpu","--threads","2"],
        [sys.executable,"-B",str(code/"infer.py"),"--checkpoint",str(root/"output_pt/relation_contact_event_seed_20261115.pt"),"--tokens",str(records[-1]),"--output",str(root/"output_npz/predictions.npz"),"--threads","2"]]
    results = []
    (root/"output_txt").mkdir()
    for index, command in enumerate(commands):
        result = subprocess.run(command,capture_output=True,text=True)
        with (root/"output_txt"/f"step_{index}.txt").open("x",encoding="utf-8") as handle:
            handle.write(result.stdout+result.stderr)
        results.append({"command":command,"exit_code":result.returncode})
        if result.returncode:
            write_json(root/"output_json/acceptance.json",{"scope":"synthetic implementation check only","steps":results})
            print(result.stdout+result.stderr)
            return result.returncode
    with np.load(root/"output_npz/predictions.npz",allow_pickle=False) as values:
        shapes={key:list(values[key].shape) for key in values.files}
        assert shapes["event_probability"] == [8]
        assert shapes["participant_mask_probability"] == [8,2,4]
        assert all(np.isfinite(values[key]).all() for key in values.files)
    # Teacher labels are not needed during inference: remove them from a NEW second input, preserving original evidence.
    token_only=root/"input_npz/tokens_only.npz"
    with np.load(records[-1],allow_pickle=False) as values, token_only.open("xb") as handle:
        np.savez_compressed(handle,tokens=values["tokens"],temporal_delta_tokens=values["temporal_delta_tokens"])
    from infer import infer
    repeated=infer(root/"output_pt/relation_contact_event_seed_20261115.pt",token_only)
    with np.load(root/"output_npz/predictions.npz",allow_pickle=False) as values:
        difference=max(float(np.max(np.abs(repeated[k].astype(float)-values[k].astype(float)))) for k in values.files)
    assert difference < 1e-6, difference
    report={"scope":"fabricated tokens and labels; real model/loss/optimizer/checkpoint/inference, no V-JEPA encoder, no CARLA quality claim", "steps":results,
        "prediction_shapes":shapes,"maximum_difference_after_removing_all_teacher_fields":difference,"python":sys.version,"status":"passed"}
    write_json(root/"output_json/acceptance.json",report)
    print(json.dumps(report,indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
