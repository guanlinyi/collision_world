# Learn RGB relations from CARLA supervision

Provenance: this route originated during an earlier company project. This repository includes the recovered training, teacher-generation and inference components, without company recordings, enterprise adapters or private services. The later geometry route is personal research. Original attribution and third-party terms are retained; this repository does not grant an umbrella license over external code, assets or weights.

This route freezes a V-JEPA 2.1 RGB encoder and learns a two-participant relation/event student. The recovered architecture predicts participant masks, dense presence/depth/speed auxiliaries, pair gap/speed change, and a post-impact event score. CARLA identities, projected regions, physics and collision times are **training/evaluation labels only**. `code/infer.py` reads only RGB-derived tokens and a saved student checkpoint.

The implementation comes from the original relation/contact/event student, with the real optimizer, losses, source calibration and evaluation retained. It is separate from the explicit geometry route in [estimate_geometry](../estimate_geometry/README.md). No published pretrained accident detector or real-world accuracy claim is included.

## Status and acceptance boundary

- Runnable now: CPU/CUDA student training on a prepared cache; safe-load checkpoint serialization; token-only inference; a miniature software acceptance check with synthetic inputs.
- Recovered but not rerun with pretrained weights here: RGB -> frozen V-JEPA spatial/delta tokens -> dense/participant-label extraction. This requires an external V-JEPA repository/checkpoint, CUDA, reviewed CARLA episode manifests and source annotations/images.
- `code/augment_contact.py` generates the historical targets from matching CARLA annotations and static geometry. Four synthetic CARLA-schema cases and a cache-to-cache CLI check match the original source arrays exactly; this is a software consistency check, not a new CARLA run. See [consistency evidence](../docs/validation/output_json/contact_teacher_consistency.json) and [additional source manifest](SOURCE_MANIFEST_CONTACT_RECOVERY.json).
- No original trained student, V-JEPA weight file, company recordings, company adapter or third-party V-JEPA source is bundled. The tiny `.pt` under smoke evidence is trained on fabricated tokens and has no practical accident-detection value.
- The sigmoid event output has not been validated as a calibrated real-world probability. A source-selected alarm threshold/debounce is stored in the checkpoint. A simulation contact score is not an accident-recognition score.

## Quick implementation check

From the repository root, in a project environment with NumPy and PyTorch:

```text
python -B learn_relations/code/synthetic_smoke.py --output-dir learn_relations/experiments/your_new_smoke
```

Use a **new** output directory. The script generates three fabricated recordings (8 windows each), runs the actual student for two CPU epochs, saves a checkpoint, runs inference, checks finite output shapes, and reruns inference after removing every teacher field. It does not invoke CARLA or the V-JEPA encoder. See the checked-in [acceptance record](../docs/validation/output_json/relation_smoke.json); all generated inputs and logs are beside it. The historical initial and intermediate checks are retained separately and are not additional independent data.

Tested environment: Python 3.12.0, NumPy 2.4.6, PyTorch 2.7.1+cu126; the student smoke used CPU. No package installation or new environment was performed during this check. `requirements.txt` separates the student runtime from the additional OpenCV dependency for extraction. Other platforms and fresh installs are unverified.

## Prepare original-style RGB features

Obtain the official [V-JEPA repository](https://github.com/facebookresearch/vjepa2) and its dependencies separately, and download the ViT-B/16 V-JEPA 2.1 checkpoint `vjepa2_1_vitb_dist_vitG_384.pt` from the [official model distribution](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt). The architecture name and URL are listed in the [official backbone implementation](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py). Record the exact upstream commit and file SHA-256 with your experiment; the original snapshot did not record an upstream commit, so exact upstream-version reproduction remains unresolved. Do not assume the latest checkout is numerically identical to the historical dependency.

This adapter imports `app.vjepa_2_1.models.vision_transformer.vit_base`, expects the `ema_encoder` state, and checks missing/unexpected keys. It uses the recovered 384px, 16px patch, 2-frame tubelet configuration. The default clip is 8 frames, stride 4; RGB resize/letterboxing and temporal-delta feature definition are retained. Extraction currently requires CUDA/BF16 as in the source; CPU extraction has not been implemented or claimed tested.

```text
python -B learn_relations/code/extract.py path/to/reviewed_episode.json --vjepa-root path/to/vjepa2 --checkpoint path/to/vjepa2_1_vitb_dist_vitG_384.pt --output-dir learn_relations/experiments/your_capture/output_npz --clip-length 8 --stride 4
```

Manifest schema is `mask2world-carla-collision-episodes-0.2`. The training quality gate requires `training_use_policy.status=eligible`, `eligible=true`, and allowed logical iteration ranges. Manifest `source_annotation` can be relative to the manifest. Annotation carries `base`, `scene_state` and `sensor`; source RGB is located by `image_dir` (optional, resolved relative to the annotation) or the original `annotation.parents[2]/images/train` convention. CARLA frames/clock, participant IDs, bounding boxes, calibration and incident timing must come from the matching capture; do not substitute geometry replay JSON for a reviewed training episode. The existing `datasets` capture tooling and these legacy manifests have distinct schemas; an automatic equivalence adapter is not claimed.

The extractor produces tokens, temporal deltas, dense targets and participant targets. Generate the original contact teacher columns in a separate **new** cache directory:

```text
python -B learn_relations/code/augment_contact.py --cache-dir learn_relations/experiments/your_capture/output_npz --static-geometry path/to/matching_static_geometry.json --output-dir learn_relations/experiments/your_capture/input_npz/source
```

Static geometry uses schema `mask2world-carla-static-geometry-0.1`, `objects[].bbox_world` with world-space corners, location and rotation. It must match the capture map. The original exporter is `export_carla_static_geometry.py`; this recovery did not connect to a CARLA server or produce new static map data. For a dataset containing only dynamic pairs, an explicit JSON with that schema and an empty `objects` array is accepted. An empty static map does not reproduce a vehicle/static contact dataset.

The recovered target definitions are preserved rather than silently replaced:

- `contact_gap_m`: the original `oriented_box_gap_3d_m` feature, sampled at cached window ends and cast to float16. Its algorithm combines the XY convex-hull separation with the z-interval gap; this is not a general exact distance between tilted 3D boxes or a vehicle-mesh/collider contact truth. Missing pairs use the original 50m sentinel.
- `contact_speed_change_mps`: sum, over the selected two actors, of absolute speed-norm changes from the preceding recorded frame, sampled at window ends and cast to float16. It is not divided by the frame interval and is not acceleration. The first frame uses a zero-speed missing previous state, as in the source.
- Dynamic participant selection and async logical-frame indexing are retained. For `single_vehicle_static`, the old target selects the static object's nearest XY **center**, which need not select the closest surface or actual physics collider. These historical limitations must accompany interpretation of the labels.

The command also preserves `contact_pair_present` and `contact_relative_speed_mps`. These labels remain privileged training/evaluation data and are not read by deployed token-only inference. The recovered company-stage source remains local, pending ownership/licensing review; it has not been published or given a new license.

For independently reviewed alternative teacher labels, `attach_contact_labels.py` remains an optional explicit import interface. Supply a JSON of this form (one entry per cached end iteration):

```json
{
  "schema_version": "reviewed-contact-teacher-1",
  "provenance": "capture ID, annotation hash, reviewed teacher source/version",
  "target_definitions": {"gap_m": "precise geometry convention", "speed_change_mps": "precise actor/time interval convention"},
  "by_end_iteration": {"8": {"gap_m": 0.4, "speed_change_mps": 2.1}}
}
```

Those numbers illustrate the alternative interface only; they are not labels for any bundled CARLA frame and are not a replacement for the historical generation definitions above. Join a complete reviewed file:

```text
python -B learn_relations/code/attach_contact_labels.py --cache path/to/raw_tokens.npz --labels path/to/reviewed_contact.json --output learn_relations/experiments/your_capture/input_npz/source/recording.npz
```

## Train and infer

Place eligible source recordings and independent validation recordings in separate cache directories. Source cameras below `--calibration-height` train; source cameras exactly at that height calibrate threshold/debounce. Default 3m preserves the original controlled split. Validation is never used to select the alarm policy; manifest paths shared with source are rejected. This split is a historical controlled protocol, not proof of unseen-scene or real-world generalization.

```text
python -B learn_relations/code/train.py --source-cache-dir path/to/source_cache --validation-cache-dir path/to/heldout_cache --output learn_relations/experiments/your_training/output_json/metrics.json --checkpoint-dir learn_relations/experiments/your_training/output_pt --epochs 150 --seeds 20260907 --device cuda --calibration-height 3
python -B learn_relations/code/infer.py --checkpoint path/to/relation_contact_event_seed_20260907.pt --tokens path/to/rgb_tokens.npz --output learn_relations/experiments/your_inference/output_npz/predictions.npz --device cpu
```

`train.py` supports `--device cpu`; historical full-scale timings/performance have not been reproduced. `--feature-mode fused` concatenates last-frame and temporal-delta spatial tokens; `last` and `temporal_delta` are also supported. Inference uses the saved feature mode. The input can contain only `tokens` and, when needed, `temporal_delta_tokens`; no manifest/annotation/ground truth is read during inference. Output NPZ fields are `contact_gap_and_speed_change` `[N,2]` (m, m/s), `participant_mask_probability` `[N,2,P]`, `event_probability` `[N]`, and debounced `event_alarm` `[N]`. Windows must be chronologically ordered within a single recording; invoke inference separately per recording so debounce never crosses recording boundaries.

Prepared training NPZ additionally requires `manifest`, `grid_size`, `presence`, `depth_m`, `speed_mps`, `ends`, `contact_gap_m`, `contact_speed_change_mps`, `relation_participant_masks`, `relation_participant_valid`, `relation_static_second`. Cache `manifest` references may be relative to the cache. Contact/dense fields are target arrays, not input features. Empty near-contact/impact metric subsets are reported as `null`, not silently treated as a zero error.

All output files use exclusive creation or reject existing output directories; original inputs are never edited. [Source provenance](SOURCE_MANIFEST.json) and [target file map/hashes](SOURCE_MANIFEST_TARGETS.json) document adaptations. See the repository's [third-party and data notes](../docs/RELEASE.md). The official [V-JEPA README license section](https://github.com/facebookresearch/vjepa2#license) describes its source licenses; this is not a blanket relicensing of downloaded weights.
