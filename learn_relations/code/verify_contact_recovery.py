"""Compare recovered contact-label functions with selected original source functions on synthetic CARLA-schema inputs.
No CARLA server, RGB encoder, GPU training or original dataset is run.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from contact_physics import recording_features
from augment_contact import values_at_ends


def original_namespace(helper, augmenter):
    names={"load_json","load_eligible_manifest","load_static_geometry","vec","norm","angle_delta","object_state","choose_actor_ids","aabb_gap","_convex_hull_xy","_point_segment_distance","_polygons_overlap","oriented_box_gap","frame_features","recording_features"}
    tree=ast.parse(helper.read_text(encoding="utf-8"))
    namespace={"np":np,"json":json,"gzip":gzip,"math":math,"Path":Path,"Any":object,"Counter":Counter}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {"FEATURE_NAMES","MANIFEST_SCHEMA"}:
                    namespace[target.id]=ast.literal_eval(node.value)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),"exec"),namespace)
    namespace["FEATURE_INDEX"]={name:i for i,name in enumerate(namespace["FEATURE_NAMES"])}
    tree=ast.parse(augmenter.read_text(encoding="utf-8"))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in {"values_at_ends","augment_record"}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(augmenter),"exec"),namespace)
    return namespace


def box(center,speed,actor_id,extent=(1.,.5,.5)):
    corners=[[center[0]+sx*extent[0],center[1]+sy*extent[1],center[2]+sz*extent[2]] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)]
    return {"id":actor_id,"type_id":"vehicle.synthetic","actor_transform":{"location":dict(zip(("x","y","z"),center)),"rotation":{"yaw":0.,"pitch":0.,"roll":0.}},
        "bbox_3d_world":corners,"velocity_world_mps":{"x":speed,"y":0.,"z":0.}}


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:json.dump(data,handle,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-helper",type=Path,required=True)
    parser.add_argument("--legacy-augmenter",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    out=args.output_dir.resolve();out.mkdir(parents=True)
    original=original_namespace(args.legacy_helper,args.legacy_augmenter)
    base={"schema_version":"mask2world-carla-collision-episodes-0.2","collision_mechanism":"rear_end","incidents":[{"actor_ids":[1,2]}],
        "training_use_policy":{"status":"eligible","eligible":True,"allowed_logical_iteration_ranges":[[1,100]]}}
    scenes=[{"iteration":10*(i+1),"simulation_time_seconds":i*.1,"objects":[box((0.,0.,.5),[1.,1.3,.2][i],1),box(([5.,3.,1.5][i],0.,.5),[.5,.7,.1][i],2)]} for i in range(3)]
    annotation={"scene_state":scenes}
    dynamic_ends=np.asarray([30,10])
    static_object={"semantic_label":"Wall","bbox_world":{"location":{"x":4.,"y":0.,"z":.5},"rotation":{},"corners":[dict(zip(("x","y","z"),p)) for p in box((4.,0.,.5),0.,99)["bbox_3d_world"]]}}
    cases=[("dynamic_reordered_ends",base,annotation,[],dynamic_ends),
           ("async_logical_clock",{**base,"capture_mode":"async_rgb_only"},annotation,[],np.asarray([3,1])),
           ("single_vehicle_static",{**base,"collision_mechanism":"single_vehicle_static","incidents":[{"actor_ids":[1]}]},annotation,[static_object],dynamic_ends),
           ("participant_missing",base,{"scene_state":[{**scene,"objects":scene["objects"][:1]} for scene in scenes]},[],dynamic_ends)]
    checks=[]
    for name,manifest,ann,static,ends in cases:
        expected,iterations,ids=original["recording_features"](manifest,ann,static)
        actual,new_iterations,new_ids=recording_features(manifest,ann,static)
        np.testing.assert_array_equal(actual,expected)
        assert iterations==new_iterations and ids==new_ids
        expected_labels=original["values_at_ends"](expected,iterations,ends)
        actual_labels=values_at_ends(actual,new_iterations,ends)
        for field in expected_labels:np.testing.assert_array_equal(expected_labels[field],actual_labels[field])
        checks.append({"case":name,"maximum_feature_difference":float(np.max(np.abs(actual-expected))),"all_four_target_arrays_exact":True,
            "gap_m":actual_labels["contact_gap_m"].astype(float).tolist(),"speed_change_mps":actual_labels["contact_speed_change_mps"].astype(float).tolist()})
        write_json(out/"input_json"/(name+".json"),{"manifest":manifest,"annotation":ann,"static_geometry":static,"ends":ends.tolist()})
    try:values_at_ends(expected,iterations,np.asarray([999]))
    except ValueError:missing_rejected=True
    else:raise AssertionError("missing end iteration accepted")
    annotation_path=out/"input_json/annotation.json";write_json(annotation_path,annotation)
    legacy_manifest=out/"input_json/legacy_manifest.json";write_json(legacy_manifest,{**base,"source_annotation":str(annotation_path)})
    portable_manifest=out/"input_json/portable_manifest.json";write_json(portable_manifest,{**base,"source_annotation":"annotation.json"})
    legacy_cache=out/"input_npz/legacy.npz";legacy_cache.parent.mkdir()
    with legacy_cache.open("xb") as handle:np.savez_compressed(handle,manifest=np.asarray(str(legacy_manifest)),ends=dynamic_ends)
    reference=original["augment_record"](legacy_cache,[])
    portable_cache=out/"input_npz/portable/raw.npz";portable_cache.parent.mkdir()
    import os
    with portable_cache.open("xb") as handle:np.savez_compressed(handle,manifest=np.asarray(os.path.relpath(portable_manifest,portable_cache.parent)),ends=dynamic_ends)
    static_file=out/"input_json/static_geometry.json";write_json(static_file,{"schema_version":"mask2world-carla-static-geometry-0.1","objects":[]})
    command=[sys.executable,"-B",str(Path(__file__).with_name("augment_contact.py")),"--cache-dir",str(portable_cache.parent),"--static-geometry",str(static_file),"--output-dir",str(out/"output_npz")]
    result=subprocess.run(command,text=True,capture_output=True)
    (out/"output_txt").mkdir()
    with (out/"output_txt/augmentation.txt").open("x",encoding="utf-8") as handle:handle.write(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    with np.load(out/"output_npz/raw.npz",allow_pickle=False) as migrated:
        for field in ("contact_pair_present","contact_gap_m","contact_relative_speed_mps","contact_speed_change_mps"):
            np.testing.assert_array_equal(migrated[field],reference[field])
        assert (out/"output_npz"/str(migrated["manifest"])).is_file()
    write_json(out/"output_json/acceptance.json",{"status":"passed","scope":"synthetic CARLA-schema input; original source functions vs recovered modules, no actual CARLA capture, encoder or training", "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.legacy_helper,args.legacy_augmenter)},
        "cases":checks,"missing_end_rejected":missing_rejected,"cli_command":command,"cli_exit_code":result.returncode,"legacy_absolute_vs_portable_relative_all_targets_exact":True})
    print(json.dumps({"status":"passed","cases":len(checks),"cli_exit_code":result.returncode,"target_arrays_exact":True}))
    return 0

if __name__ == "__main__":raise SystemExit(main())
