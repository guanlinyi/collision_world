"""Recompute the frozen case07/frame28 R/BT8 contour candidate from observations.

SAM/photometric extraction is cached upstream. SOURCE geometry, anchor selection,
correspondence refresh and all inner solves run here. Frozen poses are opened
only after the new prediction has been persisted, for regression checking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import run_candidate as engine
import prototype_frame27
import refresh_associations
from geometry_pool import GeometryPool,read,save_json
from portable_geometry import legacy
from problem import GeometryOnlyProblem

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-frozen',action='store_true')
    parser.add_argument('--observation',type=Path,default=ROOT/'input_json/observation.json',help='Explicit photometric finite-edge JSON, e.g. from observe_structure')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay inside refine_contours')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    safe=read(ROOT/'input_json/safe_inputs.json')
    support=read(ROOT/'input_json/support.json')
    assert safe['query_truth_read'] is False
    problem=GeometryOnlyProblem(safe,dict(pose=safe['baseline_pose'],cost=safe['baseline_cost']),support,legacy())
    engine.ATTEMPT=prototype_frame27.ATTEMPT='fit'
    prototype_frame27.REQUIRE_FULL_POOL=refresh_associations.REQUIRE_FULL_POOL=False
    builder=GeometryPool(out_root=out,attempt='fit');original_build=builder.build
    def build(pose,row,label):
        pool=original_build(pose,row,label);pool.setdefault('_artifact_label',label);return pool
    builder.build=build
    observation=args.observation.resolve()
    if not observation.is_relative_to(PROJECT):raise ValueError('Observation must stay inside this project')
    context=dict(sample_id=safe['key'],problem=problem,observation_path=observation,
        rgb_record=dict(image=str(PROJECT/safe['image']),crop_xyxy=[639,394,851,536]),
        source_paths=[Path(__file__),ROOT/'input_json/safe_inputs.json',ROOT/'input_json/support.json',
                      observation,ROOT/'SOURCE_TRANSFORMATION.json'])
    record=engine.fit_frame(safe['frame'],builder,problem,out_root=out,objective_mode='exterior_only',
        context=context,source_foot_allowed=None,contract_path=ROOT/'input_json/contract.json',
        acceptance_policy='true_refreshed_backtracking')
    report=dict(prediction_saved_before_comparison=True,reference_truth_read=False,
        source_geometry_rebuilt=True,SAM_and_RGB_structure_extraction_cached=True,
        elapsed_seconds=time.monotonic()-started,status=record['status'],
        accepted_outer_rounds=record['accepted_outer_rounds'],candidate_pose=record['candidate_pose'],
        candidate_cost=record['candidate_cost'],python=sys.version)
    if args.verify_frozen:
        frozen=read(ROOT/'evaluation_only_json/frozen_candidate.json')
        difference=np.asarray(record['candidate_pose'])-frozen['candidate_pose']
        assert record['candidate_pose'][3:5]==frozen['candidate_pose'][3:5]==[0.,0.], 'This replay supports upright z rotation only'
        angle=abs((record['candidate_pose'][5]-frozen['candidate_pose'][5]+180)%360-180)
        report.update(frozen_candidate_sha256=hashlib.sha256((ROOT/'evaluation_only_json/frozen_candidate.json').read_bytes()).hexdigest(),
            center_difference_m=float(np.linalg.norm(difference[:3])),rotation_difference_deg=angle,
            cost_difference=abs(record['candidate_cost']-frozen['candidate_cost']),
            same_status=record['status']==frozen['status'],
            same_accepted_rounds=record['accepted_outer_rounds']==frozen['accepted_outer_rounds'])
        # This checks migration repeatability, not accuracy: 1e-8m and 1e-7deg
        # allow final floating-point serialization/order noise far below pixels.
        report['passed']=(report['same_status'] and report['same_accepted_rounds'] and
            report['center_difference_m']<1e-8 and report['rotation_difference_deg']<1e-7 and report['cost_difference']<1e-8)
    save_json(out/'output_json/acceptance.json',report)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    if args.verify_frozen and not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
