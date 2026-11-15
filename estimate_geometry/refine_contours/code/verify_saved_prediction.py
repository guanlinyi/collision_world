"""Read-only comparison of a saved upright prediction, without rerunning a fit."""
import argparse,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    saved=args.run/'output_json/fit/joint_fresh_07_c0_f28_fit_before_scoring.json'
    r=json.loads(saved.read_text(encoding='utf-8'));f=json.loads((ROOT/'evaluation_only_json/frozen_candidate.json').read_text(encoding='utf-8'))
    assert r['candidate_pose'][3:5]==f['candidate_pose'][3:5]==[0.,0.]
    report=dict(prediction_already_persisted=True,fit_rerun=False,query_reference_truth_read=False,
        pose_convention='LAB xyz metres; degree rotation vector constrained to [0,0,rz]',
        center_difference_m=math.dist(r['candidate_pose'][:3],f['candidate_pose'][:3]),
        rotation_difference_deg=abs((r['candidate_pose'][5]-f['candidate_pose'][5]+180)%360-180),
        cost_difference=abs(r['candidate_cost']-f['candidate_cost']),same_status=r['status']==f['status'],
        same_accepted_rounds=r['accepted_outer_rounds']==f['accepted_outer_rounds'])
    report['passed']=(report['center_difference_m']<1e-8 and report['rotation_difference_deg']<1e-7 and report['cost_difference']<1e-8 and report['same_status'] and report['same_accepted_rounds'])
    target=args.output.resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Output outside purpose')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as s:json.dump(report,s,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
