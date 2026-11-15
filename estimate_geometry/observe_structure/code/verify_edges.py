"""Verify newly generated RGB photometric observations against the frozen input."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    actual=json.loads(args.observed.read_text(encoding='utf-8'))
    frozen=json.loads((ROOT.parent/'refine_contours/input_json/observation.json').read_text(encoding='utf-8'))
    report=dict(query_truth_read=False,all_fields_and_numeric_records_exact=actual==frozen,
        points=len(actual['records']),photometric_candidates=actual['photometric_candidates'])
    target=args.output.resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Output outside observer purpose')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if actual!=frozen:raise SystemExit(2)
if __name__=='__main__':main()
