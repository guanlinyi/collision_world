"""Export a small prepared CARLA subset; never alter the original capture."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    source=args.source.resolve();out=args.output.resolve()
    project=Path(__file__).resolve().parents[3]
    if not out.is_relative_to(project/'datasets/carla_surface_sample'):raise ValueError('Output outside sample purpose')
    out.mkdir(parents=True,exist_ok=False)
    (out/'input_json').mkdir();(out/'input_npz').mkdir()
    def load(name):return json.loads((source/name).read_text(encoding='utf-8'))
    states=load('states.json');records=load('records.json')
    steps=sorted({s['step'] for s in states if s['episode']==0})[:3]
    selected=[i for i,s in enumerate(states) if s['episode']==0 and s['step'] in steps]
    remap={old:new for new,old in enumerate(selected)}
    rows=[i for i,r in enumerate(records) if r['source_state'] in remap]
    newstates=[];newrecords=[]
    for i in selected:
        state=dict(states[i]);state['index']=remap[i];state['images']=[None]*6
        newstates.append(state)
    for i in rows:
        row=dict(records[i]);row['source_state']=remap[row['source_state']];row['index']=len(newrecords)
        newrecords.append(row)
    files=[]
    def record(name,target):
        files.append(dict(source_file=name,source_sha256=hashlib.sha256((source/name).read_bytes()).hexdigest(),
             target=target.relative_to(out).as_posix(),target_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    for name,indices in [('cameras.npz',selected),('reference_geometry.npz',selected),
                         ('inputs.npz',rows),('training_targets.npz',rows)]:
        with np.load(source/name,allow_pickle=False) as z:
            np.savez_compressed(out/'input_npz'/name,**{k:z[k][indices] for k in z.files})
        record(name,out/'input_npz'/name)
    with np.load(source/'proposals.npz',allow_pickle=False) as z:
        ids=[i for i,s in enumerate(z['source_indices']) if int(s) in remap]
        assert len(ids)==len(selected),'All sample states must have saved upstream proposals'
        arrays={k:z[k][ids] for k in z.files if k!='source_indices'}
        arrays['source_indices']=np.array([remap[int(z['source_indices'][i])] for i in ids])
        np.savez_compressed(out/'input_npz/proposals.npz',**arrays)
    record('proposals.npz',out/'input_npz/proposals.npz')
    contract=load('contract.json');contract.update(raw_demo_image_episodes=[],subset_only=True,
        subset_source_state_indices=selected,subset_record_indices=rows,
        sample_scope='Real CARLA crop/native-mask/calibration/reference subset, existing learned proposals. No new capture, training replication or naked-RGB inference claim.')
    for name,value in [('states.json',newstates),('records.json',newrecords),('contract.json',contract)]:
        target=out/'input_json'/name
        with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
        record(name,target)
    with (out/'input_json/SOURCE_SELECTION.json').open('x',encoding='utf-8') as f:
        json.dump(dict(states=len(selected),records=len(rows),files=files),f,indent=2)
    print(json.dumps(dict(states=len(selected),records=len(rows),output=str(out))))


if __name__=='__main__':main()
