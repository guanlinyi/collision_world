"""File contracts and shared RGB feature construction. No ground truth opened here."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def resource(path):
    """Support original flat bundles and extension-separated portable bundles."""
    path=Path(path)
    if path.is_file():return path
    for prefix in ['input_', 'output_']:
        candidate=path.parent/(prefix+path.suffix.lstrip('.'))/path.name
        if candidate.is_file():return candidate
    return path

def read_json(path):return json.loads(resource(path).read_text(encoding='utf-8'))
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2))
def read_npz(path):
    with np.load(resource(path),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def fresh(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Output must stay inside this lab folder')
    path.mkdir(parents=True,exist_ok=False)
    return path
def load_inputs(folder,include_observations=True,include_proposals=True):
    folder=Path(folder)
    required=['cameras.npz','records.json','states.json','contract.json']
    if include_observations:required.append('inputs.npz')
    if include_proposals:required.append('proposals.npz')
    missing=[n for n in required if not resource(folder/n).is_file()]
    if missing:raise ValueError('Prepared calibrated input is required; missing: '+', '.join(missing)+'. A bare video is not supported by the geometry predictor.')
    return dict(folder=folder,observations=read_npz(folder/'inputs.npz') if include_observations else {},cameras=read_npz(folder/'cameras.npz'),records=read_json(folder/'records.json'),states=read_json(folder/'states.json'),proposals=read_npz(folder/'proposals.npz') if include_proposals else {},contract=read_json(folder/'contract.json'))
def features(bundle,indices,pixel_offset=0.):
    o=bundle['observations'];camera=bundle['cameras'];arrays=[]
    for i in indices:
        r=bundle['records'][i];k=camera['k'][r['source_state'],r['view']];box=r['bbox']
        ray=(o['pixels'][i].astype(float)+pixel_offset-[k[0,2],k[1,2]])/[k[0,0],k[1,1]]
        wh=np.broadcast_to(np.log(np.maximum([box[2]-box[0],box[3]-box[1]],1)/64),(64,64,2))
        arrays.append(np.concatenate([o['rgb'][i].astype(np.float32)/255*o['mask'][i,:,:,None],o['mask'][i,:,:,None],ray,wh],-1))
    return np.array(arrays,dtype=np.float32).transpose(0,3,1,2)
def grouped_records(bundle):
    groups={}
    for r in bundle['records']:groups.setdefault(r['source_state'],[]).append(r)
    return groups
def is_clipped(box):return box[0]<=1 or box[1]<=1 or box[2]>=639 or box[3]>=359
