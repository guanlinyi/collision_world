"""Causal pose/velocity update. GT is used only by the separate calibrate function."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .common import read_json,write_json,grouped_records,is_clipped
from .geometry import upright,corners
from .motion_innovation import innovation_covariance

BASE=upright(0.)
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def matrix(x):return Rotation.from_euler('ZYX',x).as_matrix()@BASE
def angles(m):return Rotation.from_matrix(m@BASE.T).as_euler('ZYX')

def calibrate(bundle,raw,truth,model_dir):
    ix=raw['source_indices'];rows=[bundle['states'][i] for i in ix]
    z=np.c_[raw['center'],angles(raw['rotation'])];gt=np.c_[truth['center'][ix],angles(truth['rotation'][ix])]
    train=np.array([r['episode']<4 for r in rows]);error=z-gt;error[:,3:]=wrap(error[:,3:])
    lookup={(r['object_key'],r['step']):i for i,r in enumerate(rows)};acc=[]
    for i,r in enumerate(rows):
        a=lookup.get((r['object_key'],r['step']-1));b=lookup.get((r['object_key'],r['step']-2))
        if train[i] and a is not None and b is not None:
            d=gt[i]-2*gt[a]+gt[b];d[3:]=wrap(wrap(gt[i,3:]-gt[a,3:])-wrap(gt[a,3:]-gt[b,3:]));acc.append(d)
    write_json(Path(model_dir)/'state_noise.json',dict(train_episodes=[0,1,2,3],measurement=np.maximum(np.mean(error[train]**2,0),[.0001]*3+[np.deg2rad(.2)**2]*3).tolist(),process=np.maximum(np.mean(np.array(acc)**2,0),[.0001]*3+[np.deg2rad(.1)**2]*3).tolist()))

def filter_states(bundle,surface,raw,model_dir,limit=None,state_update='released'):
    if state_update not in ('released','directional'):raise ValueError('Unknown state update rule')
    cfg=read_json(Path(model_dir)/'state_noise.json');noise=read_json(Path(model_dir)/'view_noise.json')['normalized_sigma']
    ix=raw['source_indices'];rows=[bundle['states'][i] for i in ix];z=np.c_[raw['center'],angles(raw['rotation'])]
    groups=grouped_records(bundle);o=bundle['observations'];camera=bundle['cameras']
    F=np.eye(12);F[:6,6:12]=np.eye(6);G=np.r_[.5*np.eye(6),np.eye(6)]
    Q=G@np.diag(cfg['process'])@G.T;H=np.eye(12)[:6];history={};output=[];selected=[]
    for i,source in enumerate(ix):
        row=rows[i]
        if limit is not None and row['step']>=limit:continue
        info=np.zeros((6,6))
        for r in groups.get(int(source),[]):
            if not r.get('geometry_allowed',True):continue
            k=r['index'];v=r['view'];ids=np.flatnonzero(o['mask'][k]&(surface['confidence'][k]>.6))
            if len(ids)<24:continue
            _,unique=np.unique(o['pixels'][k].reshape(-1,2)[ids],axis=0,return_index=True)
            ids=ids[unique];ids=ids[np.linspace(0,len(ids)-1,min(64,len(ids))).astype(int)]
            pts=surface['coords'][k].reshape(-1,3)[ids]*raw['dims'][i]
            q=camera['q'][source,v];t=camera['t'][source,v];K=camera['k'][source,v];box=r['bbox']
            sigma=max(.5,(1+.03*min(box[2]-box[0],box[3]-box[1]))*noise['clipped' if is_clipped(box) else 'complete'])
            def project(x):
                cc=(pts@matrix(x[3:]).T+x[:3]-t)@q
                return (cc[:,:2]/cc[:,2,None].clip(.5)*[K[0,0],K[1,1]]).ravel()
            jac=[]
            for axis in range(6):
                delta=np.zeros(6);delta[axis]=1e-4;jac.append((project(z[i]+delta)-project(z[i]-delta))/2e-4)
            A=np.stack(jac,1)/sigma;info+=A.T@A/len(ids)
        R=np.linalg.inv(info+np.eye(6)*1e-6)+np.diag(cfg['measurement']);prev=history.get(row['object_key'])
        if prev is None or row['step']!=prev[2]+1:
            state=np.r_[z[i],np.zeros(6)];P=np.eye(12);P[:6,:6]=R
        else:
            state=F@prev[0];P=F@prev[1]@F.T+Q;delta=z[i]-state[:6];delta[3:]=wrap(delta[3:]);S=P[:6,:6]+R
            if delta@np.linalg.solve(S,delta)>24:
                P+=G@innovation_covariance(delta,S,rule=state_update)@G.T;S=P[:6,:6]+R
            gain=np.linalg.solve(S,P[:6,:]).T;state+=gain@delta
            A=np.eye(12)-gain@H;P=A@P@A.T+gain@R@gain.T
        history[row['object_key']]=(state,P,row['step']);selected.append(i);output.append(state[:6].copy())
    output=np.array(output);rot=matrix(output[:,3:]);dims=raw['dims'][selected];center=output[:,:3]
    return dict(source_indices=ix[selected],center=center,rotation=rot,dims=dims,corners=corners(center,rot,dims))
