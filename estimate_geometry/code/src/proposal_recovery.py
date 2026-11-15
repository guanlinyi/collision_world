"""Causal recovery when observations exist but an upstream proposal is absent.

Uses the latest earlier fitted pose of the SAME provided object identity as an
initial pose. Current pixel evidence is still required. Never reads reference
geometry; no future interpolation or invented measurement is supplied.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .common import read_json,grouped_records,is_clipped
from .geometry import pixel_sigma,corners


def recover_missing_geometry(bundle,surface,raw,model_dir,max_step=None):
    cfg=read_json(Path(model_dir)/'model.json')
    noise=read_json(Path(model_dir)/'view_noise.json')['normalized_sigma']
    groups=grouped_records(bundle);cam=bundle['cameras'];obs=bundle['observations'];rows=bundle['states']
    lookup={int(s):i for i,s in enumerate(raw['source_indices'])}
    history={};output=[];trace=[]
    order=sorted(range(len(rows)),key=lambda s:(rows[s]['episode'],rows[s]['step'],s))
    for source in order:
        row=rows[source];step=int(row['step']);key=row['object_key']
        if max_step is not None and step>max_step:continue
        if source in lookup:
            i=lookup[source];center=raw['center'][i].copy();rotation=raw['rotation'][i].copy();dims=raw['dims'][i].copy()
            output.append((source,center,rotation,dims));history[key]=(step,center,rotation,dims)
            continue
        if key not in history:
            trace.append(dict(source=source,status='unknown',reason='No earlier fitted pose for this identity'));continue
        previous_step,origin,initial,dims=history[key]
        if previous_step>=step:raise ValueError('Earlier pose must be strictly earlier')
        views=[]
        for record in groups.get(source,[]):
            j,v=record['index'],record['view'];ids=np.flatnonzero(obs['mask'][j]&(surface['confidence'][j]>.6))
            if len(ids)<24 or not cam['valid'][source,v]:continue
            ids=ids[np.linspace(0,len(ids)-1,min(64,len(ids))).astype(int)]
            points=surface['coords'][j].reshape(-1,3)[ids].astype(float)*dims
            uv=obs['pixels'][j].reshape(-1,2)[ids].astype(float)+cfg.get('pixel_center_offset',0.)
            q,t,k=cam['q'][source,v],cam['t'][source,v],cam['k'][source,v]
            box=record['bbox'];base=max(.5,(1+.03*min(box[2]-box[0],box[3]-box[1]))*noise['clipped' if is_clipped(box) else 'complete'])
            std=surface['sigma'][j].reshape(-1,3)[ids].astype(float)
            sig=pixel_sigma(points,origin,initial,dims,q,t,k,base,std)
            views.append((points,uv,q,t,k,sig))
        if not views:
            trace.append(dict(source=source,status='unknown',reason='No usable current pixel evidence',previous_step=previous_step));continue
        def orient(x):return Rotation.from_rotvec(x[3:]).as_matrix()@initial
        def residual(x):
            values=[]
            for points,uv,q,t,k,sig in views:
                cc=(points@orient(x).T+x[:3]-t)@q
                pixel=cc[:,:2]/cc[:,2,None].clip(.5)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
                values.extend(((pixel-uv)/sig).ravel())
            return np.r_[values,(x[:3]-origin)*.2,x[3:]*.2/np.deg2rad(30)]
        fit=least_squares(residual,np.r_[origin,np.zeros(3)],
            bounds=(np.r_[origin-12,np.full(3,-np.pi)],np.r_[origin+12,np.full(3,np.pi)]),loss='soft_l1',max_nfev=80)
        if not fit.success or not np.isfinite(fit.x).all():
            trace.append(dict(source=source,status='unknown',reason='Geometry optimization did not converge',previous_step=previous_step));continue
        center=fit.x[:3];rotation=orient(fit.x)
        output.append((source,center,rotation,dims));history[key]=(step,center,rotation,dims)
        trace.append(dict(source=source,status='recovered',previous_step=previous_step,views=len(views),
            sampled_points=sum(len(v[0]) for v in views),solver_cost=float(fit.cost),solver_optimality=float(fit.optimality),
            initialization='previous causal fitted pose, not a new measurement',current_evidence_required=True))
    output.sort(key=lambda item:item[0])
    geometry=dict(source_indices=np.array([x[0] for x in output],dtype=raw['source_indices'].dtype),
        center=np.array([x[1] for x in output]),rotation=np.array([x[2] for x in output]),dims=np.array([x[3] for x in output]))
    geometry['corners']=corners(geometry['center'],geometry['rotation'],geometry['dims'])
    return geometry,trace
