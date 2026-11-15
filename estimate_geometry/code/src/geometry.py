"""One rigid vehicle pose fitted to image evidence across available views."""
import itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .common import read_json,grouped_records,is_clipped

def upright(yaw):
    front=np.array([np.cos(yaw),np.sin(yaw),0.]);down=np.array([0.,0.,-1.])
    return np.stack([np.cross(down,front),down,front],1)
def corners(center,rotation,dims):
    signs=np.array(list(itertools.product([-1.,1.],repeat=3)))
    return center[:,None]+np.einsum('nkj,nij->nki',signs[None]*dims[:,None]/2,rotation)
def pixel_sigma(points,center,rotation,dims,q,t,k,base_sigma,std):
    cam=(points@rotation.T+center-t)@q;z=cam[:,2].clip(.5)
    jac=np.zeros((len(points),2,3));jac[:,0,0]=k[0,0]/z;jac[:,1,1]=k[1,1]/z
    jac[:,0,2]=-k[0,0]*cam[:,0]/z**2;jac[:,1,2]=-k[1,1]*cam[:,1]/z**2
    local=jac@(q.T@rotation@np.diag(dims))
    return np.sqrt(base_sigma**2+np.sum(local**2*std[:,None]**2,axis=-1))

def fit_geometry(bundle,surface,model_dir,dimensions_by_track=None):
    model_dir=Path(model_dir);cfg=read_json(model_dir/'model.json');assets=read_json(model_dir/'assets.json')
    noise=read_json(model_dir/'view_noise.json')['normalized_sigma'];groups=grouped_records(bundle)
    o=bundle['observations'];cam=bundle['cameras'];seed=bundle['proposals'];rows=bundle['states']
    center=seed['center'].astype(float).copy();rotation=seed['rotation'].astype(float).copy();dims=seed['dims'].copy()
    chosen={};first={};coef=np.array(assets['classifier_coef'])[0];intercept=assets['classifier_intercept'][0]
    for source in seed['source_indices']:
        if dimensions_by_track is not None:continue
        key=rows[source]['object_key']
        if key in chosen:continue
        logits=[]
        for r in groups.get(int(source),[]):
            i=r['index'];rgb=o['rgb'][i].astype(np.float32)/255*o['mask'][i,...,None]
            small=rgb.reshape(16,4,16,4,3).mean((1,3));mask=o['mask'][i].reshape(16,4,16,4).mean((1,3))
            logits.append(float(coef@np.r_[small.ravel(),mask.ravel()]+intercept))
        if logits:chosen[key]=int(np.mean(logits)>0);first[key]=rows[source]['step']
    for i,source in enumerate(seed['source_indices']):
        key=rows[source]['object_key']
        if dimensions_by_track is not None:dims[i]=dimensions_by_track[key]
        elif key in chosen and rows[source]['step']>=first[key]:dims[i]=assets['assets'][chosen[key]]['dimensions_whl_m']
        origin=center[i].copy();yaw=np.arctan2(rotation[i,1,2],rotation[i,0,2]);initial=upright(yaw);views=[]
        for r in groups.get(int(source),[]):
            if not r.get('geometry_allowed',True):continue
            j=r['index'];v=r['view'];ids=np.flatnonzero(o['mask'][j]&(surface['confidence'][j]>.6))
            if len(ids)<24 or not cam['valid'][source,v]:continue
            take=np.linspace(0,len(ids)-1,min(64,len(ids))).astype(int);ids=ids[take]
            xyz=surface['coords'][j].reshape(-1,3)[ids].astype(float)
            uv=o['pixels'][j].reshape(-1,2)[ids].astype(float)+cfg.get('pixel_center_offset',0.)
            box=r['bbox'];base=max(.5,(1+.03*min(box[2]-box[0],box[3]-box[1]))*noise['clipped' if is_clipped(box) else 'complete'])
            q=cam['q'][source,v];t=cam['t'][source,v];k=cam['k'][source,v]
            std=surface['sigma'][j].reshape(-1,3)[ids].astype(float)
            sig=pixel_sigma(xyz*dims[i],origin,initial,dims[i],q,t,k,base,std)
            views.append((xyz*dims[i],uv,q,t,k,sig))
        def orient(x):return Rotation.from_rotvec(x[3:6]).as_matrix()@initial
        def residual(x):
            values=[]
            for points,uv,q,t,k,sig in views:
                cc=(points@orient(x).T+x[:3]-t)@q
                pixels=cc[:,:2]/cc[:,2,None].clip(.5)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
                values.extend(((pixels-uv)/sig).ravel())
            return np.r_[values,(x[:3]-origin)*.2,x[3:6]*.2/np.deg2rad(30)]
        if views:
            x=np.r_[origin,np.zeros(3)]
            fit=least_squares(residual,x,bounds=(np.r_[origin-12,np.full(3,-np.pi)],np.r_[origin+12,np.full(3,np.pi)]),loss='soft_l1',max_nfev=80)
            center[i]=fit.x[:3];rotation[i]=orient(fit.x)
    return dict(source_indices=seed['source_indices'],center=center,rotation=rotation,dims=dims,corners=corners(center,rotation,dims))
