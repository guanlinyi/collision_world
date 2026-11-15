"""Experimental GT-free seeds from learned object points and current pixels.

This replaces the missing upstream cached proposal producer on fresh captures.
It is a new initializer, not an exact reproduction of the legacy proposal net.
"""
import cv2
import numpy as np
from .common import grouped_records
from .geometry import upright


def bootstrap(bundle,surface,assets,limit=None):
    groups=grouped_records(bundle);obs=bundle['observations'];camera=bundle['cameras']
    coef=np.asarray(assets['classifier_coef'])[0];intercept=assets['classifier_intercept'][0]
    chosen={};source_indices=[];centers=[];rotations=[];dims=[];details=[]
    for source,st in enumerate(bundle['states']):
        if limit is not None and st['step']>=limit:continue
        records=groups.get(source,[]);key=st['object_key']
        logits=[]
        for r in records:
            j=r['index'];rgb=obs['rgb'][j].astype(np.float32)/255*obs['mask'][j,...,None]
            small=rgb.reshape(16,4,16,4,3).mean((1,3));mask=obs['mask'][j].reshape(16,4,16,4).mean((1,3))
            logits.append(float(coef@np.r_[small.ravel(),mask.ravel()]+intercept))
        if not logits:
            details.append(dict(source_index=source,status='unknown_no_mask'))
            continue
        asset_index=chosen.get(key,int(np.mean(logits)>0))
        dimension=np.asarray(assets['assets'][asset_index]['dimensions_whl_m'],float)
        candidates=[]
        for r in records:
            j,v=r['index'],r['view']
            ids=np.flatnonzero(obs['mask'][j]&(surface['confidence'][j]>.6))
            if len(ids)<12:continue
            uv=obs['pixels'][j].reshape(-1,2)[ids].astype(float)
            _,unique=np.unique(uv,axis=0,return_index=True);ids=ids[unique]
            if len(ids)<12:continue
            ids=ids[np.linspace(0,len(ids)-1,min(128,len(ids))).astype(int)]
            uv=np.ascontiguousarray(obs['pixels'][j].reshape(-1,2)[ids],dtype=np.float64)
            local=np.ascontiguousarray(surface['coords'][j].reshape(-1,3)[ids].astype(float)*dimension)
            K=np.ascontiguousarray(camera['k'][source,v],dtype=np.float64)
            cv2.setRNGSeed(source*16+v)
            ok,rvec,tvec,inliers=cv2.solvePnPRansac(local,uv,K,None,iterationsCount=100,
                reprojectionError=4.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers)<8:continue
            R,_=cv2.Rodrigues(rvec);translation=tvec.reshape(3)
            projected,_=cv2.projectPoints(local,rvec,tvec,K,None)
            residual=np.linalg.norm(projected.reshape(-1,2)-uv,axis=1)
            if not np.isfinite(translation).all() or not .5<translation[2]<500 or np.median(residual)>4.:continue
            q,t=camera['q'][source,v],camera['t'][source,v]
            world_center=q@translation+t;world_rotation=q@R
            yaw=float(np.arctan2(world_rotation[1,2],world_rotation[0,2]))
            candidates.append(dict(center=world_center,yaw=yaw,view=v,inliers=len(inliers),
                median_reprojection_px=float(np.median(residual))))
        if not candidates:
            details.append(dict(source_index=source,status='unknown_no_pnp',observed_views=len(records)))
            continue
        chosen.setdefault(key,asset_index)
        center=np.median([c['center'] for c in candidates],axis=0)
        direction=np.array([np.cos([c['yaw'] for c in candidates]).sum(),np.sin([c['yaw'] for c in candidates]).sum()])
        yaw=float(np.arctan2(direction[1],direction[0])) if np.linalg.norm(direction)>1e-6 else candidates[0]['yaw']
        source_indices.append(source);centers.append(center);rotations.append(upright(yaw));dims.append(dimension)
        details.append(dict(source_index=source,status='pnp_seed',views=len(candidates),asset_index=asset_index,
            candidates=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in candidates]))
    if not source_indices:raise ValueError('No valid image-derived seeds; do not substitute reference poses')
    return dict(source_indices=np.asarray(source_indices,np.int32),center=np.asarray(centers),
        rotation=np.asarray(rotations),dims=np.asarray(dims)),details
