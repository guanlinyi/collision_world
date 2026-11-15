"""Causal current-center correction from observed image displacement.

This candidate uses a baseline pose and covariance, preceding published pose and
measured image tracks.  No truth or future state arguments.  Old states are never
changed. Baseline and tracks share images: covariance is a weighting heuristic,
not a calibrated Bayesian posterior or independent-evidence claim.
"""
import numpy as np
from .temporal_image_motion import project


def projection_jacobian(points,q,t,k):
    x=(points-t)@q;z=x[:,2]
    J=np.zeros((len(x),2,3))
    J[:,0,0]=k[0,0]/z;J[:,1,1]=k[1,1]/z
    J[:,0,2]=-k[0,0]*x[:,0]/z**2
    J[:,1,2]=-k[1,1]*x[:,1]/z**2
    return J@q.T


def correct_current(center,rotation,covariance,previous_center,previous_rotation,views,max_iterations=12):
    anchor=np.asarray(center,float);C=np.asarray(covariance,float)
    C=(C+C.T)/2
    eigen,axes=np.linalg.eigh(C)
    precision=(axes*(1/np.maximum(eigen,1e-8)))@axes.T
    prepared=[]
    for v in views:
        local=v['local'];past=local@previous_rotation.T+previous_center
        p0,d0=project(past,v['q0'],v['t0'],v['k0'])
        current=local@rotation.T+anchor
        _,d1=project(current,v['q1'],v['t1'],v['k1'])
        good=(d0>.5)&(d1>.5)&np.isfinite(local).all(1)
        if good.sum()<4:continue
        prepared.append(dict(v=v,local=local[good],past_uv=p0[good],observed=v['displacement'][good],
                             sigma=float(v['sigma_native_px'])))
    if not prepared:return anchor.copy(),dict(status='unknown_no_tracks',iterations=0,views=0,tracks=0)
    estimate=anchor.copy();conditioning=[]
    def linearize(c):
        A=precision.copy();rhs=precision@(anchor-c);energy=float((c-anchor)@precision@(c-anchor))
        info=np.zeros((3,3));errors=[]
        for p in prepared:
            v=p['v'];world=p['local']@rotation.T+c
            uv,depth=project(world,v['q1'],v['t1'],v['k1'])
            if np.any(depth<=.5):return A,rhs,float('inf'),info,errors
            residual=uv-p['past_uv']-p['observed']
            J=projection_jacobian(world,v['q1'],v['t1'],v['k1'])/p['sigma']
            e=residual/p['sigma'];norm=np.linalg.norm(e,axis=1)
            # One view remains one source; duplicated resized pixels and
            # correlated patch tracks do not become independent observations.
            w=np.minimum(1.,1.5/np.maximum(norm,1e-12))/len(e)
            add=np.einsum('n,nki,nkj->ij',w,J,J)
            A+=add;info+=add;rhs-=np.einsum('n,nki,nk->i',w,J,e)
            energy+=float(np.sum(np.where(norm<=1.5,norm**2,3*norm-2.25))/len(e))
            errors.extend(np.linalg.norm(residual,axis=1).tolist())
        return A,rhs,energy,info,errors
    initial_energy=linearize(estimate)[2]
    iterations=0
    for iterations in range(1,max_iterations+1):
        A,rhs,energy,_,_=linearize(estimate)
        delta=np.linalg.solve(A,rhs)
        if np.linalg.norm(delta)<1e-7:break
        factor=1.
        accepted=False
        for _ in range(8):
            candidate=estimate+factor*delta
            if linearize(candidate)[2]<=energy+1e-10:
                estimate=candidate;accepted=True;break
            factor*=.5
        if not accepted:break
    A,_,energy,info,errors=linearize(estimate)
    eig=np.linalg.eigvalsh(info)
    return estimate,dict(status='corrected',iterations=iterations,views=len(prepared),
        tracks=sum(len(p['local']) for p in prepared),correction_m=float(np.linalg.norm(estimate-anchor)),
        image_information_eigenvalues=eig.tolist(),image_condition_ratio=float(eig[0]/max(eig[-1],1e-12)),
        initial_energy=initial_energy,final_energy=energy,mean_image_error_px=float(np.mean(errors)))


def replay(baseline,position_covariances,states,measurements,camera,local=False):
    """Online replay API without truth, teachers or RGB; measurements already causal."""
    ix=baseline['source_indices'];history={};centers=baseline['center'].copy();details=[]
    for i,source in enumerate(ix):
        source=int(source);state=states[source];key=state['object_key']
        past=history.get(key);views=[]
        for m in measurements.get(source,[]):
            if past is None or m['source0']!=past['source']:continue
            v=m['view'];s0=m['source0']
            views.append(dict(local=m['local'],displacement=m['displacement'],sigma_native_px=m['sigma_native_px'],
                **{k+str(n):camera[k][s,v] for n,s in [(0,s0),(1,source)] for k in ['q','t','k']}))
        if past is not None and past['step']+1==state['step']:
            oldcenter=baseline['center'][past['i']] if local else past['center']
            center,detail=correct_current(baseline['center'][i],baseline['rotation'][i],position_covariances[i],
                oldcenter,baseline['rotation'][past['i']],views)
        else:center=baseline['center'][i].copy();detail=dict(status='unknown_no_tracks',iterations=0,views=0,tracks=0)
        centers[i]=center;details.append(detail)
        history[key]=dict(source=source,step=state['step'],i=i,center=center.copy())
    return centers,details
