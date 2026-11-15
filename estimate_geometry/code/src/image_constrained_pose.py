"""Matched joint center/orientation candidate; updates only current published pose."""
import numpy as np
from .state import matrix,wrap
from .temporal_image_motion import project
from .image_constrained_center import projection_jacobian


def correct_pose(anchor,covariance,past,views,max_iterations=12):
    anchor=np.asarray(anchor,float);C=np.asarray(covariance,float);C=(C+C.T)/2
    eig,U=np.linalg.eigh(C);precision=(U*(1/np.maximum(eig,1e-10)))@U.T
    past_rotation=matrix(past[3:]);prepared=[]
    for v in views:
        local=v['local'];p0,d0=project(local@past_rotation.T+past[:3],v['q0'],v['t0'],v['k0'])
        _,d1=project(local@matrix(anchor[3:]).T+anchor[:3],v['q1'],v['t1'],v['k1'])
        good=(d0>.5)&(d1>.5)&np.isfinite(local).all(1)
        if good.sum()<4:continue
        selected=local[good]
        noise=np.broadcast_to(np.eye(2)*v['sigma_native_px']**2,(len(selected),2,2)).copy()
        if 'local_sigma' in v:
            world0=selected@past_rotation.T+past[:3]
            world1=selected@matrix(anchor[3:]).T+anchor[:3]
            J0=projection_jacobian(world0,v['q0'],v['t0'],v['k0'])@past_rotation
            J1=projection_jacobian(world1,v['q1'],v['t1'],v['k1'])@matrix(anchor[3:])
            D=J1-J0
            noise+=np.einsum('nki,ni,nji->nkj',D,v['local_sigma'][good]**2,D)
        prepared.append(dict(v=v,local=selected,past_uv=p0[good],observed=v['displacement'][good],
                             noise_cholesky=np.linalg.cholesky(noise)))
    if not prepared:return anchor.copy(),dict(status='unknown_no_tracks',views=0,tracks=0,iterations=0)
    def linearize(x):
        delta=x-anchor;delta[3:]=wrap(delta[3:]);A=precision.copy();rhs=-precision@delta
        energy=float(delta@precision@delta);errors=[]
        for entry in prepared:
            v=entry['v'];local=entry['local'];L=entry['noise_cholesky']
            world=local@matrix(x[3:]).T+x[:3];uv,depth=project(world,v['q1'],v['t1'],v['k1'])
            if np.any(depth<=.5):return A,rhs,float('inf'),errors
            residual=uv-entry['past_uv']-entry['observed'];J=np.empty((len(local),2,6))
            J[:,:,:3]=projection_jacobian(world,v['q1'],v['t1'],v['k1'])
            for a in range(3):
                d=np.eye(3)[a]*1e-5
                uplus,_=project(local@matrix(x[3:]+d).T+x[:3],v['q1'],v['t1'],v['k1'])
                uminus,_=project(local@matrix(x[3:]-d).T+x[:3],v['q1'],v['t1'],v['k1'])
                J[:,:,a+3]=(uplus-uminus)/2e-5
            e=np.linalg.solve(L,residual[:,:,None]).squeeze(-1)
            J=np.linalg.solve(L,J);norm=np.linalg.norm(e,axis=1)
            weight=np.minimum(1.,1.5/np.maximum(norm,1e-12))/len(e)
            A+=np.einsum('n,nki,nkj->ij',weight,J,J)
            rhs-=np.einsum('n,nki,nk->i',weight,J,e)
            energy+=float(np.sum(np.where(norm<=1.5,norm**2,3*norm-2.25))/len(e))
            errors.extend(np.linalg.norm(residual,axis=1).tolist())
        return A,rhs,energy,errors
    estimate=anchor.copy();initial=linearize(estimate)[2]
    for iteration in range(1,max_iterations+1):
        A,rhs,energy,_=linearize(estimate);delta=np.linalg.solve(A,rhs)
        if np.linalg.norm(delta[:3])+np.linalg.norm(delta[3:])<1e-7:break
        factor=1.;accepted=False
        for _ in range(8):
            new=estimate+factor*delta
            if linearize(new)[2]<=energy+1e-10:estimate=new;accepted=True;break
            factor*=.5
        if not accepted:break
    final=linearize(estimate)
    return estimate,dict(status='corrected',views=len(prepared),tracks=sum(len(x['local']) for x in prepared),
        iterations=iteration,initial_energy=initial,final_energy=final[2],mean_image_error_px=float(np.mean(final[3])),
        center_correction_m=float(np.linalg.norm(estimate[:3]-anchor[:3])),
        euler_correction_rad=float(np.linalg.norm(wrap(estimate[3:]-anchor[3:]))))


def correct_pose_checked(anchor,covariance,past,views):
    """Cross-view check of new motion evidence, not independent end-to-end proof.

    Baseline itself used these RGB cameras; only their *new motion factors* are
    withheld. Multiple views may still have correlated model/appearance errors.
    """
    if len(views)<2:
        return np.asarray(anchor).copy(),dict(status='unknown_no_cross_view',views=len(views),tracks=sum(len(v['local']) for v in views),iterations=0)
    def cost(x,v):
        p0,z0=project(v['local']@matrix(past[3:]).T+past[:3],v['q0'],v['t0'],v['k0'])
        p1,z1=project(v['local']@matrix(x[3:]).T+x[:3],v['q1'],v['t1'],v['k1'])
        if np.any(z0<=.5) or np.any(z1<=.5):return float('inf')
        residual=(p1-p0-v['displacement'])/v['sigma_native_px']
        norm=np.linalg.norm(residual,axis=1)
        return float(np.mean(np.where(norm<=1.5,norm**2,3*norm-2.25)))
    votes=[]
    for j,heldout in enumerate(views):
        trained,_=correct_pose(anchor,covariance,past,[v for i,v in enumerate(views) if i!=j])
        old,new=cost(anchor,heldout),cost(trained,heldout)
        votes.append(dict(heldout=j,before=old,after=new,supports=new<old-1e-6))
    agreeing=sum(v['supports'] for v in votes)
    if agreeing<=len(votes)/2:
        return np.asarray(anchor).copy(),dict(status='rejected_cross_view',views=len(views),tracks=sum(len(v['local']) for v in views),
            iterations=0,agreeing_views=agreeing,total_views=len(votes),cross_view_votes=votes)
    result,detail=correct_pose(anchor,covariance,past,views)
    detail.update(agreeing_views=agreeing,total_views=len(votes),cross_view_votes=votes)
    return result,detail


def replay_pose(baseline_state,covariances,source_indices,states,measurements,camera,cross_view=False):
    output=baseline_state.copy();hist={};details=[]
    for i,s in enumerate(source_indices):
        s=int(s);st=states[s];past=hist.get(st['object_key']);views=[]
        for m in measurements.get(s,[]):
            if past is None or m['source0']!=past['source']:continue
            views.append(dict(local=m['local'],displacement=m['displacement'],sigma_native_px=m['sigma_native_px'],
                **({'local_sigma':m['local_sigma']} if 'local_sigma' in m else {}),
                **{k+str(n):camera[k][src,m['view']] for n,src in [(0,m['source0']),(1,s)] for k in ['q','t','k']}))
        solver=correct_pose_checked if cross_view else correct_pose
        if past is not None and past['step']+1==st['step']:x,d=solver(baseline_state[i],covariances[i],past['state'],views)
        else:x=baseline_state[i].copy();d=dict(status='unknown_no_tracks',views=0,tracks=0,iterations=0)
        output[i]=x;details.append(d);hist[st['object_key']]=dict(step=st['step'],source=s,state=x.copy())
    return output,details
