"""Experimental causal finite-window MAP with separate robust noise channels.

Same CV transition, G map, Q and measured R as the numeric replay. Gaussian
control must reproduce the no-shock Kalman filter. Huber process and observation
penalties are independent interventions, not a parameter search. Only present
and past measurements are accepted. Covariance is an IRLS approximation, not a
calibrated posterior or a replacement of the released model.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from .state import wrap,matrix
from .geometry import corners


def huber_groups(residual, threshold):
    value=np.asarray(residual).reshape(-1,6)
    norm=np.linalg.norm(value,axis=1)
    if threshold is None:
        return np.ones_like(norm),float(.5*np.sum(value**2))
    weight=np.minimum(1.,threshold/np.maximum(norm,1e-30))
    cost=np.where(norm<=threshold,.5*norm**2,threshold*(norm-.5*threshold))
    return weight,float(cost.sum())


def fit_window(mean,cov,measurements,noise,process,process_threshold=None,measurement_threshold=None,max_iterations=800):
    T=len(measurements);n=12+6*T
    F=np.eye(12);F[:6,6:]=np.eye(6)
    G=np.r_[.5*np.eye(6),np.eye(6)]
    mapping=np.zeros((12,n));mapping[:,:12]=np.linalg.cholesky((cov+cov.T)/2+np.eye(12)*1e-12)
    nominal=mean.copy();aa=[];bb=[]
    for t,(z,R) in enumerate(zip(measurements,noise)):
        nominal=F@nominal;mapping=F@mapping
        mapping[:,12+6*t:18+6*t]=G@np.diag(np.sqrt(process))
        L=np.linalg.cholesky(R)
        residual=z-nominal[:6];residual[3:]=wrap(residual[3:])
        aa.append(solve_triangular(L,mapping[:6],lower=True))
        bb.append(solve_triangular(L,residual,lower=True))
    A=np.concatenate(aa);b=np.concatenate(bb)
    u=np.zeros(n);costs=[];converged=False
    # No true pose/velocity, contact label, image or future observation is used.
    for iteration in range(max_iterations):
        wp,_=huber_groups(u[12:],process_threshold)
        wm,_=huber_groups(A@u-b,measurement_threshold)
        prior=np.r_[np.ones(12),np.repeat(wp,6)]
        obs=np.repeat(wm,6)
        lhs=A.T@(obs[:,None]*A)+np.diag(prior)
        rhs=A.T@(obs*b)
        factor=cho_factor(lhs,lower=True,check_finite=False)
        new=cho_solve(factor,rhs,check_finite=False)
        _,cp=huber_groups(new[12:],process_threshold)
        _,cm=huber_groups(A@new-b,measurement_threshold)
        cost=.5*new[:12]@new[:12]+cp+cm
        if costs and cost>costs[-1]+1e-7*max(1.,costs[-1]):
            raise AssertionError('IRLS objective increased')
        costs.append(float(cost))
        delta=float(np.linalg.norm(new-u));u=new
        if delta<=1e-7*(1+np.linalg.norm(u)):
            converged=True;break
    fallback_used=False;fallback_iterations=0
    if not converged:
        # Near a Huber transition IRLS can converge very slowly. Solve the SAME
        # convex objective in coordinates preconditioned by the last IRLS system.
        fallback_used=True
        L=np.linalg.cholesky(lhs)
        anchor=u.copy()
        def objective(v):
            x=anchor+solve_triangular(L.T,v,lower=False)
            wp,cp=huber_groups(x[12:],process_threshold)
            wm,cm=huber_groups(A@x-b,measurement_threshold)
            gradient=A.T@(np.repeat(wm,6)*(A@x-b))+np.r_[x[:12],np.repeat(wp,6)*x[12:]]
            value=.5*x[:12]@x[:12]+cp+cm
            return value,solve_triangular(L,gradient,lower=True)
        fit=minimize(objective,np.zeros(n),jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':300})
        value,gradient_preconditioned=objective(fit.x)
        if value>costs[-1]+1e-8*max(1.,costs[-1]):raise AssertionError('Fallback increased original convex objective')
        u=anchor+solve_triangular(L.T,fit.x,lower=False)
        costs.append(float(value));fallback_iterations=int(fit.nit)
        converged=bool(np.max(np.abs(gradient_preconditioned))<1e-6)
    wp,_=huber_groups(u[12:],process_threshold)
    wm,_=huber_groups(A@u-b,measurement_threshold)
    prior=np.r_[np.ones(12),np.repeat(wp,6)];obs=np.repeat(wm,6)
    gradient=A.T@(obs*(A@u-b))+prior*u
    lhs=A.T@(obs[:,None]*A)+np.diag(prior)
    factor=cho_factor(lhs,lower=True,check_finite=False)
    state=nominal+mapping@u
    posterior=mapping@cho_solve(factor,mapping.T,check_finite=False)
    posterior=(posterior+posterior.T)/2
    return state,posterior,dict(iterations=iteration+1,converged=converged,fallback_used=fallback_used,fallback_iterations=fallback_iterations,
        gradient_inf=float(np.max(np.abs(gradient))),relative_gradient=float(np.max(np.abs(gradient))/(1+np.max(np.abs(A.T@b)))),
        cost=costs[-1],measurement_weight_min=float(wm.min()),process_weight_min=float(wp.min()))


def replay_window(cache,states,dims,window=8,process_threshold=None,measurement_threshold=None,limit=None):
    if window<1:raise ValueError('Positive window required')
    history={};outputs=[];traces=[];kept=[]
    for i,source in enumerate(cache['source_indices']):
        row=states[int(source)]
        if limit is not None and row['step']>=limit:continue
        key=row['object_key'];track=history.get(key)
        z=cache['z'][i];R=cache['covariance'][i]
        if track is None or row['step']!=track[-1]['step']+1:
            state=np.r_[z,np.zeros(6)];P=np.eye(12);P[:6,:6]=R
            track=[]
            trace=dict(iterations=0,converged=True,fallback_used=False,fallback_iterations=0,gradient_inf=0.,relative_gradient=0.,cost=0.,measurement_weight_min=1.,process_weight_min=1.)
        else:
            prior_index=max(0,len(track)-window)
            anchor=track[prior_index]
            subsequent=track[prior_index+1:]
            zz=[r['measurement'] for r in subsequent]+[z]
            rr=[r['R'] for r in subsequent]+[R]
            state,P,trace=fit_window(anchor['state'],anchor['P'],zz,rr,cache['process'],
                process_threshold=process_threshold,measurement_threshold=measurement_threshold)
        track.append(dict(step=row['step'],state=state.copy(),P=P.copy(),measurement=z.copy(),R=R.copy()))
        history[key]=track
        outputs.append(state);traces.append(trace);kept.append(i)
    x=np.asarray(outputs);rot=matrix(x[:,3:6]);d=dims[kept]
    geometry=dict(source_indices=cache['source_indices'][kept],center=x[:,:3],rotation=rot,dims=d,corners=corners(x[:,:3],rot,d))
    trace={k:np.asarray([r[k] for r in traces]) for k in traces[0]};trace['state']=x
    return geometry,trace
