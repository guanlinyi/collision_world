"""Offline implicit derivative of the actual robust pose objective.

This requires an already fitted pose and supervised reference pose. It does not
train a network or define a deployable correction. All inputs are numeric.
"""
import torch
from .pose_influence import skew


def objective(parameters, local, origin, initial, camera_center, camera_rotation,
              intrinsics, observed_pixel, sigma, lever_m, prior_scale=1.):
    """Same soft-L1 objective as geometry.solve, in metre-scaled parameters."""
    center=origin+parameters[:3]
    rotation=torch.matrix_exp(skew(parameters[3:]/lever_m))@initial
    world=local@rotation.T+center
    cc=torch.einsum('ni,nij->nj',world-camera_center,camera_rotation)
    uv=cc[:,:2]/cc[:,2,None].clamp(min=.5)*torch.stack([intrinsics[:,0,0],intrinsics[:,1,1]],-1)+intrinsics[:,:2,2]
    residual=(uv-observed_pixel)/sigma
    prior=torch.cat([parameters[:3]*.2,parameters[3:]/lever_m*(.2/(torch.pi/6))])*prior_scale
    return (torch.sqrt(1+residual.square())-1).sum()+(torch.sqrt(1+prior.square())-1).sum()


def supervised_cost(parameters,origin,initial,reference_center,reference_rotation,lever_m):
    center=origin+parameters[:3]
    rotation=torch.matrix_exp(skew(parameters[3:]/lever_m))@initial
    cosine=((torch.trace(rotation@reference_rotation.T)-1)/2).clamp(-1+1e-14,1-1e-14)
    angle=torch.acos(cosine)
    return (center-reference_center).square().sum()+lever_m**2*angle.square()


def implicit_coordinate_gradient(parameters,local,origin,initial,camera_center,camera_rotation,
                                 intrinsics,observed_pixel,sigma,lever_m,
                                 reference_center,reference_rotation,prior_scale=1.,
                                 hessian='exact'):
    """d pose_error / d coordinates via the stationary-point equation.

    Valid only for a smooth isolated unconstrained local minimum. Caller must
    inspect stationarity, Hessian spectrum, and active solver bounds. No flooring
    conceals indefiniteness. This diagnostic uses reference pose in its target.
    """
    if hessian!='exact':raise ValueError('Only exact objective Hessian implemented')
    x=parameters.detach().clone().requires_grad_()
    points=local.detach().clone().requires_grad_()
    args=(origin,initial,camera_center,camera_rotation,intrinsics,observed_pixel,sigma,lever_m,prior_scale)
    value=objective(x,points,*args)
    stationarity=torch.autograd.grad(value,x,create_graph=True)[0]
    H=torch.stack([torch.autograd.grad(stationarity[j],x,retain_graph=True)[0] for j in range(6)])
    H=(H+H.T)/2
    eigenvalues=torch.linalg.eigvalsh(H)
    if eigenvalues[0]<=0:raise ValueError('Non-positive Hessian: no isolated minimum derivative')
    cost=supervised_cost(x,origin,initial,reference_center,reference_rotation,lever_m)
    target_gradient=torch.autograd.grad(cost,x)[0]
    adjoint=torch.linalg.solve(H.detach(),target_gradient.detach())
    gradient=-torch.autograd.grad(stationarity,points,grad_outputs=adjoint)[0]
    return dict(gradient=gradient.detach(),eigenvalues=eigenvalues.detach(),
                stationarity=stationarity.detach(),objective=float(value.detach()),
                supervised_cost=float(cost.detach()),hessian=H.detach())
