"""Experimental supervised diagnostic of pose-sensitive coordinate errors.

This is a local linear influence loss, not EPro-PnP, a calibrated posterior, or
a complete training recipe. It needs canonical coordinate/pose supervision and
must retain a canonical-coordinate anchor. No dataset is opened by this module.
"""
import torch


def skew(v):
    zero=torch.zeros_like(v[...,0])
    x,y,z=v.unbind(-1)
    return torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*v.shape[:-1],3,3)


def pose_influence(predicted_local, reference_local, center, rotation, camera_center,
                   camera_rotation, intrinsics, lever_m, sigma=None, relative_eigen_floor=1e-6):
    """First-order 6D pose displacement in metres induced by coordinate error.

    All points belong to one object; camera arrays may differ for every point.
    Local coordinates must already be in metres. Camera rotation columns map
    camera axes to world. The last three output components equal rotation vector
    times lever_m, making the quadratic loss dimensionally meaningful.

    The reference Jacobian, weights, and inverse information are detached so
    predictions cannot reduce the loss by weakening the information matrix.
    Eigenvalue flooring prevents numerical blow-up; rank deficiency is reported
    and must not be mistaken for good recovery of an unobservable direction.
    """
    if predicted_local.ndim!=2 or predicted_local.shape[1]!=3 or predicted_local.shape!=reference_local.shape:
        raise ValueError('Matching N by 3 object coordinates required')
    if len(predicted_local)<4 or lever_m<=0 or not 0<relative_eigen_floor<1:
        raise ValueError('At least four points, positive lever, and eigen floor in (0,1) required')
    dtype,device=predicted_local.dtype,predicted_local.device
    if dtype not in (torch.float32,torch.float64):raise ValueError('float32 or float64 required')
    n=len(predicted_local)
    def project(local):
        world=local@rotation.T+center
        cc=torch.einsum('ni,nij->nj',world-camera_center,camera_rotation)
        if torch.any(cc[:,2]<=.1):raise ValueError('Points must have positive camera depth above 0.1 m')
        uv=cc[:,:2]/cc[:,2,None]*torch.stack([intrinsics[:,0,0],intrinsics[:,1,1]],-1)+intrinsics[:,:2,2]
        return uv,cc
    predicted_pixel,_=project(predicted_local)
    with torch.no_grad():
        reference_pixel,cc=project(reference_local)
        z=cc[:,2]
        P=torch.zeros((n,2,3),dtype=dtype,device=device)
        P[:,0,0]=intrinsics[:,0,0]/z;P[:,1,1]=intrinsics[:,1,1]/z
        P[:,0,2]=-intrinsics[:,0,0]*cc[:,0]/z**2
        P[:,1,2]=-intrinsics[:,1,1]*cc[:,1]/z**2
        world_offset=reference_local@rotation.T
        tangent=torch.cat([torch.eye(3,dtype=dtype,device=device).expand(n,-1,-1),-skew(world_offset)/lever_m],-1)
        J=P@camera_rotation.transpose(-1,-2)@tangent
        weights=torch.ones((n,2),dtype=dtype,device=device) if sigma is None else sigma.detach()
        if weights.shape!=(n,2) or not torch.isfinite(weights).all() or torch.any(weights<=0):
            raise ValueError('Finite positive N by 2 sigma required')
        A=(J/weights[:,:,None]).reshape(-1,6)
        H=A.T@A
        eigenvalues,V=torch.linalg.eigh(H)
        floor=eigenvalues[-1].clamp(min=torch.finfo(dtype).eps)*relative_eigen_floor
        inverse=(V/eigenvalues.clamp(min=floor)[None])@V.T
        operator=-inverse@A.T
    residual=((predicted_pixel-reference_pixel)/weights).reshape(-1)
    displacement=operator@residual
    return dict(displacement=displacement,loss=displacement.square().sum(),
                eigenvalues=eigenvalues,clamped_directions=(eigenvalues<floor).sum(),
                relative_pixel_loss=residual.square().mean(),
                coordinate_loss=(predicted_local-reference_local).square().sum(-1).mean())
