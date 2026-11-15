"""Use image displacement to constrain translation with current angles fixed."""
import numpy as np
from .image_constrained_center import correct_current
from .state import matrix
from .temporal_image_motion import project


def correct_center_checked(anchor, covariance, past, views):
    if len(views) < 2:
        return np.asarray(anchor).copy(), dict(status='unknown_no_cross_view', views=len(views),
            tracks=sum(len(v['local']) for v in views), iterations=0)
    R, R0 = matrix(anchor[3:]), matrix(past[3:])

    def fit(chosen):
        return correct_current(anchor[:3], R, covariance, past[:3], R0, chosen)

    def cost(center, v):
        p0, z0 = project(v['local'] @ R0.T + past[:3], v['q0'], v['t0'], v['k0'])
        p1, z1 = project(v['local'] @ R.T + center, v['q1'], v['t1'], v['k1'])
        if np.any(z0 <= .5) or np.any(z1 <= .5):
            return float('inf')
        residual = (p1 - p0 - v['displacement']) / v['sigma_native_px']
        norm = np.linalg.norm(residual, axis=1)
        return float(np.mean(np.where(norm <= 1.5, norm**2, 3*norm-2.25)))

    votes = []
    for j, heldout in enumerate(views):
        center, _ = fit([v for i, v in enumerate(views) if i != j])
        old, new = cost(anchor[:3], heldout), cost(center, heldout)
        votes.append(dict(heldout=j, before=old, after=new, supports=new < old - 1e-6))
    agreeing = sum(v['supports'] for v in votes)
    if agreeing <= len(votes) / 2:
        return np.asarray(anchor).copy(), dict(status='rejected_cross_view', views=len(views),
            tracks=sum(len(v['local']) for v in views), iterations=0,
            agreeing_views=agreeing, total_views=len(votes), cross_view_votes=votes)
    center, detail = fit(views)
    pose = np.asarray(anchor).copy()
    pose[:3] = center
    detail.update(agreeing_views=agreeing, total_views=len(votes), cross_view_votes=votes)
    return pose, detail
