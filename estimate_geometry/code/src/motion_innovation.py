"""Motion covariance adaptation with an explicit legacy/research contract.

Directional inflation preserves correlations in the observed innovation. It is
equivariant under consistent linear reparameterization of the pose covariance.
Neither rule determines whether an innovation came from motion or bad sensing;
neither supplies a calibrated coverage probability.
"""
import numpy as np


def innovation_covariance(innovation, covariance, rule='released', nis=None):
    """Return a PSD 6D covariance addition BEFORE mapping into motion state.

Caller owns the unchanged NIS > 24 trigger and temporal injection map. Keeping
those explicit avoids silently changing three algorithm choices at once.
"""
    delta=np.asarray(innovation,dtype=float)
    S=np.asarray(covariance,dtype=float)
    if delta.shape!=(6,) or S.shape!=(6,6):
        raise ValueError('Six-dimensional innovation and 6x6 covariance required')
    if not np.isfinite(delta).all() or not np.isfinite(S).all():
        raise ValueError('Finite innovation and covariance required')
    if rule=='released':
        return np.diag(np.maximum(delta**2-np.diag(S),0.))
    if rule!='directional':
        raise ValueError('Unknown motion update rule: '+str(rule))
    score=float(delta@np.linalg.solve(S,delta)) if nis is None else float(nis)
    if not np.isfinite(score) or score<0:
        raise ValueError('Finite nonnegative normalized innovation required')
    if score<=6.:
        return np.zeros((6,6))
    return (1.-6./score)*np.outer(delta,delta)
