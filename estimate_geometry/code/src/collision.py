"""Exact separating-axis test for two finite oriented cuboids.

Contact is a geometric diagnostic, not a CARLA collision callback.
"""
import itertools
import numpy as np


def box_contact(a, b, tolerance_m=1e-7):
    center_a=np.asarray(a['center'],dtype=float)
    center_b=np.asarray(b['center'],dtype=float)
    axes_a=np.asarray(a['rotation'],dtype=float)
    axes_b=np.asarray(b['rotation'],dtype=float)
    half_a=np.asarray(a['dims'],dtype=float)/2
    half_b=np.asarray(b['dims'],dtype=float)/2
    if not (np.isfinite(center_a).all() and np.isfinite(center_b).all()
            and np.isfinite(axes_a).all() and np.isfinite(axes_b).all()
            and (half_a>0).all() and (half_b>0).all()):
        raise ValueError('Finite centers, rotations, and positive dimensions are required')
    candidates=[axes_a[:,i] for i in range(3)]+[axes_b[:,i] for i in range(3)]
    candidates += [np.cross(axes_a[:,i],axes_b[:,j]) for i,j in itertools.product(range(3),repeat=2)]
    signed_margins=[]
    displacement=center_b-center_a
    for axis in candidates:
        norm=np.linalg.norm(axis)
        if norm<1e-10:continue
        axis=axis/norm
        radius_a=np.sum(half_a*np.abs(axes_a.T@axis))
        radius_b=np.sum(half_b*np.abs(axes_b.T@axis))
        signed_margins.append(float(radius_a+radius_b-abs(displacement@axis)))
    margin=min(signed_margins)
    return dict(overlap=margin>=-tolerance_m,minimum_axis_margin_m=margin,
                meaning='Positive: every tested axis overlaps; negative: a separating axis exists. This margin is not necessarily Euclidean surface distance.')


def frame_contacts(objects):
    return [dict(a=a.get('track',str(i)),b=b.get('track',str(j)),**box_contact(a,b))
            for (i,a),(j,b) in itertools.combinations(enumerate(objects),2)]
