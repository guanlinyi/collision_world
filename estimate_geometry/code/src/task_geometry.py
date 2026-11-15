"""Conservative task decisions for uncertain convex cuboids, without ground truth.

An error budget e bounds Hausdorff distance to the true convex cuboid. Enlarging
each half extent by e contains it; shrinking each half extent by e is inside it.
Certificates are conditional on this bound, not estimated probabilities.
"""
import numpy as np
from .collision import box_contact


def _checked(box):
    result = {key: np.asarray(box[key], dtype=float) for key in ('center', 'rotation', 'dims')}
    if (result['center'].shape != (3,) or result['rotation'].shape != (3, 3)
            or result['dims'].shape != (3,)):
        raise ValueError('Expected center (3,), rotation (3,3), dims (3,)')
    if not all(np.isfinite(v).all() for v in result.values()) or np.any(result['dims'] <= 0):
        raise ValueError('Finite geometry and positive dimensions required')
    r = result['rotation']
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('Rotation must be a proper orthonormal matrix')
    return result


def bounded_contact(a, b, error_a_m, error_b_m, numerical_guard_m=1e-7):
    """Return separated/overlap/unknown. Touching is deliberately unknown.

    No truth or learned confidence is read. Bounds must cover the WHOLE cuboid,
    including center, rotation and size errors. Center mean error is insufficient.
    Disjoint outer boxes prove separation; intersecting inner boxes prove overlap.
    Failure to certify is inconclusive, not proof of physical ambiguity.
    """
    a, b = _checked(a), _checked(b)
    errors = np.asarray([error_a_m, error_b_m, numerical_guard_m], dtype=float)
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError('Finite nonnegative error budgets and guard required')
    outer_a = dict(a, dims=a['dims'] + 2 * error_a_m)
    outer_b = dict(b, dims=b['dims'] + 2 * error_b_m)
    outer = box_contact(outer_a, outer_b)['minimum_axis_margin_m']
    inner = None
    if outer < -numerical_guard_m:
        decision = 'separated'
    else:
        dims_a, dims_b = a['dims'] - 2 * error_a_m, b['dims'] - 2 * error_b_m
        if np.all(dims_a > 0) and np.all(dims_b > 0):
            inner = box_contact(dict(a, dims=dims_a), dict(b, dims=dims_b))['minimum_axis_margin_m']
        decision = 'overlap' if inner is not None and inner > numerical_guard_m else 'unknown'
    return dict(decision=decision, outer_axis_margin_m=float(outer),
                inner_axis_margin_m=None if inner is None else float(inner))
