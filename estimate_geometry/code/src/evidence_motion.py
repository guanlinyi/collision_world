"""Experimental causal image-evidence update, not a deployed default.

Carries pose and velocity, but updates pose against the original pixel evidence
instead of treating a noisy single-frame pose as an exact Gaussian observation.
No reference geometry, depth, or training target is accepted by this module.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from .common import grouped_records, read_json, is_clipped
from .state import matrix, wrap
from .geometry import corners, pixel_sigma


def observations(bundle, surface, raw, model_dir, use_point_sigma=False):
    noise = read_json(Path(model_dir)/'view_noise.json')['normalized_sigma']
    config = read_json(Path(model_dir)/'model.json')
    o, cams = bundle['observations'], bundle['cameras']
    groups = grouped_records(bundle)
    result = []
    for i, source in enumerate(raw['source_indices']):
        views = []
        for row in groups.get(int(source), []):
            j, v = row['index'], row['view']
            ids = np.flatnonzero(o['mask'][j] & (surface['confidence'][j] > .6))
            if len(ids) < 24 or not cams['valid'][source, v]:
                continue
            # Match the original geometry's sampling, not an unlabelled resampling change.
            ids = ids[np.linspace(0, len(ids)-1, min(64, len(ids))).astype(int)]
            pts = surface['coords'][j].reshape(-1, 3)[ids].astype(float)*raw['dims'][i]
            uv = o['pixels'][j].reshape(-1, 2)[ids].astype(float)+config.get('pixel_center_offset', 0.)
            q, t, k = cams['q'][source, v], cams['t'][source, v], cams['k'][source, v]
            bb = row['bbox']
            base = max(.5, (1+.03*min(bb[2]-bb[0], bb[3]-bb[1])) * noise['clipped' if is_clipped(bb) else 'complete'])
            sig = np.full((len(ids), 2), base)
            if use_point_sigma:
                sigma = surface['sigma'][j].reshape(-1, 3)[ids].astype(float)
                sig = pixel_sigma(pts, raw['center'][i], raw['rotation'][i], raw['dims'][i], q, t, k, base, sigma)
            views.append((pts, uv, q, t, k, sig))
        result.append(views)
    return result


def replay_evidence(cache, states, dims, views_by_state, effective_points=1., shock_structure='diagonal', limit=None):
    F = np.eye(12)
    F[:6, 6:] = np.eye(6)
    G = np.r_[.5*np.eye(6), np.eye(6)]
    Q = G @ np.diag(cache['process']) @ G.T
    H = np.eye(12)[:6]
    history, outputs, kept, traces = {}, [], [], []
    for i, source in enumerate(cache['source_indices']):
        row = states[int(source)]
        if limit is not None and row['step'] >= limit:
            continue
        z, R = cache['z'][i], cache['covariance'][i]
        prev = history.get(row['object_key'])
        nis, evaluations, success, residual_cost = 0., 0, True, 0.
        if prev is None or row['step'] != prev[2]+1 or not views_by_state[i]:
            state = np.r_[z, np.zeros(6)]
            P = np.eye(12)
            P[:6, :6] = R
        else:
            predicted = F @ prev[0]
            prior = F @ prev[1] @ F.T + Q
            innovation = z-predicted[:6]
            innovation[3:] = wrap(innovation[3:])
            S = prior[:6, :6]+R
            nis = float(innovation @ np.linalg.solve(S, innovation))
            if nis > 24:
                shock = (np.diag(np.maximum(innovation**2-np.diag(S), 0)) if shock_structure == 'diagonal'
                         else max(0., 1.-6./nis)*np.outer(innovation, innovation))
                prior += G @ shock @ G.T
                S = prior[:6, :6]+R
            K = np.linalg.solve(S, prior[:6, :]).T
            initial = predicted[:6]+K[:6] @ innovation
            L = np.linalg.cholesky(prior[:6, :6])
            W = solve_triangular(L, np.eye(6), lower=True)

            def residual(x):
                rotation = matrix(x[3:])
                values = []
                for pts, uv, q, t, k, sigma in views_by_state[i]:
                    cc = (pts @ rotation.T+x[:3]-t) @ q
                    projected = cc[:, :2]/cc[:, 2, None].clip(.5)*[k[0, 0], k[1, 1]]+[k[0, 2], k[1, 2]]
                    raw = ((projected-uv)/sigma).ravel()
                    # sqrt soft-L1, keeping the prior quadratic. Each view's
                    # assumed independent information is explicit, not hidden.
                    robust = raw*np.sqrt(2/(np.sqrt(1+raw*raw)+1))
                    values.append(robust*np.sqrt(effective_points/len(pts)))
                delta = x-predicted[:6]
                delta[3:] = wrap(delta[3:])
                return np.r_[np.concatenate(values), W @ delta]

            fit = least_squares(residual, initial, max_nfev=30, xtol=1e-7, ftol=1e-7, gtol=1e-7)
            evaluations, success, residual_cost = fit.nfev, bool(fit.success), float(2*fit.cost)
            if not success or not np.isfinite(fit.x).all():
                # Keep a defined causal fallback and expose its occurrence.
                state = predicted+K @ innovation
                A = np.eye(12)-K @ H
                P = A @ prior @ A.T+K @ R @ K.T
            else:
                conditional = np.linalg.solve(prior[:6, :6], prior[:6]).T
                delta = fit.x-predicted[:6]
                delta[3:] = wrap(delta[3:])
                state = predicted+conditional @ delta
                pose_cov = np.linalg.inv(fit.jac.T @ fit.jac+np.eye(6)*1e-10)
                P = prior+conditional @ (pose_cov-prior[:6, :6]) @ conditional.T
                P = (P+P.T)/2
            if not np.isfinite(state).all() or np.linalg.eigvalsh(P).min() < -1e-7:
                raise ValueError('Invalid nonlinear state/covariance')
        history[row['object_key']] = (state.copy(), P.copy(), row['step'])
        outputs.append(state)
        kept.append(i)
        traces.append(dict(nis=nis, evaluations=evaluations, success=success, cost=residual_cost, covariance=P[:6, :6]))
    out = np.array(outputs)
    rot = matrix(out[:, 3:6])
    geometry = dict(source_indices=cache['source_indices'][kept], center=out[:, :3], rotation=rot, dims=dims[kept],
                    corners=corners(out[:, :3], rot, dims[kept]))
    trace = {k: np.array([t[k] for t in traces]) for k in traces[0]}
    trace['state'] = out
    return geometry, trace
