"""Experimental, instrumented state replay. Does not open reference geometry.

Cache image-derived measurement covariances once; replay causal state rules with
exactly the same poses and observations. Not the released inference default.
"""
from pathlib import Path
import numpy as np
from .common import read_json, grouped_records, is_clipped
from .geometry import corners, pixel_sigma
from .state import angles, matrix, wrap
from .motion_innovation import innovation_covariance


def measurement_cache(bundle, surface, raw, model_dir):
    cfg = read_json(Path(model_dir)/'state_noise.json')
    noise = read_json(Path(model_dir)/'view_noise.json')['normalized_sigma']
    ix = raw['source_indices']
    z = np.c_[raw['center'], angles(raw['rotation'])]
    groups = grouped_records(bundle)
    o, camera = bundle['observations'], bundle['cameras']
    covariance, point_covariance, qualities = [], [], []
    for i, source in enumerate(ix):
        info, point_info = np.zeros((6, 6)), np.zeros((6, 6))
        used_views, pixels, min_size, clipped = 0, 0, [], 0
        for r in groups.get(int(source), []):
            if not r.get('geometry_allowed',True):
                continue
            k, v = r['index'], r['view']
            ids = np.flatnonzero(o['mask'][k] & (surface['confidence'][k] > .6))
            if len(ids) < 24:
                continue
            _, unique = np.unique(o['pixels'][k].reshape(-1, 2)[ids], axis=0, return_index=True)
            ids = ids[unique]
            ids = ids[np.linspace(0, len(ids)-1, min(64, len(ids))).astype(int)]
            pts = surface['coords'][k].reshape(-1, 3)[ids] * raw['dims'][i]
            q, t, K, box = camera['q'][source, v], camera['t'][source, v], camera['k'][source, v], r['bbox']
            sig = max(.5, (1+.03*min(box[2]-box[0], box[3]-box[1])) * noise['clipped' if is_clipped(box) else 'complete'])

            def project(x):
                cc = (pts @ matrix(x[3:]).T + x[:3] - t) @ q
                return (cc[:, :2]/cc[:, 2, None].clip(.5) * [K[0, 0], K[1, 1]]).ravel()

            jac = []
            for axis in range(6):
                delta = np.zeros(6)
                delta[axis] = 1e-4
                jac.append((project(z[i]+delta)-project(z[i]-delta))/2e-4)
            A = np.stack(jac, 1)/sig
            info += A.T @ A/len(ids)
            std = surface['sigma'][k].reshape(-1, 3)[ids].astype(float)
            ps = pixel_sigma(pts, raw['center'][i], raw['rotation'][i], raw['dims'][i], q, t, K, sig, std)
            Ap = np.stack(jac, 1)/ps.ravel()[:, None]
            point_info += Ap.T @ Ap/len(ids)
            used_views += 1
            pixels += len(ids)
            min_size.append(min(box[2]-box[0], box[3]-box[1]))
            clipped += int(is_clipped(box))
        covariance.append(np.linalg.inv(info+np.eye(6)*1e-6)+np.diag(cfg['measurement']))
        point_covariance.append(np.linalg.inv(point_info+np.eye(6)*1e-6)+np.diag(cfg['measurement']))
        qualities.append([used_views, pixels, min(min_size) if min_size else 0, max(min_size) if min_size else 0, clipped])
    return dict(source_indices=ix, z=z, covariance=np.array(covariance),
                point_covariance=np.array(point_covariance), quality=np.array(qualities, dtype=float),
                process=np.array(cfg['process']), measurement_floor=np.array(cfg['measurement']))


def replay(cache, states, dims, covariance_key='covariance', measurement_scale=1., process_scale=1.,
           shock_mode='released', shock_position_gain=.5, shock_structure='diagonal', limit=None):
    """Reproduce released constant-velocity filter, recording internal decisions.

    Velocity units are per observation step, matching the original implementation.
    shock_mode='none' is an ablation; 'released' reproduces its NIS>24 rule.
    """
    if shock_mode not in ('released', 'none'):
        raise ValueError(shock_mode)
    if shock_structure not in ('diagonal', 'rank_one') or shock_position_gain <= 0:
        raise ValueError('Invalid shock covariance specification')
    F = np.eye(12)
    F[:6, 6:12] = np.eye(6)
    G = np.r_[.5*np.eye(6), np.eye(6)]
    process_covariance = cache.get('process_covariance', np.diag(cache['process']))
    Q = G @ (process_covariance*process_scale) @ G.T
    H = np.eye(12)[:6]
    history, output, diagnostics, kept = {}, [], [], []
    for i, source in enumerate(cache['source_indices']):
        row = states[int(source)]
        if limit is not None and row['step'] >= limit:
            continue
        z, R = cache['z'][i], cache[covariance_key][i]*measurement_scale
        prev = history.get(row['object_key'])
        shock, nis = False, 0.
        innovation, predicted = np.zeros(6), z.copy()
        prior_cov = R.copy()
        predicted_state = np.r_[z, np.zeros(6)]
        predicted_full_covariance = np.eye(12)
        predicted_full_covariance[:6, :6] = R
        if prev is None or row['step'] != prev[2]+1:
            state = np.r_[z, np.zeros(6)]
            P = np.eye(12)
            P[:6, :6] = R
            gain = np.r_[np.eye(6), np.zeros((6, 6))]
        else:
            state = F @ prev[0]
            P = F @ prev[1] @ F.T + Q
            predicted, prior_cov = state[:6].copy(), P[:6, :6].copy()
            innovation = z-state[:6]
            innovation[3:] = wrap(innovation[3:])
            S = P[:6, :6]+R
            nis = float(innovation @ np.linalg.solve(S, innovation))
            if nis > 24 and shock_mode == 'released':
                shock = True
                shock_map = np.r_[shock_position_gain*np.eye(6), np.eye(6)]
                shock_cov = innovation_covariance(innovation, S, nis=nis,
                    rule='released' if shock_structure == 'diagonal' else 'directional')
                P += shock_map @ shock_cov @ shock_map.T
                S = P[:6, :6]+R
            gain = np.linalg.solve(S, P[:6, :]).T
            predicted_state = state.copy()
            predicted_full_covariance = P.copy()
            state += gain @ innovation
            A = np.eye(12)-gain @ H
            P = A @ P @ A.T + gain @ R @ gain.T
        history[row['object_key']] = (state.copy(), P.copy(), row['step'])
        output.append(state.copy())
        kept.append(i)
        diagnostics.append(dict(nis=nis, shock=shock, innovation=innovation, predicted=predicted,
                                prior_covariance=prior_cov, posterior_covariance=P[:6, :6], gain=gain[:6],
                                full_covariance=P.copy(), predicted_state=predicted_state,
                                predicted_full_covariance=predicted_full_covariance))
    out = np.array(output)
    rot = matrix(out[:, 3:6])
    geometry = dict(source_indices=cache['source_indices'][kept], center=out[:, :3], rotation=rot,
                    dims=dims[kept], corners=corners(out[:, :3], rot, dims[kept]))
    trace = {k: np.array([d[k] for d in diagnostics]) for k in diagnostics[0]}
    trace['state'] = out
    return geometry, trace


def fixed_lag_geometry(cache, states, dims, lag=1, shock_structure='diagonal', limit=None):
    """RTS-style local retrospective refinement, with explicit lag in frames.

    An output for t may use measurements through t+lag. Never call it zero-delay
    causal filtering. End-of-track outputs have shorter lag. Adaptive shock
    covariances are plug-in estimates, not exact Bayesian posteriors.
    """
    if lag < 0 or not isinstance(lag, int):
        raise ValueError('Lag must be a nonnegative integer')
    geometry, trace = replay(cache, states, dims, shock_structure=shock_structure, limit=limit)
    source = geometry['source_indices']
    rows = [states[int(s)] for s in source]
    indices = {(r['object_key'], r['step']): i for i, r in enumerate(rows)}
    F = np.eye(12)
    F[:6, 6:] = np.eye(6)
    outputs, release_steps = [], []
    for i, row in enumerate(rows):
        chain = [i]
        for offset in range(1, lag+1):
            j = indices.get((row['object_key'], row['step']+offset))
            if j is None:
                break
            chain.append(j)
        value = trace['state'][chain[-1]].copy()
        for k, following in reversed(list(zip(chain[:-1], chain[1:]))):
            cross = trace['full_covariance'][k] @ F.T
            gain = np.linalg.solve(trace['predicted_full_covariance'][following], cross.T).T
            delta = value-trace['predicted_state'][following]
            delta[3:6] = wrap(delta[3:6])
            value = trace['state'][k]+gain @ delta
        outputs.append(value)
        release_steps.append(rows[chain[-1]]['step'])
    output = np.array(outputs)
    rot = matrix(output[:, 3:6])
    geometry.update(center=output[:, :3], rotation=rot,
                    corners=corners(output[:, :3], rot, geometry['dims']))
    return geometry, dict(state=output, released_at_step=np.array(release_steps),
                           used_lag=np.array(release_steps)-np.array([r['step'] for r in rows]))


def replay_imm(cache, states, dims, process_scales=(.1, 1., 100.), persistence=.95,
               covariance_key='covariance', limit=None):
    """Classical interacting CV models with different maneuver process noise.

    Mode likelihood uses the pre-update predictive measurement density. No truth,
    future poses, collision labels, or learned switch thresholds are used.
    This is a baseline, not a novel world-model method or guaranteed uncertainty.
    """
    if not 0 < persistence < 1 or min(process_scales) <= 0:
        raise ValueError('Positive scales and transition probability in (0,1) required')
    n = len(process_scales)
    if n < 2:
        raise ValueError('IMM requires at least two modes')
    transition = np.full((n, n), (1-persistence)/(n-1))
    np.fill_diagonal(transition, persistence)
    F = np.eye(12)
    F[:6, 6:] = np.eye(6)
    G = np.r_[.5*np.eye(6), np.eye(6)]
    H = np.eye(12)[:6]
    qs = [G @ np.diag(cache['process']*s) @ G.T for s in process_scales]
    history, outputs, mode_history, covariances, kept = {}, [], [], [], []
    for i, source in enumerate(cache['source_indices']):
        row = states[int(source)]
        if limit is not None and row['step'] >= limit:
            continue
        z, R = cache['z'][i], cache[covariance_key][i]
        prev = history.get(row['object_key'])
        if prev is None or row['step'] != prev[3]+1:
            xs = np.tile(np.r_[z, np.zeros(6)], (n, 1))
            ps = np.tile(np.eye(12), (n, 1, 1))
            ps[:, :6, :6] = R
            mu = np.full(n, 1/n)
        else:
            old_x, old_p, old_mu, _ = prev
            # Align each mode to the first mode's Euler branch before mixing.
            aligned = old_x.copy()
            aligned[:, 3:6] = old_x[0, 3:6]+wrap(old_x[:, 3:6]-old_x[0, 3:6])
            prior = old_mu @ transition
            mixing = old_mu[:, None]*transition/prior[None]
            xs, ps, logs = [], [], []
            for j in range(n):
                x = mixing[:, j] @ aligned
                diff = aligned-x
                P = np.sum(mixing[:, j, None, None]*(old_p+diff[:, :, None]*diff[:, None, :]), axis=0)
                x = F @ x
                P = F @ P @ F.T + qs[j]
                innovation = z-x[:6]
                innovation[3:] = wrap(innovation[3:])
                S = P[:6, :6]+R
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    raise ValueError('Nonpositive predictive covariance')
                logs.append(np.log(prior[j])-.5*(logdet+innovation @ np.linalg.solve(S, innovation)))
                K = np.linalg.solve(S, P[:6]).T
                x += K @ innovation
                A = np.eye(12)-K @ H
                P = A @ P @ A.T + K @ R @ K.T
                xs.append(x)
                ps.append(P)
            xs, ps = np.array(xs), np.array(ps)
            weights = np.exp(np.array(logs)-max(logs))
            mu = weights/weights.sum()
        aligned = xs.copy()
        aligned[:, 3:6] = xs[0, 3:6]+wrap(xs[:, 3:6]-xs[0, 3:6])
        mean = mu @ aligned
        diff = aligned-mean
        cov = np.sum(mu[:, None, None]*(ps+diff[:, :, None]*diff[:, None, :]), axis=0)
        history[row['object_key']] = (xs, ps, mu, row['step'])
        outputs.append(mean)
        mode_history.append(mu)
        covariances.append(cov[:6, :6])
        kept.append(i)
    out = np.array(outputs)
    rot = matrix(out[:, 3:6])
    geometry = dict(source_indices=cache['source_indices'][kept], center=out[:, :3], rotation=rot,
                    dims=dims[kept], corners=corners(out[:, :3], rot, dims[kept]))
    return geometry, dict(state=out, mode_probabilities=np.array(mode_history), posterior_covariance=np.array(covariances))
