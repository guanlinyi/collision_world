"""Carry image-informed current pose and drift in one causal state.

The published past is immutable. Observation caches and image-motion support
match the released baseline. Image updates change the 12D mean only; covariance
is retained because the same RGB also formed the single-frame observation.
The final six components remain per-step latent drift, not metric velocity.
"""
import numpy as np
from .state import matrix, wrap
from .geometry import corners
from .motion_innovation import innovation_covariance
from .image_constrained_pose import correct_pose_checked
from .center_motion_checked import correct_center_checked


def replay_integrated(cache, states, dims, measurements, camera, image_enabled=True,
                      limit=None, transport_drift=True, image_target='pose', allow_angular_drift=False):
    """Pure numerical replay; no files, truth, training or history revision."""
    if image_target not in ('pose', 'center'):
        raise ValueError('Unknown image target')
    F = np.eye(12)
    F[:6, 6:] = np.eye(6)
    G = np.r_[.5*np.eye(6), np.eye(6)]
    Q = G @ np.diag(cache['process']) @ G.T
    H = np.eye(12)[:6]
    history, output, kept, details, covariances = {}, [], [], [], []
    for i, source in enumerate(cache['source_indices']):
        source = int(source)
        row = states[source]
        if limit is not None and row['step'] >= limit:
            continue
        z, R = cache['z'][i], cache['covariance'][i]
        previous = history.get(row['object_key'])
        consecutive = previous is not None and row['step'] == previous['step'] + 1
        nis, shock = 0., False
        if not consecutive:
            state = np.r_[z, np.zeros(6)]
            P = np.eye(12)
            P[:6, :6] = R
        else:
            state = F @ previous['state']
            P = F @ previous['covariance'] @ F.T + Q
            innovation = z - state[:6]
            innovation[3:] = wrap(innovation[3:])
            S = P[:6, :6] + R
            nis = float(innovation @ np.linalg.solve(S, innovation))
            if nis > 24:
                shock = True
                P += G @ innovation_covariance(innovation, S, rule='released') @ G.T
                S = P[:6, :6] + R
            K = np.linalg.solve(S, P[:6, :]).T
            state += K @ innovation
            A = np.eye(12) - K @ H
            P = A @ P @ A.T + K @ R @ K.T
        visual = state.copy()
        detail = dict(status='image_disabled' if not image_enabled else 'unknown_no_tracks',
                      views=0, tracks=0, iterations=0)
        if image_enabled and consecutive:
            views = []
            for m in measurements.get(source, []):
                if m['source0'] != previous['source']:
                    continue
                views.append(dict(local=m['local'], displacement=m['displacement'],
                    sigma_native_px=m['sigma_native_px'],
                    **{key+str(n): camera[key][s, m['view']]
                       for n, s in [(0, m['source0']), (1, source)] for key in ['q', 't', 'k']}))
            if image_target == 'center':
                # Same Gaussian prior, restricted mean coordinates. Conditioning
                # is a parameter restriction, not a claim that angles are known.
                active = np.array([0,1,2,6,7,8] + ([9,10,11] if allow_angular_drift else []))
                fixed = np.array([j for j in range(12) if j not in active])
                Pa, Pf = P[np.ix_(active,active)], P[np.ix_(fixed,fixed)]
                cross = P[np.ix_(active,fixed)]
                reduced = Pa - cross @ np.linalg.solve(Pf, cross.T)
                reduced = (reduced + reduced.T)/2
                pose, detail = correct_center_checked(visual[:6], reduced[:3,:3],
                                                      previous['state'][:6], views)
                delta = pose[:3] - visual[:3]
                if transport_drift:
                    conditional = np.linalg.solve(reduced[:3,:3], reduced[:3,:]).T
                    state[active] += conditional @ delta
                else:
                    state[:3] += delta
            else:
                pose, detail = correct_pose_checked(visual[:6], P[:6, :6],
                                                   previous['state'][:6], views)
                delta = pose - visual[:6]
                delta[3:] = wrap(delta[3:])
                if transport_drift:
                    conditional = np.linalg.solve(P[:6, :6], P[:6, :]).T
                    state += conditional @ delta
                else:
                    state[:6] += delta
            assert np.max(np.abs(state[:3] - pose[:3])) < 1e-9
            assert np.max(np.abs(wrap(state[3:6] - pose[3:6]))) < 1e-9
        detail.update(source_index=source, episode=row['episode'], step=row['step'],
            object_key=row['object_key'], nis=nis, shock=shock,
            drift_correction_per_step=float(np.linalg.norm(state[6:9]-visual[6:9])),
            angular_drift_correction_per_step=float(np.linalg.norm(state[9:12]-visual[9:12])),
            own_visual_center=visual[:3].tolist(), image_covariance_reduction=False,
            image_target=image_target, allow_angular_drift=allow_angular_drift)
        # Each published array is copied before retaining a separate live state.
        output.append(state.copy())
        covariances.append(P.copy())
        details.append(detail)
        kept.append(i)
        history[row['object_key']] = dict(state=state.copy(), covariance=P.copy(),
                                         step=row['step'], source=source)
    output = np.array(output)
    rot = matrix(output[:, 3:6])
    selected_dims = dims[kept]
    geometry = dict(source_indices=cache['source_indices'][kept], center=output[:, :3],
                    rotation=rot, dims=selected_dims,
                    corners=corners(output[:, :3], rot, selected_dims))
    return geometry, dict(state=output, covariance=np.array(covariances)), details
