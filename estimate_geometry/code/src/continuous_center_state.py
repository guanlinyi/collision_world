"""Frozen experimental CV/CA center estimator, independent of teachers and files.

All derivatives are SI quantities at the box center. Interval estimates use
the joint posterior available now; previous published positions stay immutable.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


def transition(h, order):
    if not np.isfinite(h) or h < 0 or order not in (2, 3):
        raise ValueError('Finite nonnegative seconds and order 2/3 required')
    f = np.eye(order)
    f[0, 1] = h
    if order == 3:
        f[0, 2], f[1, 2] = h*h/2, h
    return np.kron(f, np.eye(3))


def process_covariance(h, q, order):
    if not np.isfinite(q) or q < 0:
        raise ValueError('Nonnegative finite process spectral density required')
    transition(h, order)  # also validate time/order
    if order == 3:
        block = [[h**5/20, h**4/8, h**3/6],
                 [h**4/8, h**3/3, h*h/2], [h**3/6, h*h/2, h]]
    else:
        block = [[h**3/3, h*h/2], [h*h/2, h]]
    return q*np.kron(block, np.eye(3))


def symmetric(a):
    return (a+a.T)/2


def positive_factor(a, context):
    try:
        return cho_factor(a, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError(f'{context}: min_eigenvalue={np.linalg.eigvalsh(a).min()}; '
                         f'covariance={a.tolist()}') from error


def predict(raw, times, keys, observation_covariance, frozen_config):
    """No teacher inputs. raw contains center, source_indices, step only.

    Entries must arrive in increasing time within each identity. Different
    identities can interleave freely. Step gaps reset, matching the baseline.
    """
    z = np.asarray(raw['center'], dtype=float)
    times = np.asarray(times, dtype=float)
    source = np.asarray(raw['source_indices'])
    steps = np.asarray(raw['step'])
    rs = np.asarray(observation_covariance, dtype=float)
    n = len(z)
    if z.shape != (n, 3) or rs.shape != (n, 3, 3):
        raise ValueError('Position/covariance shape contract')
    if any(len(v) != n for v in (times, source, steps, keys)):
        raise ValueError('Length contract')
    if not np.isfinite(z).all() or not np.isfinite(times).all() or not np.isfinite(rs).all():
        raise ValueError('Finite observation/time/covariance required')
    if len(set(source.tolist())) != n or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError('Unique sources and explicit nonempty identities required')
    order = int(frozen_config['order'])
    q = float(frozen_config['q'])
    size = order*3
    transition(0., order)
    process_covariance(0., q, order)
    sigmas = [float(frozen_config['sigma_velocity'])]
    if order == 3:
        sigmas.append(float(frozen_config['sigma_acceleration']))
    if min(sigmas) <= 0 or not np.isfinite(sigmas).all():
        raise ValueError('Positive finite initialization scales required')
    hobs = np.eye(size)[:3]
    threshold = float(chi2.ppf(.999, 3))
    fields = {k: [] for k in ('state', 'covariance', 'prior_state', 'prior_covariance',
                             'innovation', 'innovation_covariance', 'gain',
                             'previous_conditioned_state', 'previous_conditioned_covariance',
                             'previous_current_covariance', 'interval_velocity',
                             'interval_velocity_covariance', 'published_velocity',
                             'revision_previous', 'correction', 'nis', 'nll',
                             'motion_known', 'conflict', 'interval_start', 'dt')}
    history = {}
    for i in range(n):
        R = symmetric(rs[i])
        positive_factor(R, f'R source={source[i]}')
        old = history.get(keys[i])
        if old is not None and (times[i] <= old[3] or steps[i] <= old[2]):
            raise ValueError(f'Nonincreasing time/step for {keys[i]} at {source[i]}')
        known = old is not None and steps[i] == old[2]+1
        if not known:
            mu = np.r_[z[i], np.zeros(size-3)]
            P = np.zeros((size, size)); P[:3, :3] = R
            for j, sigma in enumerate(sigmas, 1):
                P[3*j:3*j+3, 3*j:3*j+3] = np.eye(3)*sigma*sigma
            prior, prior_P = mu.copy(), P.copy()
            prev, prev_P, cross = mu.copy(), P.copy(), P.copy()
            innovation = np.zeros(3); S = R.copy(); K = np.zeros((size, 3))
            interval = published = revision = np.zeros(3)
            interval_P = np.zeros((3, 3))
            nis = nll = dt = 0.; start = times[i]
        else:
            mu0, P0, _, start = old
            dt = times[i]-start
            F = transition(dt, order)
            prior = F@mu0
            prior_P = symmetric(F@P0@F.T+process_covariance(dt, q, order))
            innovation = z[i]-prior[:3]
            S = symmetric(prior_P[:3, :3]+R)
            factor = positive_factor(S, f'S source={source[i]}')
            K = cho_solve(factor, prior_P[:3]).T
            mu = prior+K@innovation
            A = np.eye(size)-K@hobs
            P = symmetric(A@prior_P@A.T+K@R@K.T)
            L = cho_solve(factor, (P0@F.T)[:,:3].T).T
            prev = mu0+L@innovation
            prev_P = symmetric(P0-L@S@L.T)
            cross = P0@F.T-L@S@K.T
            interval = (mu[:3]-prev[:3])/dt
            published = (mu[:3]-mu0[:3])/dt
            revision = prev[:3]-mu0[:3]
            interval_P = symmetric((P[:3,:3]+prev_P[:3,:3]-cross[:3,:3]-cross[:3,:3].T)/(dt*dt))
            nis = float(innovation@cho_solve(factor, innovation))
            logdet = 2*np.log(np.diag(factor[0])).sum()
            nll = float(.5*(nis+logdet+3*np.log(2*np.pi)))
        history[keys[i]] = (mu.copy(), P.copy(), int(steps[i]), float(times[i]))
        values = dict(state=mu, covariance=P, prior_state=prior, prior_covariance=prior_P,
                      innovation=innovation, innovation_covariance=S, gain=K,
                      previous_conditioned_state=prev, previous_conditioned_covariance=prev_P,
                      previous_current_covariance=cross, interval_velocity=interval,
                      interval_velocity_covariance=interval_P, published_velocity=published,
                      revision_previous=revision, correction=mu-prior, nis=nis, nll=nll,
                      motion_known=known, conflict=known and nis>threshold,
                      interval_start=start, dt=dt)
        for name, value in values.items():
            fields[name].append(value)
    result = {name: np.asarray(values) for name, values in fields.items()}
    result.update(source_indices=source.copy(), estimate_at_time=times.copy(),
                  available_at_time=times.copy(), interval_end=times.copy())
    return result


def conditional_bridge(tau, h, mu0, P0, innovation, observation_covariance, config):
    """Current-data conditional trajectory. No historical output is overwritten."""
    if not 0 <= tau <= h:
        raise ValueError('Bridge time outside interval')
    order, q = config['order'], config['q']
    Fh, Ft = transition(h, order), transition(tau, order)
    S = (Fh@P0@Fh.T+process_covariance(h, q, order))[:3,:3]+observation_covariance
    cross = Ft@P0@Fh.T+process_covariance(tau, q, order)@transition(h-tau, order).T
    return Ft@mu0+cross[:,:3]@cho_solve(positive_factor(S, 'bridge'), innovation)


def forecast(state, horizon, order):
    return (transition(horizon, order)@state)[:3]
