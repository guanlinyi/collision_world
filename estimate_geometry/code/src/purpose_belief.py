"""Purpose-driven set reasoning for calibrated, labelled rigid-marker cameras.

Scope: two convex cuboids with GIVEN dimensions/rotations, unknown 3D centers,
bounded 2D marker errors, optional unknown common camera translation. This is a
geometric reasoning laboratory, NOT the learned RGB predictor. No true centers
are accepted by the belief or view-selection functions.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Camera:
    center: np.ndarray
    rotation: np.ndarray  # columns: camera x/right, y/down, z/forward in world
    focal: float = 400.
    principal: tuple = (400., 300.)


def camera_looking_at(center, target=(0., 0., 0.)):
    center, target = np.array(center, float), np.array(target, float)
    forward = target-center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return Camera(center, np.stack([right, down, forward], axis=1))


def marker_offsets(dims, rotation):
    signs = np.array(list(itertools.product([-1., 1.], repeat=3)))
    return (signs*np.asarray(dims)/2) @ np.asarray(rotation).T


def project(camera, centers, offsets, camera_shift=None):
    shift = np.zeros(3) if camera_shift is None else np.asarray(camera_shift)
    cc = (np.asarray(centers)[:, None]+offsets-camera.center-shift) @ camera.rotation
    if np.any(cc[..., 2] <= .1):
        raise ValueError('Markers must be in front of camera')
    return cc[..., :2]/cc[..., 2, None]*camera.focal+np.asarray(camera.principal)


class CenterBelief:
    """Intersection of linear constraints induced by bounded pinhole pixels.

    The joint camera-shift variable preserves common calibration uncertainty;
    replacing it by independent object padding would lose this correlation.
    """
    def __init__(self, dims, rotations, center_bounds, camera_shift_bound=0., pixel_bound=2.):
        self.dims = np.asarray(dims, float)
        self.rotations = np.asarray(rotations, float)
        if self.dims.shape != (2, 3) or self.rotations.shape != (2, 3, 3):
            raise ValueError('Exactly two cuboids required')
        if np.any(self.dims <= 0) or not np.isfinite(self.dims).all():
            raise ValueError('Positive finite dimensions required')
        for r in self.rotations:
            if not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or np.linalg.det(r) < 0:
                raise ValueError('Proper orthonormal rotations required')
        if not np.isfinite([pixel_bound, camera_shift_bound]).all() or pixel_bound < 0 or camera_shift_bound < 0:
            raise ValueError('Nonnegative observation bounds required')
        self.offsets = np.array([marker_offsets(d, r) for d, r in zip(self.dims, self.rotations)])
        self.pixel_bound = float(pixel_bound)
        self.bounds = [tuple(map(float, b)) for b in center_bounds]
        if len(self.bounds) != 6 or not np.isfinite(self.bounds).all() or any(lo >= hi for lo, hi in self.bounds):
            raise ValueError('Six ordered center bounds required')
        self.bounds += [(-float(camera_shift_bound), float(camera_shift_bound))]*3
        self.A, self.b = [], []
        self.views = []
        self.axes, self.radii = self._contact_axes()

    def _contact_axes(self):
        a, b = self.rotations
        candidates = [a[:, i] for i in range(3)]+[b[:, i] for i in range(3)]
        candidates += [np.cross(a[:, i], b[:, j]) for i, j in itertools.product(range(3), repeat=2)]
        axes = []
        for axis in candidates:
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis = axis/norm
            if any(abs(axis @ old) > 1-1e-10 for old in axes):
                continue
            axes.append(axis)
        radii = [sum(np.sum(d/2*np.abs(r.T @ axis)) for d, r in zip(self.dims, self.rotations)) for axis in axes]
        return np.array(axes), np.array(radii)

    def add_view(self, name, camera, pixels):
        pixels = np.asarray(pixels, float)
        if pixels.shape != (2, 8, 2) or not np.isfinite(pixels).all():
            raise ValueError('Two sets of eight finite labelled marker pixels required')
        if name in self.views:
            raise ValueError('Do not count the same image twice')
        for obj, offset in itertools.product(range(2), range(8)):
            shift = self.offsets[obj, offset]-camera.center
            for dim in range(2):
                low, high = pixels[obj, offset, dim]+np.array([-1., 1.])*self.pixel_bound-camera.principal[dim]
                for normal in [camera.focal*camera.rotation[:, dim]-high*camera.rotation[:, 2],
                               low*camera.rotation[:, 2]-camera.focal*camera.rotation[:, dim]]:
                    a = np.zeros(9)
                    a[obj*3:obj*3+3] = normal
                    a[6:] = -normal
                    scale = max(np.linalg.norm(normal), 1.)
                    self.A.append(a/scale)
                    self.b.append(float(-normal @ shift)/scale)
            a = np.zeros(9)
            a[obj*3:obj*3+3] = -camera.rotation[:, 2]
            a[6:] = camera.rotation[:, 2]
            self.A.append(a)
            self.b.append(float(camera.rotation[:, 2] @ shift-.1))
        self.views.append(name)

    def _solve(self, objective=None, extra_a=(), extra_b=()):
        aa, bb = self.A+list(extra_a), self.b+list(extra_b)
        result = linprog(np.zeros(9) if objective is None else objective,
                         A_ub=np.array(aa) if aa else None, b_ub=np.array(bb) if bb else None,
                         bounds=self.bounds, method='highs',
                         options={'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
        if result.status not in (0, 2):
            raise RuntimeError('Belief LP failed: '+result.message)
        return result.x if result.success else None

    def contains(self, centers, camera_shift=None, tolerance=1e-7):
        x = np.r_[np.asarray(centers).ravel(), np.zeros(3) if camera_shift is None else camera_shift]
        return bool(all(lo-tolerance <= value <= hi+tolerance for value, (lo, hi) in zip(x, self.bounds))
                    and (not self.A or np.all(np.asarray(self.A) @ x <= np.asarray(self.b)+tolerance)))

    def query(self, purpose='contact', precision_m=.1, guard=1e-7):
        if purpose not in ('contact', 'absolute_position', 'relative_position'):
            raise ValueError('Unknown purpose')
        if not np.isfinite([precision_m, guard]).all() or precision_m <= 0 or guard < 1e-7:
            raise ValueError('Finite positive precision and numerical guard >= 1e-7 required')
        feasible = self._solve()
        if feasible is None:
            return dict(decision='inconsistent', witnesses=[], views_used=len(self.views))
        if purpose == 'contact':
            a, b = [], []
            for axis, radius in zip(self.axes, self.radii):
                for sign in [-1, 1]:
                    a.append(np.r_[-sign*axis, sign*axis, np.zeros(3)])
                    b.append(float(radius))
            contact = self._solve(extra_a=a, extra_b=np.array(b)+guard)
            if contact is None:
                return dict(decision='separated', witnesses=[feasible.tolist()], views_used=len(self.views))
            for normal, radius in zip(a, b):
                witness = self._solve(-normal)
                if normal @ witness > radius-guard:
                    return dict(decision='unknown', witnesses=[contact.tolist(), witness.tolist()], views_used=len(self.views))
            return dict(decision='overlap', witnesses=[feasible.tolist()], views_used=len(self.views))
        if purpose not in ('absolute_position', 'relative_position') or precision_m <= 0:
            raise ValueError('Unknown purpose or nonpositive precision')
        objectives = []
        if purpose == 'absolute_position':
            for i in range(6):
                v = np.zeros(9)
                v[i] = 1
                objectives.append(v)
        else:
            for i in range(3):
                v = np.zeros(9)
                v[i], v[i+3] = -1, 1
                objectives.append(v)
        widest, witnesses = -1., []
        for normal in objectives:
            low, high = self._solve(normal), self._solve(-normal)
            width = float(normal @ (high-low))
            if width > widest:
                widest, witnesses = width, [low.tolist(), high.tolist()]
        return dict(decision='precise' if widest <= precision_m else 'unknown',
                    max_coordinate_width_m=widest, witnesses=witnesses, views_used=len(self.views))

    def choose_view(self, cameras, query, available):
        """Choose a sensor that best separates two still-possible task answers.

        Uses only camera geometry and feasible hypotheses, never unseen pixels.
        It is a greedy witness heuristic, not an optimal information-gain proof.
        """
        if not available:
            return None
        witnesses = query.get('witnesses', [])
        if len(witnesses) < 2:
            return available[0]
        left, right = map(np.asarray, witnesses[:2])
        scores = []
        for name in available:
            a = project(cameras[name], left[:6].reshape(2, 3), self.offsets, left[6:])
            b = project(cameras[name], right[:6].reshape(2, 3), self.offsets, right[6:])
            scores.append(float(np.max(np.linalg.norm(a-b, axis=-1))))
        return available[int(np.argmax(scores))]
