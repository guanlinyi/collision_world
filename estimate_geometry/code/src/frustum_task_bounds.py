"""Conditional separation certificates from masks, cameras, and enclosing radii.

No learned correspondences, orientation, initial 3D proposal, reference pose, or
image appearance is required. The premise is that each used native mask pixel
belongs to this body and that the body lies in a ball of the given radius about
its center. Nominal calibration must be valid. Results are conditional geometric
certificates, NOT calibrated guarantees for noisy masks or unknown categories.
"""
import numpy as np
from scipy.optimize import linprog, minimize


def center_outer_polytope(views,radius_m,pixel_half_width=1.,translation_error_m=0.,rotation_error_rad=0.):
    """Outer center constraints, A c <= b, from extrema of actual mask pixels.

    A pixel cell defines a cone from the camera. A body point in that cone and
    within radius r of c implies n.c <= n.t+r||n|| for every cone halfspace n.
    Keeping only extrema weakens constraints safely; no pixel is invented.
    One-pixel half-width covers the two pixel-center conventions under audit;
    it is an explicit premise, not an estimated mask-error distribution.
    """
    if not np.isfinite(radius_m) or radius_m<=0:raise ValueError('Positive finite radius required')
    if not np.isfinite(pixel_half_width) or pixel_half_width<0:raise ValueError('Nonnegative finite pixel bound required')
    if not np.isfinite(translation_error_m) or translation_error_m<0:raise ValueError('Nonnegative translation bound required')
    if not np.isfinite(rotation_error_rad) or not 0<=rotation_error_rad<=np.pi:raise ValueError('Rotation bound must lie in [0,pi]')
    normal_change=2*np.sin(rotation_error_rad/2)
    A=[];b=[];absolute_bounds=[]
    for view in views:
        uv=np.asarray(view['pixels'],float);q=np.asarray(view['q'],float);t=np.asarray(view['t'],float);k=np.asarray(view['k'],float)
        if uv.ndim!=2 or uv.shape[1]!=2 or not len(uv):continue
        if q.shape!=(3,3) or t.shape!=(3,) or k.shape!=(3,3):raise ValueError('Camera shape mismatch')
        if not all(np.isfinite(x).all() for x in (uv,q,t,k)):raise ValueError('Nonfinite camera/pixel data')
        if not np.allclose(q.T@q,np.eye(3),atol=1e-5) or np.linalg.det(q)<=0:raise ValueError('Proper camera rotation required')
        if k[0,0]<=0 or k[1,1]<=0:raise ValueError('Positive focal lengths required')
        # An analytic finite coordinate bound supplies a noncircular error
        # allowance for LP dual rounding. For two separated pixel columns,
        # adding opposite cone inequalities bounds center depth from above.
        radius_camera=radius_m*np.linalg.norm(q,2)
        low=[];high=[];depth_upper=[]
        for axis in (0,1):
            lo=(uv[:,axis].max()-pixel_half_width-k[axis,2])/k[axis,axis]
            hi=(uv[:,axis].min()+pixel_half_width-k[axis,2])/k[axis,axis]
            low.append(lo);high.append(hi)
            if lo>hi:depth_upper.append(radius_camera*(np.hypot(1,lo)+np.hypot(1,hi))/(lo-hi))
        range_upper=None
        if depth_upper:
            depths=np.array([-radius_camera,min(depth_upper)])
            bounds_camera=[]
            for lo,hi in zip(low,high):
                lower=np.min(lo*depths)-radius_camera*np.hypot(1,lo)
                upper=np.max(hi*depths)+radius_camera*np.hypot(1,hi)
                bounds_camera.append(max(abs(lower),abs(upper)))
            bounds_camera.append(float(np.max(np.abs(depths))))
            inverse=np.linalg.inv(q.T)
            # Spectral norm is invariant under a common world rotation. An
            # absolute-entry matrix norm here would add coordinate-dependent
            # conservatism to the allowed calibration error.
            range_upper=float(np.linalg.norm(inverse,2)*np.linalg.norm(bounds_camera))
            extra=translation_error_m+normal_change*range_upper
            expanded_camera=np.asarray(bounds_camera)+np.linalg.norm(q,axis=0)*extra
            bound=np.abs(t)+np.abs(inverse)@expanded_camera
            if np.isfinite(bound).all():absolute_bounds.append(bound*(1+1e-10)+1e-5)
            # These camera-oriented planes transform with the cameras. They
            # both contain every admissible true center and make the later
            # coordinate error bound valid for the relaxed LP feasible set.
            for j in range(3):
                for sign in (-1,1):
                    n=sign*q[:,j];norm=np.linalg.norm(n);n=n/norm
                    A.append(n);b.append(float(n@t+expanded_camera[j]/norm))
        if normal_change>0 and range_upper is None:
            # Without a finite range bound, an angular error cannot be changed
            # into a finite metric slack. Drop this view conservatively.
            continue
        normals=[np.array([0.,0.,-1.])]
        for axis in (0,1):
            for pixel in np.unique([uv[:,axis].min(),uv[:,axis].max()]):
                low=(pixel-pixel_half_width-k[axis,2])/k[axis,axis]
                high=(pixel+pixel_half_width-k[axis,2])/k[axis,axis]
                left=np.zeros(3);left[axis]=-1;left[2]=low
                right=np.zeros(3);right[axis]=1;right[2]=-high
                normals.extend([left,right])
        for n in normals:
            normal=q@n;normal/=np.linalg.norm(normal)
            slack=radius_m+translation_error_m+normal_change*(range_upper or 0.)
            A.append(normal);b.append(float(normal@t+slack))
    if not A:return dict(status='no_observation',A=np.empty((0,3)),b=np.empty(0),radius=radius_m)
    coordinate_bounds=np.min(absolute_bounds,axis=0) if absolute_bounds else None
    A=np.asarray(A);b=np.asarray(b)
    feasibility=linprog(np.zeros(3),A_ub=A,b_ub=b,bounds=[(None,None)]*3,method='highs')
    if not feasibility.success:
        return dict(status='inconsistent' if feasibility.status==2 else 'solver_failure',A=A,b=b,radius=radius_m,solver_status=int(feasibility.status))
    return dict(status='feasible',A=A,b=b,radius=radius_m,witness=feasibility.x,
        coordinate_abs_bounds=coordinate_bounds)


def support_bound(polytope,direction):
    """Upper support value with a checked LP dual witness.

    For max n.c subject to A c<=b, a nonnegative y with A^T y=n gives
    upper bound b^T y. Floating residual is checked; conservative rounding
    allowance is attached using independently computed coordinate bounds.
    """
    n=np.asarray(direction,float)
    fit=linprog(-n,A_ub=polytope['A'],b_ub=polytope['b'],bounds=[(None,None)]*3,method='highs')
    if not fit.success:return None
    dual=-np.asarray(fit.ineqlin.marginals)
    if np.min(dual)<-1e-10:return None
    dual=np.maximum(dual,0.)
    residual=n-polytope['A'].T@dual
    # Bound residual dot c by a validated finite coordinate box.
    bounds=polytope.get('coordinate_abs_bounds')
    if bounds is None:return None
    upper=float(polytope['b']@dual+np.abs(residual)@bounds+1e-5)
    return dict(upper=upper,dual=dual,dual_residual=float(np.max(np.abs(residual))),primal=fit.x)


def separated_by_bounds(a,b,extra_direction=None):
    """Return separated only after a verified linear support gap exceeds radii.

    The optional convex nearest-point optimizer proposes an axis; the actual
    decision always comes from independent LP support bounds. Failed or
    unbounded solves produce unknown, never a positive contact judgment.
    """
    if a['status']!='feasible' or b['status']!='feasible':return dict(status='unknown',reason='Unavailable or inconsistent observation set')
    directions=list(np.eye(3))
    if extra_direction is not None:directions.append(np.asarray(extra_direction,float))
    directions.append(b['witness']-a['witness'])
    # A nearest-center pair often supplies a stronger task direction than axes.
    A=np.zeros((len(a['A'])+len(b['A']),6));A[:len(a['A']),:3]=a['A'];A[len(a['A']):,3:]=b['A']
    rhs=np.r_[a['b'],b['b']]
    start=np.r_[a['witness'],b['witness']]
    def objective(x):d=x[3:]-x[:3];return float(d@d)
    def gradient(x):d=x[3:]-x[:3];return np.r_[-2*d,2*d]
    fit=minimize(objective,start,jac=gradient,method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:rhs-A@x,jac=lambda x:-A)],
        options=dict(maxiter=100,ftol=1e-9))
    if np.isfinite(fit.x).all():directions.append(fit.x[3:]-fit.x[:3])
    best=None
    for vector in directions:
        if not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-10:continue
        vector=vector/np.linalg.norm(vector)
        for sign in (1,-1):
            n=vector*sign;ua=support_bound(a,n);nb=support_bound(b,-n)
            if ua is None or nb is None:continue
            gap=-nb['upper']-ua['upper']-a['radius']-b['radius']
            if best is None or gap>best['gap_lower_m']:
                best=dict(gap_lower_m=float(gap),axis=n.tolist(),dual_residual_max=max(ua['dual_residual'],nb['dual_residual']))
    if best is None:return dict(status='unknown',reason='No bounded certified task axis')
    return dict(status='separated' if best['gap_lower_m']>1e-4 else 'unknown',
        reason='Outer enclosing balls strictly separated' if best['gap_lower_m']>1e-4 else 'Outer sets do not certify separation',**best)
