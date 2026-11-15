"""Local projected polygon-union tangent cones plus same-q camera-z proof."""
import numpy as np

PIXEL_TOL = 1e-7
GUARD = 128*np.finfo(np.float64).eps


def cone_intervals(cones):
    if any(not len(cone) for cone in cones): return 'full', [], None
    if not cones: return 'unknown', [], None
    normals = np.concatenate(cones)
    angles = np.mod(np.r_[np.arctan2(normals[:, 1], normals[:, 0])+np.pi/2,
                          np.arctan2(normals[:, 1], normals[:, 0])+3*np.pi/2], 2*np.pi)
    ordered = np.sort(angles)
    events = [float(ordered[0])]
    for value in ordered[1:]:
        if value-events[-1] > GUARD: events.append(float(value))
    if len(events)>1 and events[0]+2*np.pi-events[-1] <= GUARD: events.pop()
    intervals = []
    for index, start in enumerate(events):
        stop = events[(index+1)%len(events)]+(2*np.pi if index==len(events)-1 else 0)
        if stop-start <= GUARD: continue
        mid = (start+stop)/2
        direction = np.array([np.cos(mid), np.sin(mid)])
        statuses = []
        for cone in cones:
            dot = cone@direction
            statuses.append(False if np.any(dot < -GUARD) else (None if np.any(abs(dot)<=GUARD) else True))
        covered = True if True in statuses else (None if None in statuses else False)
        intervals.append(dict(start_rad=start, stop_rad=stop, covered=covered))
    flags = [interval['covered'] for interval in intervals]
    if None in flags: return 'unknown', intervals, None
    if flags and all(flags): return 'full', intervals, None
    if not any(flags): return 'unknown', intervals, None
    return 'boundary', intervals, next(interval for interval in intervals if not interval['covered'])


def certify(projected, q, z_edge, source_edge_vertices, edge_t, triangle_vertex_ids, exact_incident_depth=False):
    ids = projected.bins.get(tuple(np.floor(q/projected.bin_px).astype(int)), np.empty(0, int))
    ids = ids[np.all(q >= projected.lo[ids]-PIXEL_TOL, axis=1)&np.all(q <= projected.hi[ids]+PIXEL_TOL, axis=1)]
    tri = projected.triangles[ids]
    vector = np.roll(tri, -1, axis=1)-tri
    length = np.linalg.norm(vector, axis=2)
    area = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    inward = np.stack([-vector[:, :, 1], vector[:, :, 0]], 2)*np.sign(area)[:, None, None]/length[:, :, None]
    signed = np.sum((q-tri)*inward, axis=2)
    cover = np.all(signed >= -PIXEL_TOL, axis=1)
    ids, tri, inward, signed = ids[cover], tri[cover], inward[cover], signed[cover]
    evidence = dict(covering_face_ids=ids.tolist(), signed_edge_distances_px=signed.tolist(),
        q_xy=q.tolist(), edge_camera_z=float(z_edge), guard=GUARD)
    if not len(ids): return 'numerical_unknown_no_closed_cover', evidence
    # Same-query front depth is independent of the cone decision and can certify an occluded source.
    bary = projected.barycentric(ids, q)
    z = 1/np.sum(bary/projected.z[ids], axis=1)
    structural = np.zeros(len(ids), bool)
    if exact_incident_depth:
        # A known original 3-D edge parameter determines its incident-face
        # barycentric coordinates without an ill-conditioned 2-D inverse.
        for i,face in enumerate(ids):
            vertices = triangle_vertex_ids[face]
            beta = np.zeros(3)
            if edge_t == 0 or edge_t == 1:
                vertex = source_edge_vertices[int(edge_t)]
                matches = np.where(vertices == vertex)[0]
                if len(matches) != 1: continue
                beta[matches[0]] = 1.
            else:
                left = np.where(vertices == source_edge_vertices[0])[0]
                right = np.where(vertices == source_edge_vertices[1])[0]
                if len(left) != 1 or len(right) != 1: continue
                beta[left[0]], beta[right[0]] = 1-edge_t, edge_t
            bary[i] = beta*projected.z[face]/z_edge
            z[i] = z_edge
            structural[i] = True
    if not np.isfinite(z).all() or np.any(z <= 0): return 'numerical_unknown_depth', evidence
    front = float(np.min(z)); tol = 32*np.finfo(np.float64).eps*max(1, abs(front), abs(z_edge))
    gap = abs(front-z_edge)
    extrapolated = np.any(bary < -GUARD, axis=1) & ~structural
    trusted = z[~extrapolated]
    extrapolation_changes = False
    if extrapolated.any():
        if not len(trusted): extrapolation_changes = True
        else:
            trusted_front = float(np.min(trusted))
            trusted_tol = 32*np.finfo(np.float64).eps*max(1, abs(trusted_front), abs(z_edge))
            extrapolation_changes = (gap <= tol) != (abs(trusted_front-z_edge) <= trusted_tol)
    evidence.update(front_camera_z=front, depth_gap=float(gap), depth_tolerance=float(tol),
        nearest_tied_face_ids=ids[abs(z-front)<=tol].tolist(),
        exact_incident_depth_face_ids=ids[structural].tolist(),
        extrapolated_face_ids=ids[extrapolated].tolist(), extrapolation_changes_visibility=bool(extrapolation_changes))
    if extrapolation_changes: return 'numerical_unknown_depth_extrapolation', evidence
    if gap > tol: return 'depth_conflict', evidence
    if np.any(np.all(signed > PIXEL_TOL, axis=1)): return 'strict_union_interior', evidence
    endpoint = int(source_edge_vertices[0]) if edge_t == 0 else (int(source_edge_vertices[1]) if edge_t == 1 else None)
    min_cones, max_cones, active_evidence = [], [], []
    for face, normals, distances in zip(ids, inward, signed):
        active = np.where(abs(distances) <= PIXEL_TOL)[0]
        if not len(active): return 'strict_union_interior', evidence
        vertices = triangle_vertex_ids[face]
        known = []
        for k in active:
            pair = vertices[[k, (k+1)%3]]
            if endpoint is not None:
                exact_support = endpoint in pair
            else:
                exact_support = set(pair.tolist()) == set(source_edge_vertices.tolist())
            if exact_support: known.append(int(k))
        min_cones.append(normals[active])
        max_cones.append(normals[known])
        active_evidence.append(dict(face_id=int(face), active_edges=active.tolist(), exact_support_edges=known))
    minimum, intervals, decisive = cone_intervals(min_cones)
    evidence.update(active_edges_per_face=active_evidence, angular_intervals=intervals, decisive_uncovered_interval=decisive)
    if minimum == 'full': return 'full_circle_union_interior', evidence
    if minimum != 'boundary': return 'numerical_unknown_cone', evidence
    # At uncertain tolerance-band faces, removing unproven constraints bounds possible local coverage.
    maximum, max_intervals, _ = cone_intervals(max_cones)
    evidence['max_possible_cone_status'] = maximum
    if maximum != 'boundary': return 'numerical_unknown_near_face_can_change_full_circle', evidence
    return 'accepted', evidence


def synthetic_checks(projected_class):
    def run(triangles, depths, q, z_edge, edge, vertex_ids, expected, name):
        triangles=np.array(triangles, float); depths=np.array(depths, float)
        projected=projected_class(triangles, depths, (8, 8))
        status, evidence=certify(projected, np.array(q, float), z_edge, np.array(edge), .5, np.array(vertex_ids))
        passed=(status=='accepted') == expected
        assert passed, (name, status)
        return dict(name=name, status=status, expected_accepted=expected, passed=passed)
    square=[[[0,0],[1,0],[1,1]], [[0,0],[1,1],[0,1]]]
    checks=[run(square, [[2]*3]*2, [.5,.5], 2, [0,2], [[0,1,2],[0,2,3]], False, 'square_internal_diagonal_rejected')]
    near=[[[.01,0],[.01,1],[0,.5]], [[.01,0],[1,.5],[.01,1]]]
    checks.append(run(near, [[2]*3]*2, [.01,.5], 2, [0,1], [[0,1,2],[0,3,1]], False, 'internal_seam_0_01px_from_exterior_rejected'))
    thin=[[[.1,.1],[.9,.1],[.5,.101]]]
    checks.append(run(thin, [[2]*3], [.5,.1], 2, [0,1], [[0,1,2]], True, 'thin_no_integer_pixel_support_true_edge_accepted'))
    overlap=thin+thin
    checks.append(run(overlap, [[3]*3,[2]*3], [.5,.1], 3, [0,1], [[0,1,2],[3,4,5]], False, 'same_projection_rear_layer_edge_rejected'))
    return checks
