import numpy as np
from collections import Counter,defaultdict
from geometry_pool import ROOT,read
OBS=ROOT
def observation(frame, path=None):
    path = OBS / f'output_json/{('attempt2' if frame == 27 else 'frames28_32_v1')}/frame_{frame}.json' if path is None else path
    doc = read(path)
    records = doc['records']
    by_contour = defaultdict(list)
    for record in records:
        by_contour[record['contour_id']].append(record)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    segments = []
    perimeters = {}
    for cid, rr in by_contour.items():
        rr.sort(key=lambda r: r['contour_point_index'])
        seeds = np.array([r['SAM_seed_full_image_xy'] for r in rr])
        length = np.linalg.norm(np.roll(seeds, -1, axis=0) - seeds, axis=1)
        coordinate = np.r_[0, np.cumsum(length)]
        perimeters[cid] = float(coordinate[-1])
        for i, left in enumerate(rr):
            right = rr[(i + 1) % len(rr)]
            if left['status'] != 'photometric_candidate' or right['status'] != 'photometric_candidate':
                continue
            a, b = (np.array(left['main_peak_full_image_xy']), np.array(right['main_peak_full_image_xy']))
            span = float(np.linalg.norm(b - a))
            if not 0 < span <= 2:
                continue
            normal = np.array(left['outward_normal_xy']) + np.array(right['outward_normal_xy'])
            normal /= np.linalg.norm(normal)
            ra, rb = (find(left['id']), find(right['id']))
            if ra != rb:
                parent[rb] = ra
            segments.append(dict(left_id=left['id'], right_id=right['id'], points=np.array([a, b]), normal=normal, contour_id=cid, SAM_start_coordinate=float(coordinate[i]), SAM_length=float(length[i])))
    arc_names = {}
    for segment in segments:
        key = find(segment['left_id'])
        arc_names.setdefault(key, len(arc_names))
        segment['arc_id'] = arc_names[key]
    return dict(path=path, doc=doc, segments=segments, points=np.array([s['points'] for s in segments]), normal=np.array([s['normal'] for s in segments]), arc_id=np.array([s['arc_id'] for s in segments]), contour_id=np.array([s['contour_id'] for s in segments]), coordinate=np.array([s['SAM_start_coordinate'] for s in segments]), seed_length=np.array([s['SAM_length'] for s in segments]), perimeters=perimeters)

def match_observed(point, normal, obs):
    segments = obs['points']
    vector = segments[:, 1] - segments[:, 0]
    alpha = np.sum((point - segments[:, 0]) * vector, axis=1) / np.sum(vector * vector, axis=1)
    foot = segments[:, 0] + alpha[:, None] * vector
    distance = np.linalg.norm(point - foot, axis=1)
    interior = (alpha > 0) & (alpha < 1)
    close = interior & (distance <= 5)
    angle = obs['normal'] @ normal >= np.cos(np.pi / 6)
    eligible = np.where(close & angle)[0]
    if not len(eligible):
        return (None, 'no_strict_segment_foot_within_5px' if not close.any() else 'outward_normal_over_30deg')
    eligible = eligible[np.argsort(distance[eligible], kind='stable')]
    best = int(eligible[0])
    best_coordinate = float(obs['coordinate'][best] + alpha[best] * obs['seed_length'][best])
    competitor = None
    for candidate in eligible[1:]:
        if obs['contour_id'][candidate] == obs['contour_id'][best]:
            other_coordinate = obs['coordinate'][candidate] + alpha[candidate] * obs['seed_length'][candidate]
            separation = abs(other_coordinate - best_coordinate)
            perimeter = obs['perimeters'][int(obs['contour_id'][best])]
            separation = min(separation, perimeter - separation)
            if separation < 10:
                continue
        competitor = int(candidate)
        break
    if competitor is not None and distance[competitor] - distance[best] < 2:
        return (None, 'competing_distinct_observed_arc_margin_under_2px')
    return (dict(observed_xy=foot[best], normal=obs['normal'][best], observed_segment_id=best, observed_arc_id=int(obs['arc_id'][best]), contour_id=int(obs['contour_id'][best]), contour_coordinate=best_coordinate, distance=float(distance[best]), competitor_segment_id=competitor, competitor_distance=float(distance[competitor]) if competitor is not None else None), None)

def initial_anchors(pool, obs, source_foot_allowed=None):
    support = np.zeros(len(pool['local']))
    for left, right in pool['segment_ids']:
        half = np.linalg.norm(pool['xy'][right] - pool['xy'][left]) / 2
        support[left] += half
        support[right] += half
    groups = defaultdict(list)
    for index in np.where(support > 0)[0]:
        groups[pool['unique_point_keys'][index]].append(int(index))
    anchors, rejected = ([], [])
    for key, indices in groups.items():
        weight = float(support[indices].sum())
        normal = np.sum(pool['normal'][indices] * support[indices, None], axis=0)
        if np.linalg.norm(normal) <= 1e-12:
            rejected.append(dict(source_sample_ids=indices, reason='degenerate_shared_source_normal'))
            continue
        normal /= np.linalg.norm(normal)
        index = indices[0]
        if source_foot_allowed is not None and (not bool(source_foot_allowed(pool['xy'][index:index + 1])[0])):
            rejected.append(dict(source_sample_ids=indices, reason='source_foot_mutual_occlusion_unknown', source_xy=pool['xy'][index].tolist()))
            continue
        match, reason = match_observed(pool['xy'][index], normal, obs)
        if reason is not None:
            rejected.append(dict(source_sample_ids=indices, reason=reason, source_xy=pool['xy'][index].tolist()))
            continue
        if source_foot_allowed is not None and (not bool(source_foot_allowed(np.asarray(match['observed_xy'])[None, :])[0])):
            rejected.append(dict(source_sample_ids=indices, reason='observed_foot_mutual_occlusion_unknown', source_xy=pool['xy'][index].tolist()))
            continue
        anchors.append(dict(**match, source_sample_id=index, source_sample_ids=indices, local=pool['local'][index], source_edge_id=int(pool['edge_ids'][index]), source_t=float(pool['t'][index]), source_xy=pool['xy'][index], raw_weight=weight))
    return (anchors, rejected)

def admission(anchors, pool_builder, problem, pose, weights=None):
    if not anchors:
        return dict(passed=False, reasons=['no_unique_external_anchors'])
    local = np.array([a['local'] for a in anchors])
    raw = np.array([a['raw_weight'] for a in anchors])
    weight = raw / raw.sum() if weights is None else np.asarray(weights)
    length = float(raw.sum())
    fraction = float(np.ptp(local[:, 0]) / pool_builder.length)
    lower = float(weight[local[:, 0] < pool_builder.midpoint].sum() / weight.sum())
    upper = float(weight[local[:, 0] >= pool_builder.midpoint].sum() / weight.sum())
    normal = np.array([a['normal'] for a in anchors])
    observed = np.array([a['observed_xy'] for a in anchors])
    x = np.asarray(pose)[[0, 1, 5]]

    def residual(v):
        xy, _ = pool_builder.module.project(local, problem.geo.full(v), problem.row)
        return np.sum((xy - observed) * normal, axis=1)
    jac = np.column_stack([(residual(x + np.eye(3)[j] * step) - residual(x - np.eye(3)[j] * step)) / (2 * step) for j, step in enumerate([0.0001, 0.0001, 0.001])])
    scaled = np.sqrt(weight)[:, None] * jac * np.array([0.01, 0.01, 0.1])[None, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if len(singular) == 3 and singular[0] > 0 else 0.0
    reasons = []
    if length < 30:
        reasons.append('initial_supported_length_under_30px')
    if fraction < 0.5:
        reasons.append('longitudinal_span_under_half_source_bbox_length')
    if min(lower, upper) < 0.2:
        reasons.append('one_longitudinal_half_under_20percent_weight')
    if not np.isfinite(singular).all() or ratio < 0.01:
        reasons.append('scaled_jacobian_singular_ratio_under_0_01')
    return dict(passed=not reasons, reasons=reasons, supported_initial_length_px=length, longitudinal_span_fraction=fraction, lower_half_weight_fraction=lower, upper_half_weight_fraction=upper, scaled_jacobian_singular_values=singular.tolist(), scaled_jacobian_minmax_ratio=ratio, parameter_scaling=[0.01, 0.01, 0.1])
