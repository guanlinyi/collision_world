import argparse,hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from scipy.optimize import least_squares
from geometry_pool import ROOT,GeometryPool,load_problems,read,save_json
from prototype_frame27 import prepare,visible_problem,plain
from exterior_matching import observation
from refresh_associations import refresh
ATTEMPT=''
def huber(r):
    r = np.asarray(r)
    return np.where(abs(r) <= 2, r * r, 4 * abs(r) - 4)

def exterior_residual(problem, builder, anchors, state, x):
    values = []
    pose = problem.geo.full(x)
    for anchor, association in zip(anchors, state['associations']):
        if association['status'] == 'missing':
            values.append(np.sqrt(anchor['weight'] * 16 / 3))
            continue
        xy, _ = builder.module.project(np.array([association['local']]), pose, problem.row)
        raw = float((xy[0] - anchor['observed_xy']) @ anchor['normal'])
        values.append(np.sign(raw) * np.sqrt(anchor['weight'] * huber(raw) / 3))
    return np.asarray(values)

def residual(problem, builder, anchors, state, x, objective_mode='mixed'):
    values = exterior_residual(problem, builder, anchors, state, x)
    if objective_mode == 'exterior_only':
        return values
    if objective_mode != 'mixed':
        raise ValueError('Unknown objective_mode: ' + objective_mode)
    return np.r_[problem.mask_residual(x, groups=3), values]

def state_cost(problem, builder, anchors, state, pose, objective_mode='mixed'):
    r = residual(problem, builder, anchors, state, pose[[0, 1, 5]], objective_mode)
    return float(r @ r)

def solve_inner(problem, builder, anchors, state, x, lower, upper, objective_mode='mixed'):
    fun = lambda v: residual(problem, builder, anchors, state, v, objective_mode)
    options = dict(max_nfev=150, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    fit = least_squares(fun, x, bounds=(lower, upper), diff_step=1e-05, loss='linear', **options)
    if not fit.success:
        fit = least_squares(lambda delta: fun(x + delta), np.zeros(3), bounds=(lower - x, upper - x), loss='linear', **options)
        fit.x += x
    return fit

def load_initial(frame, builder, original, out_root=None, initial_directory=None, context=None, source_foot_allowed=None):
    out_root = ROOT if out_root is None else Path(out_root)
    if context is not None and context.get('common_anchor_path') is not None:
        pool, obs, anchors, state = initialize_common_anchors(frame, builder, original, context, source_foot_allowed)
        prefix = context['sample_id']

        def thaw(value):
            if isinstance(value, (dict, MappingProxyType)):
                return {key: thaw(v) for key, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [thaw(v) for v in value]
            return plain(value)
        save_json(out_root / 'output_json' / ATTEMPT / f'{prefix}_initial_anchors_local_competition.json', dict(frame=frame, query_truth_used=False, initial_pose=pool['pose'].tolist(), source_pool_label=f'{prefix}_initial', common_anchor_path=str(Path(context['common_anchor_path']).resolve()), common_anchor_sha256=hashlib.sha256(Path(context['common_anchor_path']).read_bytes()).hexdigest(), fixed_anchors=thaw(anchors), admission=plain(state['admission']), refreshed_initial_state=plain(state), canonical_admission_reused=False))
        return (pool, obs, anchors, state['admission'])
    prefix = f'f{frame}' if context is None else context['sample_id']
    directory = out_root / 'output_json' / ATTEMPT if initial_directory is None else Path(initial_directory)
    path = directory / f'{prefix}_initial_anchors_local_competition.json'
    if not path.exists():
        if initial_directory is not None:
            raise FileNotFoundError(path)
        return prepare(frame, builder, original, out_root=out_root, context=context, source_foot_allowed=source_foot_allowed)
    record = read(path)
    pose = np.array(record['initial_pose'])
    pool = builder.build(pose, original.row, f'{prefix}_initial')
    anchors = record['fixed_anchors']
    for anchor in anchors:
        for key in ['normal', 'observed_xy', 'local', 'source_xy']:
            anchor[key] = np.array(anchor[key], float)
    return (pool, observation(frame, None if context is None else context['observation_path']), anchors, record['admission'])

def immutable_common_anchors(path):
    """Independent immutable observations; SOURCE fields remain provenance only."""
    record = read(Path(path))
    anchors = []

    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType({k: freeze(v) for k, v in value.items()})
        if isinstance(value, list):
            return tuple((freeze(v) for v in value))
        return value
    for index, anchor in enumerate(record['fixed_anchors']):
        if anchor['id'] != index:
            raise ValueError('Common anchor id must equal its ordered index')
        anchors.append(freeze(anchor))
    return tuple(anchors)

def initialize_common_anchors(frame, builder, original, context, source_foot_allowed=None):
    """Opt-in initialization at this variant's pose, never the domain owner's pose."""
    anchors = immutable_common_anchors(context['common_anchor_path'])
    pose = np.array(original.meta['baseline_pose'], float)
    pool = builder.build(pose, original.row, f'{context['sample_id']}_initial')
    if not np.array_equal(pool['pose'], pose):
        raise ValueError('Initial SOURCE pool differs from actual start')
    obs = observation(frame, context['observation_path'])
    state = refresh(builder, pool, anchors, context['problem'], source_foot_allowed)
    return (pool, obs, anchors, state)

def original_overlap_audit(pool):
    ids = pool['segment_ids']
    ends = pool['local'][ids]
    vector = ends[:, 1] - ends[:, 0]
    length = np.linalg.norm(vector, axis=1)
    guard = 128 * np.finfo(float).eps
    overlaps = 0
    for index in range(1, len(ids)):
        parallel = np.linalg.norm(np.cross(vector[:index], vector[index]), axis=1) <= guard * length[:index] * length[index]
        distance = np.linalg.norm(np.cross(ends[:index, 0] - ends[index, 0], vector[index]), axis=1) / length[index]
        same = parallel & (distance <= guard * max(1, float(np.linalg.norm(ends[index], axis=1).max())))
        for other in np.where(same)[0]:
            fraction = (ends[other] - ends[index, 0]) @ (vector[index] / length[index])
            overlap = min(length[index], float(fraction.max())) - max(0.0, float(fraction.min()))
            overlaps += overlap > guard * max(1, length[index])
    return int(overlaps)

def refreshed_backtracking(problem, builder, anchors, state, current, inner_x, lower, upper, prefix, cost_current, source_foot_allowed=None):
    """Opt-in first valid true-cost descent; rejected witnesses never commit state."""
    if not np.isfinite(cost_current):
        raise FloatingPointError('Nonfinite current refreshed cost: ' + str(cost_current))
    active = np.array([a['local'] for a in state['associations'] if a['status'] == 'matched'])
    start_xy, _ = builder.module.project(active, current, problem.row)
    x = current[[0, 1, 5]]
    trials = []
    attempted = []
    permitted = 0
    for fraction in [1.0, 0.5, 0.25, 0.125]:
        proposed_x = x + fraction * (inner_x - x)
        pose = problem.geo.full(proposed_x)
        label = prefix + '_fraction' + str(fraction).replace('.', 'p')
        trial = dict(fraction=fraction, pose=pose.tolist(), witness_label=label, current_refreshed_cost=cost_current)
        trials.append(trial)
        try:
            image, _ = builder.module.project(active, pose, problem.row)
            displacement = float(np.linalg.norm(image - start_xy, axis=1).max())
            within = bool(np.all(proposed_x >= lower) & np.all(proposed_x <= upper))
            trial.update(max_actual_projected_displacement_px=displacement, within_bounds=within)
            if not within or not np.isfinite(displacement) or displacement > 2:
                trial['status'] = 'rejected_bounds_or_projected_step'
                continue
            permitted += 1
            attempted.append(pose.tolist())
            pool = builder.build(pose, problem.row, label)
            refreshed = refresh(builder, pool, anchors, problem, source_foot_allowed)
            cost = state_cost(problem, builder, anchors, refreshed, pose, 'exterior_only')
            missing = sum((a['weight'] for a, s in zip(anchors, refreshed['associations']) if s['status'] == 'missing'))
            trial.update(proposed_support=plain(refreshed), support_weight=refreshed['support_weight'], missing_initial_weight=missing, missing_penalty_cost=missing * 16 / 3, trial_refreshed_cost=float(cost) if np.isfinite(cost) else None)
            if not refreshed['admission']['passed']:
                trial['status'] = 'rejected_refreshed_admission'
            elif not np.isfinite(cost):
                trial.update(status='rejected_nonfinite_cost', nonfinite_cost_repr=str(cost))
            elif cost_current - cost <= 1e-10 * max(1, cost_current):
                trial['status'] = 'rejected_no_refreshed_descent'
            else:
                trial['status'] = 'accepted'
                return dict(accepted=True, pose=pose, pool=pool, state=refreshed, cost=cost, accepted_fraction=fraction, trials=trials, attempted=attempted, terminal_reason='accepted_first_admissible_descent')
        except Exception as exc:
            known_clipping = isinstance(exc, ValueError) and str(exc) in ['Unsupported SOURCE near clipping', 'SOURCE image clipping']
            trial.update(status='rejected_known_geometry_clipping' if known_clipping else 'fatal_trial_exception', exception_type=type(exc).__name__, exception_message=str(exc))
            if not known_clipping:
                exc.refreshed_backtracking_trials = plain(trials)
                raise
    return dict(accepted=False, trials=trials, attempted=attempted, accepted_fraction=None, terminal_reason='stopped_no_admissible_refreshed_descent' if permitted else 'stopped_no_permitted_projected_step')

def fit_frame(frame, builder, original, out_root=None, objective_mode='mixed', initial_directory=None, contract_path=None, comparison_poses=None, context=None, source_foot_allowed=None, acceptance_policy='legacy'):
    if acceptance_policy not in ['legacy', 'true_refreshed_backtracking']:
        raise ValueError('Unknown acceptance policy')
    if acceptance_policy != 'legacy' and objective_mode != 'exterior_only':
        raise ValueError('True refreshed backtracking is exterior_only opt-in')
    out_root = ROOT if out_root is None else Path(out_root)
    prefix = f'f{frame}' if context is None else context['sample_id']
    destination = out_root / 'output_json' / ATTEMPT / f'{prefix}_fit_before_scoring.json'
    if destination.exists():
        return read(destination)
    baseline = np.array(original.meta['baseline_pose'])
    if context is None:
        problem = visible_problem(original)
        previous = next((r for r in read(ROOT.parent / '20261108_visible_lamp_direction/output_json/fits_before_scoring.json')['records'] if r['frame'] == frame))
    else:
        if objective_mode != 'exterior_only':
            raise ValueError('Generic context requires exterior_only; no internal masks were supplied')
        problem = context['problem']
        previous = dict(mask_only_pose=None, mask_only_cost=None)
    pool, obs, anchors, initial_gate = load_initial(frame, builder, original, out_root, initial_directory, context, source_foot_allowed)
    duplicates = original_overlap_audit(pool)
    if duplicates:
        initial_gate = dict(initial_gate, passed=False, reasons=initial_gate['reasons'] + ['overlapping_SOURCE_support_requires_union_deduplication'])
    source_paths = [Path(__file__), ROOT / 'input_json/contract.json', Path(__file__).with_name('refresh_associations.py'), Path(__file__).with_name('exterior_matching.py'), Path(__file__).with_name('geometry_pool.py'), Path(__file__).with_name('prototype_frame27.py')]
    if contract_path is not None:
        source_paths.append(Path(contract_path))
    if context is not None:
        source_paths += context.get('source_paths', [])
    save_json(out_root / 'input_json' / ATTEMPT / f'{prefix}_fit_implementation.json', dict(query_truth_used=False, objective_mode=objective_mode, sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}, weight_overlap_audit=dict(nontrivial_original_3d_collinear_overlap_pairs=duplicates), initial_pose=baseline.tolist(), global_start=original.start.tolist(), global_bounds=original.bound.tolist(), observer='saved pose-free RGB point/normal observations only; no wholeSAM or rejected seeds in objective'))
    rounds = []
    current = baseline.copy()
    accepted_rounds = 0
    attempted = []
    final_state = None
    if initial_gate['passed']:
        state = refresh(builder, pool, anchors, problem, source_foot_allowed)
        save_json(out_root / 'output_json' / ATTEMPT / f'{prefix}_initial_refreshed_associations.json', plain(state))
        for outer in range(8):
            if not state['admission']['passed']:
                rounds.append(dict(round=outer, status='stopped_current_refreshed_support_gate', state=plain(state)))
                break
            x = current[[0, 1, 5]]
            limit = np.array([0.01, 0.01, 0.5])
            lower = np.maximum(original.start - original.bound, x - limit)
            upper = np.minimum(original.start + original.bound, x + limit)
            cost_current = state_cost(problem, builder, anchors, state, current, objective_mode)
            fit = solve_inner(problem, builder, anchors, state, x, lower, upper, objective_mode)
            detail = dict(round=outer, starting_pose=current.tolist(), current_refreshed_cost=cost_current, inner_success=bool(fit.success), inner_nfev=fit.nfev, inner_cost=float(fit.fun @ fit.fun), attempted_inner_pose=problem.geo.full(fit.x).tolist(), local_bounds=[lower.tolist(), upper.tolist()])
            if not fit.success or not np.isfinite(fit.x).all():
                detail['status'] = 'stopped_inner_failure'
                rounds.append(detail)
                break
            if acceptance_policy == 'true_refreshed_backtracking':
                result = refreshed_backtracking(problem, builder, anchors, state, current, fit.x, lower, upper, f'{prefix}_round{outer}_proposal', cost_current, source_foot_allowed)
                attempted.extend(result['attempted'])
                detail.update(fraction_trials=result['trials'], accepted_fraction=result['accepted_fraction'], terminal_reason=result['terminal_reason'], same_anchor_domain=True, step_checks=[{k: t[k] for k in ['fraction', 'max_actual_projected_displacement_px', 'within_bounds'] if k in t} for t in result['trials']])
                if result['accepted']:
                    detail.update(status='accepted', proposed_pose=result['pose'].tolist(), proposed_refreshed_cost=result['cost'], proposed_support=plain(result['state']))
                    current = result['pose']
                    pool = result['pool']
                    state = result['state']
                    accepted_rounds += 1
                else:
                    detail['status'] = result['terminal_reason']
                rounds.append(detail)
                if not result['accepted']:
                    break
                continue
            active = np.array([a['local'] for a in state['associations'] if a['status'] == 'matched'])
            start_xy, _ = builder.module.project(active, current, problem.row)
            proposal = None
            fractions = []
            for fraction in [1.0, 0.5, 0.25, 0.125]:
                proposed_x = x + fraction * (fit.x - x)
                proposed_pose = problem.geo.full(proposed_x)
                image, _ = builder.module.project(active, proposed_pose, problem.row)
                displacement = float(np.linalg.norm(image - start_xy, axis=1).max())
                within = bool(np.all(proposed_x >= lower) & np.all(proposed_x <= upper))
                fractions.append(dict(fraction=fraction, max_actual_projected_displacement_px=displacement, within_bounds=within))
                if within and displacement <= 2:
                    proposal = proposed_pose
                    break
            detail['step_checks'] = fractions
            if proposal is None:
                detail['status'] = 'stopped_no_permitted_projected_step'
                rounds.append(detail)
                break
            attempted.append(proposal.tolist())
            proposed_pool = builder.build(proposal, problem.row, f'{prefix}_round{outer}_proposal')
            proposed_state = refresh(builder, proposed_pool, anchors, problem, source_foot_allowed)
            cost_proposed = state_cost(problem, builder, anchors, proposed_state, proposal, objective_mode)
            detail.update(proposed_pose=proposal.tolist(), proposed_refreshed_cost=cost_proposed, proposed_support=plain(proposed_state), same_anchor_domain=True)
            decrease = cost_current - cost_proposed
            if not proposed_state['admission']['passed']:
                detail['status'] = 'stopped_proposed_refreshed_support_gate'
            elif decrease <= 1e-10 * max(1, cost_current):
                detail['status'] = 'stopped_no_refreshed_objective_descent'
            else:
                detail['status'] = 'accepted'
                current = proposal
                pool = proposed_pool
                state = proposed_state
                accepted_rounds += 1
            rounds.append(detail)
            if detail['status'] != 'accepted':
                break
        final_pool = builder.build(current, problem.row, f'{prefix}_final')
        final_state = refresh(builder, final_pool, anchors, problem, source_foot_allowed)
        if not final_state['admission']['passed']:
            current = baseline.copy()
            status = 'fallback_final_support_failed'
        elif accepted_rounds == 0:
            current = baseline.copy()
            status = 'fallback_no_outer_step_accepted'
        else:
            status = 'accepted'
    else:
        status = 'fallback_initial_admission_failed'
    final_cost = state_cost(problem, builder, anchors, final_state, current, objective_mode) if final_state is not None and status == 'accepted' else None
    old_r = original.mask_residual(current[[0, 1, 5]]) if context is None else None
    visible_r = problem.mask_residual(current[[0, 1, 5]]) if context is None else None
    record = dict(key=original.meta['key'], frame=frame, car=1 if context is None else original.meta['car'], baseline_pose=baseline.tolist(), visible_lamp_pose=previous['mask_only_pose'], candidate_pose=current.tolist(), baseline_cost=original.meta['baseline_cost'], visible_lamp_cost=previous['mask_only_cost'], candidate_cost=final_cost, candidate_original_mask_cost=float(old_r @ old_r) if old_r is not None else None, candidate_visible_lamp_mask_cost=float(visible_r @ visible_r) if visible_r is not None else None, status=status, accepted=status == 'accepted', query_truth_read=False, no_wheel_direction=True, initial_admission=initial_gate, accepted_outer_rounds=accepted_rounds, attempted_poses=attempted, fixed_initial_anchor_count=len(anchors), rounds=rounds, final_refreshed_state=plain(final_state), comparison='Old filtered baseline and prior visible-lamp mask-only both preserved; independent GT scorer is separate')
    if objective_mode != 'mixed':
        record['objective_mode'] = objective_mode
        record['comparison'] = 'Frozen exterior-only control; all pose scoring is separate and follows six-frame freeze'
    if comparison_poses is not None:
        record.update(comparison_poses)
    if acceptance_policy != 'legacy':
        record['acceptance_policy'] = acceptance_policy
    if context is not None:
        record.update(case=original.meta['case'], sample_id=prefix, source_foot_guard_used=source_foot_allowed is not None)
    save_json(destination, plain(record))
    print('FRAME_FIT_FROZEN', frame, status, 'accepted_rounds', accepted_rounds, flush=True)
    return record
