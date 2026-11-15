import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from geometry_pool import ROOT,GeometryPool,read,save_json,save_npz,load_problems,FrozenProblem
from exterior_matching import observation,initial_anchors,admission
AUDIT=ROOT
ATTEMPT=''
REQUIRE_FULL_POOL=False
def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value

def visible_problem(original):
    meta = dict(original.meta)
    meta['blocks'] = [b for b in original.meta['blocks'] if b['source_part'] != 'rear_right_lamp']
    meta['group_sizes'] = {}
    for block in meta['blocks']:
        group = block['name'].split(':')[0]
        meta['group_sizes'][group] = meta['group_sizes'].get(group, 0) + len(original.arrays[block['xyz_array']])
    assert set(meta['group_sizes']) == {'windshield', 'light'}
    return FrozenProblem(meta, original.arrays, original.geo)

def prepare(frame, builder, problem, out_root=None, context=None, source_foot_allowed=None):
    out_root = ROOT if out_root is None else Path(out_root)
    prefix = f'f{frame}' if context is None else context['sample_id']
    contract = read(ROOT / 'input_json/contract.json')
    assert contract['id'] == 'reliable_partial_exterior_pose_v1'
    pose = np.array(problem.meta['baseline_pose'])
    pool = builder.build(pose, problem.row, f'{prefix}_initial')
    obs = observation(frame, None if context is None else context['observation_path'])
    anchors, rejected = initial_anchors(pool, obs, source_foot_allowed)
    feet = Counter((tuple(anchor['observed_xy']) for anchor in anchors))
    duplicate = sum((count - 1 for count in feet.values() if count > 1))
    if duplicate:
        raise ValueError('Exact observed-foot duplicate handling requires the frozen supplement')
    if anchors:
        total = sum((a['raw_weight'] for a in anchors))
        for index, anchor in enumerate(anchors):
            anchor.update(id=index, weight=anchor['raw_weight'] / total)
    evidence = admission(anchors, builder, problem, pose)
    if REQUIRE_FULL_POOL and (not pool['admitted']):
        evidence['passed'] = False
        evidence['reasons'].append('SOURCE_pool_geometry_not_admitted')
    record = dict(frame=frame, query_truth_used=False, pose_used_for_observer=False, initial_pose=pose.tolist(), source_pool_label=f'{prefix}_initial', observed_file=str(obs['path']), observed_segments=len(obs['segments']), observed_arcs=len(np.unique(obs['arc_id'])), anchors=len(anchors), exact_duplicate_footpoints=duplicate, rejected_counts=dict(Counter((r['reason'] for r in rejected))), admission=evidence, fixed_anchors=plain(anchors), rejected_matches=plain(rejected))
    if context is not None:
        record.update(key=problem.meta['key'], car=problem.meta['car'], case=problem.meta['case'], source_foot_guard_used=source_foot_allowed is not None)
    path = out_root / 'output_json' / ATTEMPT / f'{prefix}_initial_anchors_local_competition.json'
    if path.exists():
        assert read(path) == record
    else:
        save_json(path, record)
    sources = [ROOT / 'input_json/contract.json', Path(__file__), ROOT / 'code/geometry_pool.py', ROOT / 'code/exterior_matching.py', AUDIT / 'code/tangent_kernel.py', obs['path']]
    if context is not None:
        sources += context.get('source_paths', [])
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    path = out_root / 'input_json' / ATTEMPT / f'{prefix}_initial_provenance_local_competition.json'
    if path.exists():
        assert read(path) == provenance
    else:
        save_json(path, provenance)
    if context is None:
        whole = read(ROOT.parent / '20261108_whole_vehicle_observation/output_json/observations.json')
        rgb_record = next((r for r in whole['records'] if r['frame'] == frame))
    else:
        rgb_record = context['rgb_record']
    rgb = Image.open(rgb_record['image']).convert('RGB')
    canvas = rgb.copy()
    draw = ImageDraw.Draw(canvas)
    for segment in obs['segments']:
        draw.line([tuple(p) for p in segment['points']], fill='#e5ce54')
    for left, right in pool['segment_ids']:
        draw.line([tuple(pool['xy'][left]), tuple(pool['xy'][right])], fill='#3fc6ff')
    for anchor in anchors:
        draw.line([tuple(anchor['source_xy']), tuple(anchor['observed_xy'])], fill='#2bff80')
    for reject in rejected:
        if 'source_xy' in reject:
            draw.point(tuple(reject['source_xy']), fill='#ff7159')
    panel = canvas.crop(rgb_record['crop_xyxy']).resize((820, 470))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, 820, 20], fill='black')
    draw.text((3, 4), f'f{frame}: SOURCE cyan / RGB segments yellow / accepted match green / rejected red', fill='white')
    out = out_root / 'output_png' / ATTEMPT / f'{prefix}_initial_matches_local_competition.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with out.open('xb') as stream:
            panel.save(stream, format='PNG')
    print('INITIAL_ADMISSION', frame, evidence, flush=True)
    return (pool, obs, anchors, evidence)
