"""Causal box-center relations, not the legacy privileged GRU feature contract."""
import itertools
import numpy as np
from .collision import box_contact


def extract_relations(geometry, states):
    lookup = {int(s): i for i, s in enumerate(geometry['source_indices'])}
    if len(lookup) != len(geometry['source_indices']):
        raise ValueError('Duplicate geometry sources')
    groups = {}
    for source, row in enumerate(states):
        groups.setdefault((row['episode'], row['step']), []).append(source)
    history = {}; result = []
    for (episode, step), sources in sorted(groups.items()):
        objects = {}
        times = [float(states[s]['time_s']) for s in sources]
        if not np.isfinite(times).all() or max(times)-min(times) > 1e-8:
            raise ValueError('Finite synchronized object timestamps required')
        for source in sources:
            row = states[source]; key = (episode, row['object_key']); previous = history.get(key)
            current = dict(step=step, time=times[0], velocity=None, dt=None)
            if source not in lookup:
                history.pop(key, None)
                continue
            index = lookup[source]
            box = {k: np.asarray(geometry[k][index], float) for k in ('center', 'rotation', 'dims')}
            current['center'] = box['center']; acceleration = None
            if previous is not None and previous['step']+1 == step:
                dt = current['time']-previous['time']
                if dt <= 0:
                    raise ValueError('Non-increasing timestamps')
                current['velocity'] = (current['center']-previous['center'])/dt
                current['dt'] = dt
                if previous['velocity'] is not None:
                    acceleration = (current['velocity']-previous['velocity']) / ((dt+previous['dt'])/2)
            objects[source] = dict(box=box, velocity=current['velocity'], acceleration=acceleration)
            history[key] = current
        for a, b in itertools.combinations(sources, 2):
            item = dict(episode=episode, step=step, time_s=times[0], sources=[a,b],
                tracks=[states[a]['object_key'],states[b]['object_key']], status='unknown',
                center_distance_3d_m=None, minimum_axis_margin_m=None,
                closing_speed_center_proxy_mps=None, relative_interval_velocity_mps=None,
                acceleration_difference_proxy_mps2=None)
            if a in objects and b in objects:
                oa, ob = objects[a], objects[b]
                contact = box_contact(oa['box'], ob['box'])
                d = ob['box']['center']-oa['box']['center']; distance = float(np.linalg.norm(d))
                item.update(status='overlap' if contact['overlap'] else 'separated',
                            center_distance_3d_m=distance,minimum_axis_margin_m=contact['minimum_axis_margin_m'])
                if oa['velocity'] is not None and ob['velocity'] is not None:
                    dv = ob['velocity']-oa['velocity']
                    item['relative_interval_velocity_mps'] = float(np.linalg.norm(dv))
                    if distance > 1e-8:
                        item['closing_speed_center_proxy_mps'] = -float(d@dv)/distance
                if oa['acceleration'] is not None and ob['acceleration'] is not None:
                    item['acceleration_difference_proxy_mps2'] = float(np.linalg.norm(ob['acceleration']-oa['acceleration']))
            result.append(item)
    return result
