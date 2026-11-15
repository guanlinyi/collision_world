"""Build conditional task bounds without neural inference or reference geometry."""
import itertools
import numpy as np
from .frustum_task_bounds import center_outer_polytope,separated_by_bounds


def mask_pair_bounds(bundle,assets,translation_error_m,rotation_error_rad,pixel_half_width=1.):
    if not bundle['contract'].get('native_masks_and_actor_association',False):
        raise ValueError('This conditional method requires declared native masks and provided identities')
    if not assets:raise ValueError('An explicitly bounded known asset set is required')
    dimensions=np.asarray([asset['dimensions_whl_m'] for asset in assets],float)
    if dimensions.ndim!=2 or dimensions.shape[1]!=3 or not np.isfinite(dimensions).all() or np.any(dimensions<=0):raise ValueError('Invalid enclosing dimensions')
    radius=float(np.linalg.norm(dimensions,axis=1).max()/2)
    states=bundle['states'];camera=bundle['cameras'];observations=bundle['observations'];groups={};frames={}
    for r in bundle['records']:groups.setdefault(r['source_state'],[]).append(r)
    polytopes=[];statuses=[]
    for source,row in enumerate(states):
        frames.setdefault((row['episode'],row['step']),[]).append(source);views=[]
        for record in groups.get(source,[]):
            index,view=record['index'],record['view']
            if not camera['valid'][source,view]:continue
            uv=observations['pixels'][index][observations['mask'][index].astype(bool)]
            if len(uv):views.append(dict(pixels=uv,q=camera['q'][source,view],t=camera['t'][source,view],k=camera['k'][source,view]))
        poly=center_outer_polytope(views,radius,pixel_half_width,translation_error_m,rotation_error_rad)
        polytopes.append(poly);statuses.append(dict(source=source,status=poly['status'],used_view_count=len(views)))
    pairs=[]
    for (episode,step),sources in sorted(frames.items()):
        for a,b in itertools.combinations(sources,2):
            answer=separated_by_bounds(polytopes[a],polytopes[b])
            pairs.append(dict(episode=episode,step=step,sources=[a,b],tracks=[states[a]['object_key'],states[b]['object_key']],**answer))
    return dict(pairs=pairs,observation_status=statuses,
        summary=dict(total=len(pairs),separated=sum(p['status']=='separated' for p in pairs),unknown=sum(p['status']=='unknown' for p in pairs)),
        assumptions=dict(translation_error_bound_m=float(translation_error_m),rotation_error_bound_deg=float(np.rad2deg(rotation_error_rad)),
            pixel_cell_half_width=float(pixel_half_width),enclosing_radius_m=radius,
            native_masks_and_correct_identities=True,known_asset_set_encloses_every_body=True,
            meaning='Bounds are supplied assumptions, not measured calibration accuracy. Unspecified intrinsic/mask/category errors are not covered.'),
        ground_truth_opened=False,network_used=False,
        meaning='A separated result follows from enclosing sets under the stated premises. Unknown does not imply contact. This is not physical-accident classification.')
