"""Package inferred cuboids for the local Three.js viewer."""
import os
from pathlib import Path
import numpy as np
from .common import ROOT,read_json,read_npz,write_json
from .collision import frame_contacts
from .task_coverage import pair_status


def _rel_to_viewer(path):
    return Path(os.path.relpath(path,ROOT/'viewer')).as_posix()


def export_world(bundle,geometry,output,model_dir,include_reference=False):
    rows=bundle['states'];cam=bundle['cameras'];assets=read_json(Path(model_dir)/'assets.json')['assets']
    known=[(v['dimensions_whl_m'],v.get('type_id',str(k))) for k,v in enumerate(assets)]
    truth=read_npz(bundle['folder']/'reference_geometry.npz') if include_reference else None
    episodes={};group={};expected={};task_groups={}
    for source,row in enumerate(rows):expected.setdefault((str(row['episode']),int(row['step'])),[]).append(source)
    for record in pair_status(geometry,rows):task_groups.setdefault((str(record['episode']),int(record['step'])),[]).append(record)
    for i,source in enumerate(geometry['source_indices']):
        row=rows[int(source)];ep=str(row['episode']);step=int(row['step']);group.setdefault((ep,step),[]).append((i,int(source)))
    for (ep,step),sources in sorted(expected.items()):
        items=group.get((ep,step),[])
        first=rows[sources[0]];image_paths=first['images']
        objs=[];refs=[]
        for i,source in items:
            row=rows[source];dims=geometry['dims'][i].astype(float)
            asset=min(known,key=lambda v:np.linalg.norm(dims-np.asarray(v[0])))[1]
            obj=dict(track=row['object_key'],asset=asset,center=geometry['center'][i].tolist(),
                     rotation=geometry['rotation'][i].tolist(),dims=dims.tolist())
            objs.append(obj)
        if truth is not None:
            for source in sources:
                row=rows[source];asset=min(known,key=lambda v:np.linalg.norm(truth['dims'][source]-np.asarray(v[0])))[1]
                refs.append(dict(track=row['object_key'],asset=asset,center=truth['center'][source].tolist(),
                                 rotation=truth['rotation'][source].tolist(),dims=truth['dims'][source].tolist()))
        cameras=[dict(q=cam['q'][sources[0],v].tolist(),t=cam['t'][sources[0],v].tolist(),k=cam['k'][sources[0],v].tolist()) for v in range(6)]
        paths=[_rel_to_viewer(ROOT/p) if p else None for p in image_paths]
        pairs=frame_contacts(objs)
        frame=dict(step=step,time=step/bundle['contract']['fps'],images=paths,cameras=cameras,
                   variants=dict(prediction=objs,**({'truth':refs} if truth is not None else {})),contacts=pairs,
                   task_status=task_groups.get((ep,step),[]),provided_object_count=len(sources),predicted_object_count=len(objs))
        episodes.setdefault(ep,dict(frames=[],reference_grid_z=float(np.mean([o['center'][2]-o['dims'][1]/2 for o in objs])) if objs else 0.))['frames'].append(frame)
    payload=dict(episodes=episodes,meshes={},scope=dict(prediction='model-produced rigid cuboids from prepared calibrated CARLA inputs',
                 reference='CARLA reference loaded only with explicit --include-reference' if truth is not None else 'not loaded',
                 contacts='predicted cuboid overlap; not collision event labels',
                 task_status='Missing geometry is unknown. Coverage is relative to provided identities, not detection completeness.'))
    write_json(output,payload)
    return payload
