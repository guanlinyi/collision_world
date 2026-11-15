"""Create an abstract cuboid fixture, not CARLA data or a quality benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.model import SurfaceNet
from src.geometry import upright, corners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Fixture must stay within estimate_geometry')
    out.mkdir(parents=True, exist_ok=False)
    def dump(name, obj):
        with (out / name).open('x', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
    states = [dict(episode=0, step=step, object_key=f'toy{car}', images=[None]*6)
              for step in range(3) for car in range(2)]
    n = len(states)
    center = np.array([[car*5 + step*.1, 0, 10] for step in range(3) for car in range(2)])
    rotation = np.broadcast_to(upright(0), (n, 3, 3)).copy()
    dims = np.tile([2., 1.5, 4.], (n, 1))
    q = np.broadcast_to(np.eye(3), (n, 6, 3, 3)).copy()
    t = np.zeros((n, 6, 3)); t[:, :, 1] = np.linspace(-.5, .5, 6)
    k = np.broadcast_to([[100.,0,320],[0,100.,180],[0,0,1]], (n,6,3,3)).copy()
    yy, xx = np.mgrid[-.4:.4:64j, -.4:.4:64j]
    coords = np.stack([xx, yy, np.zeros_like(xx)], -1).astype('f4')
    pixels=[]; records=[]
    for source, state in enumerate(states):
        for view in range(6):
            xyz = (coords*dims[source]) @ rotation[source].T + center[source] - t[source,view]
            uv = xyz[...,:2] / xyz[...,2,None] * 100 + [320,180]
            pixels.append(uv)
            records.append(dict(index=len(records),source_state=source,view=view,
                                bbox=[0,100,140,250] if view==0 else [250,100,390,250],
                                **{a:state[a] for a in ['episode','step']}))
    m=len(records)
    np.savez_compressed(out/'cameras.npz', q=q,t=t,k=k,valid=np.ones((n,6),bool))
    np.savez_compressed(out/'inputs.npz',rgb=np.full((m,64,64,3),128,np.uint8),
                        mask=np.ones((m,64,64),bool),pixels=np.array(pixels,dtype='f4'))
    np.savez_compressed(out/'training_targets.npz',coords=np.tile(coords,(m,1,1,1)),
                        valid=np.ones((m,64,64),bool),faces=np.zeros((m,64,64),np.uint8))
    geometry=dict(center=center,rotation=rotation,dims=dims,corners=corners(center,rotation,dims))
    np.savez_compressed(out/'reference_geometry.npz',**geometry)
    np.savez_compressed(out/'proposals.npz',source_indices=np.arange(n),**geometry)
    dump('states.json',states);dump('records.json',records)
    dump('contract.json',dict(fps=10,synthetic=True,scope='Abstract fixture; proposals equal reference by construction. Never use for accuracy claims.'))
    dump('assets.json',dict(assets=[dict(type_id='abstract-cuboid',dimensions_whl_m=[2,1.5,4])]*2,
                           classifier_coef=[[0.]*1024],classifier_intercept=[-1.]))
    torch.manual_seed(20261115)
    model=SurfaceNet(8)
    with torch.no_grad():model.out.bias[3]=10
    with (out/'initialization.pt').open('xb') as f:torch.save(model.state_dict(),f)
    dump('fixture_notice.json',dict(source='Generated mathematical cuboids',
         trained_weights=False,carla=False,reference_leaks_into_proposals=True,
         purpose='Software train/save/infer/export acceptance only'))
    print(out)


if __name__=='__main__':main()
