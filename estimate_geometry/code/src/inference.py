"""Prediction from explicit calibrated inputs. Never opens training targets or GT."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .common import features,read_json,write_json,resource
from .model import SurfaceNet

def uncertainty_head():
    return nn.Sequential(nn.Conv2d(31,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,3,1))

def infer_surfaces(bundle,model_dir,device='cuda',batch_size=32):
    torch.set_num_threads(4);model_dir=Path(model_dir);cfg=read_json(model_dir/'model.json')
    model=SurfaceNet(8,global_context=cfg.get('global_context',False)).to(device).eval()
    model.load_state_dict(torch.load(resource(model_dir/'surface.pt'),map_location=device,weights_only=True))
    head=uncertainty_head().to(device).eval();head.load_state_dict(torch.load(resource(model_dir/'uncertainty.pt'),map_location=device,weights_only=True))
    coords=[];confidence=[];sigma=[]
    with torch.inference_mode():
        for start in range(0,len(bundle['records']),batch_size):
            indices=range(start,min(start+batch_size,len(bundle['records'])))
            x=torch.as_tensor(features(bundle,indices,cfg.get('pixel_center_offset',0.)),device=device)
            c,logits,f=model(x,return_features=True)
            uncertainty=head(torch.cat([f,c,x[:,4:]],1)).clamp(-5.5,-.3).exp()
            coords.append(c.cpu().numpy().transpose(0,2,3,1).astype(np.float16))
            confidence.append(logits.softmax(1).max(1).values.cpu().numpy().astype(np.float16))
            sigma.append(uncertainty.cpu().numpy().transpose(0,2,3,1).astype(np.float16))
    return dict(coords=np.concatenate(coords),confidence=np.concatenate(confidence),sigma=np.concatenate(sigma))

def run(bundle,model_dir,out,device='cuda',limit=None,state_update='released',recover_missing=False):
    from .geometry import fit_geometry
    from .state import filter_states
    start=time.perf_counter();surface=infer_surfaces(bundle,model_dir,device)
    np.savez_compressed(out/'surfaces.npz',**surface)
    raw=fit_geometry(bundle,surface,model_dir)
    recovered_count=0
    if recover_missing:
        from .proposal_recovery import recover_missing_geometry
        np.savez_compressed(out/'raw_geometry_before_recovery.npz',**raw)
        raw,recovery=recover_missing_geometry(bundle,surface,raw,model_dir)
        recovered_count=sum(r['status']=='recovered' for r in recovery)
        write_json(out/'proposal_recovery.json',recovery)
    np.savez_compressed(out/'raw_geometry.npz',**raw)
    state=filter_states(bundle,surface,raw,model_dir,limit=limit,state_update=state_update)
    np.savez_compressed(out/'geometry.npz',**state)
    write_json(out/'inference.json',dict(seconds=time.perf_counter()-start,states=len(state['source_indices']),device=device,model=str(Path(model_dir).resolve()),state_update=state_update,experimental_state_update=state_update!='released',experimental_proposal_recovery=recover_missing,recovered_states=recovered_count,ground_truth_opened=False,depth_opened=False,provided_inputs=['RGB crops','native instance masks','calibration','object identities','saved upstream learned proposals'],causality_scope='State and optional recovery use only current/past evidence; proposal provenance is a separate upstream contract.',scope='Prepared CARLA input only; not arbitrary video deployment'))
    return state
