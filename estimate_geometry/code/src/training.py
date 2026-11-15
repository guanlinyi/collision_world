"""Grouped CARLA supervision, then train-only uncertainty/noise calibration.

This continues a checkpoint by default; it does not claim scratch training is
equivalent to the released checkpoint's full historical training schedule.
"""
from pathlib import Path
import shutil,time
import numpy as np
import torch
from torch.nn import functional as F
from .common import features,read_npz,write_json,is_clipped,ROOT
from .model import SurfaceNet
from .inference import uncertainty_head,infer_surfaces
from .geometry import fit_geometry
from .state import calibrate

def train(bundle,out,init,asset_prior,epochs=24,uncertainty_epochs=16,device='cuda',seed=20261006):
    start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(seed)
    rows=bundle['records'];o=bundle['observations'];b=bundle['cameras']
    t=read_npz(bundle['folder']/'training_targets.npz');gt=read_npz(bundle['folder']/'reference_geometry.npz')
    tensor=lambda v:torch.as_tensor(np.asarray(v),device=device,dtype=torch.float32)
    x=tensor(features(bundle,range(len(rows))))
    targets=tensor(t['coords'].transpose(0,3,1,2));valid=torch.as_tensor(t['valid'][:,None],device=device);faces=torch.as_tensor(t['faces'],device=device).long()
    sources=[r['source_state'] for r in rows];views=[r['view'] for r in rows]
    dims=tensor(gt['dims'][sources]);R=tensor(gt['rotation'][sources]);center=tensor(gt['center'][sources]);q=tensor(b['q'][sources,views]);origin=tensor(b['t'][sources,views]);K=tensor(b['k'][sources,views])
    uv=tensor(o['pixels'].reshape(-1,4096,2));wh=tensor([[r['bbox'][2]-r['bbox'][0],r['bbox'][3]-r['bbox'][1]] for r in rows])
    focal=torch.stack([K[:,0,0],K[:,1,1]],1);principal=torch.stack([K[:,0,2],K[:,1,2]],1)
    trainids=[r['index'] for r in rows if r['episode']<4];groups={}
    for i in trainids:groups.setdefault(rows[i]['source_state'],[]).append(i)
    groups=list(groups.values());rng=np.random.default_rng(seed)
    model=SurfaceNet(8).to(device)
    if init is not None:model.load_state_dict(torch.load(init,map_location=device,weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.0001)
    for epoch in range(epochs):
        losses=[];model.train();order=rng.permutation(len(groups))
        for start_group in range(0,len(groups),4):
            ids=[i for group in [groups[k] for k in order[start_group:start_group+4]] for i in group]
            coords,logits=model(x[ids]);v=valid[ids];scale=dims[ids,:,None,None]/2
            loss=(F.smooth_l1_loss(coords*scale,targets[ids]*scale,beta=.05,reduction='none')*v).sum()/(v.sum()*3).clamp(min=1)
            loss+=.08*F.cross_entropy(logits,faces[ids],ignore_index=255)
            local=coords.flatten(2).transpose(1,2)*dims[ids,None]
            world=torch.bmm(local,R[ids].transpose(1,2))+center[ids,None]
            cc=torch.bmm(world-origin[ids,None],q[ids])
            projected=cc[:,:,:2]/cc[:,:,2,None].clamp(min=.5)*focal[ids,None]+principal[ids,None]
            error=projected-uv[ids];vv=v.flatten(2).transpose(1,2)
            loss+=.1*(F.smooth_l1_loss(error/wh[ids,None],torch.zeros_like(error),beta=.05,reduction='none')*vv).sum()/(vv.sum()*2).clamp(min=1)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        with (out/'training.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(epoch=epoch+1,loss=float(np.mean(losses))))+'\n')
        print(f'train epoch {epoch+1}/{epochs}',flush=True)
    torch.save(model.state_dict(),out/'surface.pt')
    write_json(out/'model.json',dict(input_channels=8,global_context=False,pixel_center_offset=0.,train_episodes=[0,1,2,3],epochs=epochs,initialization=str(init),seed=seed,selection='final checkpoint; no validation selection',scope='Fine-tune supported two-asset CARLA pipeline; not new verified accuracy'))
    with Path(asset_prior).open('rb') as source, (out/'assets.json').open('xb') as target:shutil.copyfileobj(source,target)
    # Frozen predictor features, same uncertainty architecture and train-only loss.
    torch.manual_seed(20261005);model.eval();features_all=[];coord=[]
    with torch.no_grad():
        for batch in x.split(32):
            c,_,f=model(batch,return_features=True);features_all.append(torch.cat([f,c,batch[:,4:]],1).cpu());coord.append(c.cpu())
    fx=torch.cat(features_all).to(device);prediction=torch.cat(coord).to(device)
    target=targets[trainids];v=valid[trainids];error2=(prediction[trainids]-target).square().detach();tx=fx[trainids]
    head=uncertainty_head().to(device);opt=torch.optim.AdamW(head.parameters(),lr=.001)
    for epoch in range(uncertainty_epochs):
        for ids in torch.randperm(len(trainids),device=device).split(32):
            ls=head(tx[ids]).clamp(-5.5,-.3);loss=((.5*error2[ids]*torch.exp(-2*ls)+ls)*v[ids]).sum()/(v[ids].sum()*3).clamp(min=1)
            if not torch.isfinite(loss):raise ValueError('Non-finite uncertainty loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    torch.save(head.state_dict(),out/'uncertainty.pt')
    # Drop in-memory activations before complete teacher-free prediction.
    del x,fx,tx,features_all,coord,prediction
    if device.startswith('cuda'):torch.cuda.empty_cache()
    surface=infer_surfaces(bundle,out,device);noise={'complete':[],'clipped':[]}
    for r in rows:
        if r['episode']>=4:continue
        i=r['index'];s=r['source_state'];v=r['view'];box=r['bbox'];keep=o['mask'][i]&(surface['confidence'][i]>.6)
        xyz=surface['coords'][i].astype(float)*gt['dims'][s]
        cam=(xyz@gt['rotation'][s].T+gt['center'][s]-b['t'][s,v])@b['q'][s,v];k=b['k'][s,v]
        pixel=cam[...,:2]/cam[...,2,None].clip(.5)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
        e=np.linalg.norm(pixel-o['pixels'][i],axis=-1)
        if keep.any():noise['clipped' if is_clipped(box) else 'complete'].append(float(np.median(e[keep])/(1+.03*min(box[2]-box[0],box[3]-box[1]))))
    write_json(out/'view_noise.json',dict(train_episodes=[0,1,2,3],normalized_sigma={k:float(np.median(v)) for k,v in noise.items()}))
    raw=fit_geometry(bundle,surface,out);calibrate(bundle,raw,gt,out)
    write_json(out/'training_complete.json',dict(seconds=time.perf_counter()-start,epochs=epochs,uncertainty_epochs=uncertainty_epochs,trained_on_episodes=[0,1,2,3],copied_known_asset_prior=True,checkpoint_not_yet_evaluated=True))
