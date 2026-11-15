"""Single entry point for the released CARLA research pipeline."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def evaluate(bundle,geometry,out):
    import numpy as np
    from src.common import read_npz,write_json
    from src.task_coverage import evaluate_pair_coverage
    reference=read_npz(bundle['folder']/'reference_geometry.npz');indices=geometry['source_indices'];rows=[bundle['states'][int(s)] for s in indices]
    episode={};corner=np.linalg.norm(geometry['corners']-reference['corners'][indices],axis=-1).mean(-1)
    center=np.linalg.norm(geometry['center']-reference['center'][indices],axis=-1)
    relative=np.einsum('nij,njk->nik',np.transpose(geometry['rotation'],(0,2,1)),reference['rotation'][indices])
    angles=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    for ep in sorted({r['episode'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['episode']==ep]
        episode[str(ep)]=dict(states=len(ids),center_mean_m=float(center[ids].mean()),corner_mean_m=float(corner[ids].mean()),rotation_mean_deg=float(angles[ids].mean()))
    coverage=evaluate_pair_coverage(geometry,reference,bundle['states'])
    result=dict(episodes=episode,pair_coverage=coverage,source='CARLA reference_geometry.npz opened ONLY by evaluate',
                scope='This compares predicted vehicle cuboids against CARLA 3D boxes for known episodes; it is not arbitrary-video accuracy.')
    write_json(out,result)
    return result


def main():
    p=argparse.ArgumentParser(description='Calibrated CARLA 2D-to-3D research demo; a bare arbitrary video is not yet supported.')
    sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('infer',help='Predict from a prepared calibrated CARLA bundle')
    i.add_argument('--input',type=Path,required=True);i.add_argument('--model',type=Path,required=True)
    i.add_argument('--output',type=Path,required=True);i.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    i.add_argument('--state-update',choices=['released','directional'],default='released',help='directional is an experimental coordinate-equivariant motion update, not verified overall accuracy improvement')
    i.add_argument('--recover-missing-proposals',action='store_true',help='Experimental current-pixel fitting initialized from an earlier same-track pose; may add false contact predictions')
    i.add_argument('--task-bounds',action='store_true',help='Also compute conditional mask-based separation/unknown decisions; explicit camera error bounds required')
    i.add_argument('--camera-position-bound-m',type=float)
    i.add_argument('--camera-angle-bound-deg',type=float)
    i.add_argument('--include-reference',action='store_true',help='Explicitly add CARLA GT to the viewer for inspection; prediction itself stays GT-free')
    t=sub.add_parser('train',help='Fine-tune on episodes 0-3 only')
    t.add_argument('--input',type=Path,required=True);t.add_argument('--output',type=Path,required=True)
    t.add_argument('--init',type=Path,help='Optional initialization checkpoint; omitted means random initialization, not historical reproduction');t.add_argument('--asset-prior',type=Path,required=True);t.add_argument('--epochs',type=int,default=24)
    t.add_argument('--uncertainty-epochs',type=int,default=16);t.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    e=sub.add_parser('evaluate',help='Compare an existing inference with CARLA reference boxes')
    e.add_argument('--input',type=Path,required=True);e.add_argument('--run',type=Path,required=True)
    x=sub.add_parser('export',help='Build the interactive world and box contacts from an existing prediction')
    x.add_argument('--input',type=Path,required=True);x.add_argument('--run',type=Path,required=True)
    x.add_argument('--model',type=Path,required=True);x.add_argument('--include-reference',action='store_true')
    args=p.parse_args()
    if args.command=='infer':
        if args.task_bounds and (args.camera_position_bound_m is None or args.camera_angle_bound_deg is None):
            p.error('--task-bounds requires explicit --camera-position-bound-m and --camera-angle-bound-deg assumptions')
        if not args.task_bounds and (args.camera_position_bound_m is not None or args.camera_angle_bound_deg is not None):
            p.error('Camera error bounds require --task-bounds')
        if args.task_bounds and (not 0<=args.camera_position_bound_m<float('inf') or not 0<=args.camera_angle_bound_deg<=180):
            p.error('Camera bounds must be finite and nonnegative; angle at most 180 degrees')
    from src.common import fresh,load_inputs,read_npz,write_json
    bundle=load_inputs(args.input,include_observations=args.command in ('infer','train'),include_proposals=args.command in ('infer','train'))
    if args.command=='train':
        from src.training import train
        if args.epochs<1 or args.uncertainty_epochs<1:raise ValueError('Both epoch counts must be positive')
        out=fresh(args.output);train(bundle,out,args.init,args.asset_prior,args.epochs,args.uncertainty_epochs,args.device)
        print(f'Training finished: {out}');return
    if args.command=='infer':
        from src.inference import run as infer
        from src.export import export_world
        out=fresh(args.output);geometry=infer(bundle,args.model,out,args.device,state_update=args.state_update,recover_missing=args.recover_missing_proposals)
        if args.task_bounds:
            import numpy as np
            from src.common import read_json
            from src.task_bounds_pipeline import mask_pair_bounds
            task_result=mask_pair_bounds(bundle,read_json(args.model/'assets.json')['assets'],args.camera_position_bound_m,np.deg2rad(args.camera_angle_bound_deg))
            write_json(out/'purpose_status.json',task_result)
        world=export_world(bundle,geometry,out/'world.json',args.model,args.include_reference)
        contacts={ep:[dict(step=f['step'],pairs=f['contacts'],task_status=f['task_status']) for f in item['frames']] for ep,item in world['episodes'].items()}
        write_json(out/'contacts.json',dict(episodes=contacts,scope='Oriented cuboid overlap from predictions; not physical contact labels'))
        if args.include_reference:evaluate(bundle,geometry,out/'evaluation.json')
        relative=(out/'world.json').relative_to(ROOT).as_posix()
        print(f'Prediction ready: {out}\nStart the repository replay viewer; open this world.json with its file selector.')
        return
    run_dir=args.run.resolve()
    if not run_dir.is_relative_to(ROOT):raise ValueError('Evaluation output must stay inside this lab folder')
    geometry=read_npz(run_dir/'geometry.npz')
    if args.command=='export':
        from src.export import export_world
        world=export_world(bundle,geometry,run_dir/'world.json',args.model,args.include_reference)
        contacts={ep:[dict(step=f['step'],pairs=f['contacts'],task_status=f['task_status']) for f in item['frames']] for ep,item in world['episodes'].items()}
        write_json(run_dir/'contacts.json',dict(episodes=contacts,scope='Oriented cuboid overlap from predictions; not physical contact labels'))
        print(f'Viewer data ready: {run_dir / "world.json"}')
    else:
        report=evaluate(bundle,geometry,run_dir/'evaluation.json')
        print(json.dumps(report['episodes'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
