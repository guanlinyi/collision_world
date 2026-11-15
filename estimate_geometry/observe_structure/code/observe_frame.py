"""Run original one-frame SAM selection using explicit weights or cached candidates."""
import argparse
from contextlib import nullcontext
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import whole_observer as whole

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cached-candidates',type=Path)
    p.add_argument('--sam-root',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--verify-frozen',action='store_true');args=p.parse_args()
    if args.cached_candidates and (args.sam_root or args.checkpoint):p.error('Choose cached candidates or explicit SAM weights')
    if not args.cached_candidates and not (args.sam_root and args.checkpoint):p.error('SAM mode requires --sam-root and --checkpoint')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Output outside observer purpose')
    output.mkdir(parents=True,exist_ok=False)
    row=json.loads((ROOT/'input_json/frame28_inputs.json').read_text(encoding='utf-8'))
    row['image']=str(PROJECT/row['image'])
    if args.cached_candidates:
        with np.load(args.cached_candidates,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        class CachedPredictor:
            def set_image(self,rgb):self.shape=rgb.shape
            def predict(self,**kwargs):
                assert arrays['masks'].shape[1:]==self.shape[:2]
                assert np.array_equal(kwargs['point_coords'],np.array(row['positive_points_crop_xy']+row['negative_points_crop_xy'],np.float32))
                assert np.array_equal(kwargs['box'],np.array(row['body_box_crop_xyxy'],np.float32))
                return arrays['masks'],arrays['scores'],arrays['logits']
        ui=SimpleNamespace(torch=SimpleNamespace(inference_mode=nullcontext),load_predictor=CachedPredictor,
                           SAM2_CHECKPOINT='cached-SAM-candidates',DEVICE='cached')
    else:
        from sam_loader import backend
        ui=backend(args.sam_root,args.checkpoint,args.device)
    whole.ROOT=output;whole.DENSE=ROOT/'seed';whole.load_backend=lambda:ui
    whole.main(frames=[28],car=0,inputs_override=[row],seed_fresh=True,
        review_name='case07_f28_whole_review.png',prompt_transform_id='case07_single_frame_consensus3of4_v1',
        initialization_limit='Frozen own07 frame27 seed and frame28 NCC-translated prompts/ROI; not autonomous video discovery.')
    saved=json.loads((output/'output_json/observations.json').read_text(encoding='utf-8'))['records'][0]
    receipt=dict(cached_candidates=bool(args.cached_candidates),SAM_forward_rerun=not bool(args.cached_candidates),
        new_GPU_forward_verified=False,query_truth_read=False,source_image_sha256=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest(),
        selected=saved['selected'],selected_area_pixels=saved['selected_area_pixels'])
    if args.verify_frozen:
        expected=json.loads((ROOT/'evaluation_only_json/frame28_whole.json').read_text(encoding='utf-8'))
        assert {k:saved[k] for k in expected}==expected
        current=np.array(Image.open(saved['mask_path']))
        original=np.array(Image.open(ROOT/'evaluation_only_png/frame28_whole_mask.png'))
        assert np.array_equal(current,original)
        receipt.update(candidate_summaries_exact=True,selected_mask_pixel_identical=True,passed=True)
    whole.save_json(output/'output_json/selection_acceptance.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
