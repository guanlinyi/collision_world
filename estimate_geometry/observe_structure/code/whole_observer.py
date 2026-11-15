import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
DENSE=ROOT/'seed'
LOADER=Path(__file__).with_name('sam_loader.py')
WHEELS=TEMPORAL=RGB=None
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
def load_backend():raise RuntimeError('Use observe_frame.py with explicit backend parameters')
def save_json(path, value, reuse_identical=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_identical and path.exists():
        assert read(path) == value, path
        return
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)

def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        image.save(stream, format='PNG')

def overlay(rgb, mask, points, negatives, text):
    color = np.array(rgb)
    color[mask] = (0.6 * color[mask] + 0.4 * np.array([25, 240, 180])).astype(np.uint8)
    image = Image.fromarray(color)
    draw = ImageDraw.Draw(image)
    for point, color in [(point, '#00ff50') for point in points] + [(point, '#ff3040') for point in negatives]:
        x, y = point
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    draw.rectangle([0, 0, image.width, 17], fill='black')
    draw.text((4, 2), text, fill='white')
    return image

def main(frames=range(28, 33), review_name='wholecar_review_27_32.png', prompt_transform=None, prompt_transform_id=None, car=1, inputs_override=None, initialization_limit=None, seed_fresh=False):
    if (ROOT / 'output_json/observations.json').exists():
        raise FileExistsError('Completed observations protected')
    for directory in ['input_json', 'output_json', 'output_npz', 'output_png']:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    seed = next((row for row in read(DENSE / 'input_json/segmentation.json')['records'] if row['car'] == car))
    body = next((row for row in read(DENSE / 'input_json/rgb_prompts.json')['cars'] if row['car'] == car))['body']
    seed_mask_file = Path(seed['png_directory']) / f'car{car}_whole.png'
    seed_mask = np.array(Image.open(seed_mask_file)) > 0
    seed_pixels = int(seed_mask.sum())
    crop27 = np.array(seed['crop_xyxy'])
    h, w = seed_mask.shape
    negative27 = np.array([[2, 2], [w - 3, 2], [w - 3, h - 3], [2, h - 3]], dtype=float)
    seed_wheels = next((row for row in read(WHEELS)['cars'] if row['car'] == car)) if inputs_override is None else None
    wheel27 = {wheel['name']: np.array(wheel['ellipse']['center_full_image_xy']) for wheel in seed_wheels['wheels']} if seed_wheels else {}
    inputs = []
    for row in read(TEMPORAL)['frames'] if inputs_override is None else []:
        if row['frame'] not in frames:
            continue
        assert row['accepted']
        wheels = {wheel['name']: np.array(wheel['center_full_xy']) for wheel in row['wheels']}
        delta = np.median(np.array([wheels[name] - wheel27[name] for name in ['front_wheel', 'rear_wheel']]), axis=0)
        integer_delta = np.rint(delta).astype(int)
        crop = crop27 + np.tile(integer_delta, 2)
        local_delta = delta - integer_delta
        inputs.append(dict(frame=row['frame'], car=car, image=str(RGB / f'rgb_{row['frame']:03d}.png'), crop_xyxy=crop.tolist(), median_wheel_displacement_full_image_px=delta.tolist(), integer_crop_displacement_px=integer_delta.tolist(), positive_points_crop_xy=(np.array(body['points']) + local_delta).tolist(), negative_points_crop_xy=(negative27 + local_delta).tolist(), body_box_crop_xyxy=(np.array(body['box']) + np.tile(local_delta, 2)).tolist()))
    if inputs_override is not None:
        inputs = inputs_override
    if prompt_transform is not None:
        inputs = prompt_transform(inputs)
    contract = dict(id=prompt_transform_id or 'same_scene_wheel_shift_fresh_wholecar_rgb_sam_v1', car=car, seed_frame=27, frames=list(frames), input_policy='Saved wheel centers only initialize translated crop/prompts. Fresh SAM runs independently on each current RGB crop. No mask warp or temporal mask reuse.', initialization_limit='Strong same-scene car identity, frame27 body box/points and accepted wheel correspondences; not autonomous discovery or cross-scene validation.', selection=dict(require_all_original_body_positive_points=True, require_no_original_body_negative_points=True, min_area_pixels=int(np.ceil(0.5 * seed_pixels)), min_area_definition='half of the seed wholecar SAM pixel count, frozen before current predictions', ranking='highest SAM self-score among eligible; no GT, pose, fitting or downstream residual selection'), frame27_mask_reference=str(seed_mask_file), frame27_crop_xyxy=crop27.tolist(), seed_whole_pixels=seed_pixels, query_truth_used=False, pose_used=False, used_in_fit=False, needs_image_review=True)
    if initialization_limit is not None:
        contract['initialization_limit'] = initialization_limit
        contract['input_policy'] = 'Fixed-scale RGB template NCC initializes translated crop/prompts; each accepted current RGB receives fresh independent SAM. No wheel, GT or pose input.'
    if prompt_transform is not None:
        contract['input_policy'] = 'Frozen external wheel-relative prompt transform; fresh independent RGB SAM. See experiment input_json/frozen_similarity.json. Original selection thresholds unchanged.'
    contract['selection'].update(require_all_original_body_positive_points=False,minimum_positive_hits=3)
    save_json(ROOT / 'input_json/contract.json', contract, reuse_identical=True)
    save_json(ROOT / 'input_json/translated_prompts.json', inputs, reuse_identical=True)
    sources = [Path(__file__), LOADER, DENSE / 'input_json/rgb_prompts.json', DENSE / 'input_json/segmentation.json', seed_mask_file]
    if inputs_override is None:
        sources += [TEMPORAL, WHEELS]
    save_json(ROOT / 'input_json/source_hashes.json', {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, reuse_identical=True)
    print('Importing project local SAM2 loader', flush=True)
    ui = load_backend()
    predictor = ui.load_predictor()
    records = [dict(frame=27, car=car, image=seed['image'], crop_xyxy=seed['crop_xyxy'], mask_path=str(seed_mask_file), mask_storage='reference only, no copy', selected=seed['whole_selected'], selected_area_pixels=seed_pixels, status='existing_seed_observation', fresh_prediction=False)]
    preview = []
    seed_rgb = np.array(Image.open(seed['image']).convert('RGB').crop(seed['crop_xyxy']))
    preview.append(overlay(seed_rgb, seed_mask, body['points'], negative27, 'f27: existing RGB wholecar seed'))
    if seed_fresh:
        records = []
        preview = []
    for item in inputs:
        frame = item['frame']
        image = Image.open(item['image']).convert('RGB')
        crop = item['crop_xyxy']
        assert 0 <= crop[0] < crop[2] <= image.width and 0 <= crop[1] < crop[3] <= image.height
        rgb = np.array(image.crop(crop))
        points, negatives = (item['positive_points_crop_xy'], item['negative_points_crop_xy'])
        if prompt_transform is not None or inputs_override is not None:
            save_png(ROOT / f'output_png/f{frame}_pure_RGB.png', Image.fromarray(rgb))
            save_png(ROOT / f'output_png/f{frame}_RGB_prompts.png', overlay(rgb, np.zeros(rgb.shape[:2], bool), points, negatives, f'f{frame} RGB prompts only'))
        with ui.torch.inference_mode():
            predictor.set_image(rgb)
            masks, scores, logits = predictor.predict(point_coords=np.array(points + negatives, np.float32), point_labels=np.array([1] * len(points) + [0] * len(negatives)), box=np.array(item['body_box_crop_xyxy'], np.float32), multimask_output=True)
        with (ROOT / f'output_npz/f{frame}_whole_candidates.npz').open('xb') as stream:
            np.savez_compressed(stream, masks=masks, scores=scores, logits=logits)
        candidates = []
        for index, mask in enumerate(masks.astype(bool)):
            positive_hits = sum((bool(mask[int(point[1]), int(point[0])]) for point in points))
            negative_hits = sum((bool(mask[int(point[1]), int(point[0])]) for point in negatives))
            area = int(mask.sum())
            reasons = []
            if positive_hits < 3:
                reasons.append('misses_body_positive_point')
            if negative_hits:
                reasons.append('includes_body_negative_point')
            if area < contract['selection']['min_area_pixels']:
                reasons.append('insufficient_wholecar_area')
            candidates.append(dict(index=index, sam_self_score=float(scores[index]), area_pixels=area, positive_hits=positive_hits, negative_hits=negative_hits, eligible=not reasons, reasons=reasons))
        eligible = [candidate for candidate in candidates if candidate['eligible']]
        selected = max(eligible, key=lambda candidate: candidate['sam_self_score'])['index'] if eligible else None
        for candidate in candidates:
            index = candidate['index']
            status = 'SELECTED' if index == selected else 'eligible' if candidate['eligible'] else 'rejected'
            review = overlay(rgb, masks[index].astype(bool), points, negatives, f'f{frame} candidate{index} {status} score={scores[index]:.3f}')
            save_png(ROOT / f'output_png/f{frame}_candidate{index}_overlay.png', review)
        if selected is not None:
            mask = masks[selected].astype(bool)
            mask_path = ROOT / f'output_png/f{frame}_whole_mask.png'
            save_png(mask_path, Image.fromarray(mask.astype(np.uint8) * 255))
            review = overlay(rgb, mask, points, negatives, f'f{frame} fresh wholecar: candidate{selected}')
            save_png(ROOT / f'output_png/f{frame}_whole_overlay.png', review)
        else:
            mask_path = None
            review = overlay(rgb, np.zeros(rgb.shape[:2], bool), points, negatives, f'f{frame} no eligible wholecar candidate')
        preview.append(review)
        record = dict(**item, selected=selected, candidates=candidates, mask_path=str(mask_path) if mask_path else None, selected_area_pixels=int(masks[selected].sum()) if selected is not None else None, candidates_file=str(ROOT / f'output_npz/f{frame}_whole_candidates.npz'), fresh_prediction=True, status='proposed_unreviewed' if selected is not None else 'unknown_no_eligible_mask', image_sha256=hashlib.sha256(Path(item['image']).read_bytes()).hexdigest())
        save_json(ROOT / f'output_json/frame_{frame}.json', record)
        records.append(record)
        print('FRAME', frame, 'selected', selected, 'area', record['selected_area_pixels'], flush=True)
    sheet = Image.new('RGB', (2 * w, (len(preview) + 1) // 2 * h), '#1c2028')
    for index, review in enumerate(preview):
        sheet.paste(review, (index % 2 * w, index // 2 * h))
    save_png(ROOT / 'output_png' / review_name, sheet)
    save_json(ROOT / 'output_json/observations.json', dict(query_truth_used=False, pose_used=False, used_in_fit=False, checkpoint=str(ui.SAM2_CHECKPOINT), device=ui.DEVICE, records=records, contract_id=contract['id']))
    print('WHOLECAR_OBSERVATIONS_SAVED', len(records), flush=True)
