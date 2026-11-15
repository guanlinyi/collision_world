"""RGB exterior-edge proposals from an observed SAM ROI; no pose, SOURCE or GT."""
import hashlib
import json
import argparse
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
WHOLE=ROOT/'input_json/observations.json'
RUN='attempt2'
CAR=1


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)
    with path.open('x',encoding='utf-8') as stream:stream.write(text)


def save_png(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:image.save(stream,format='PNG')


def sample(image,points):
    return map_coordinates(image,[points[...,1],points[...,0]],order=1,mode='nearest',prefilter=False)


def observe_frame(frame):
    if (ROOT/f'output_json/{RUN}/frame_{frame}.json').exists():raise FileExistsError('Completed observation protected')
    rec=next(r for r in read(WHOLE)['records'] if r['frame']==frame and r['car']==CAR)
    source_rgb=Path(rec['image']);source_mask=Path(rec['mask_path'])
    rgb=np.array(Image.open(source_rgb).convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float64)
    mask=np.array(Image.open(source_mask))>0
    crop=np.array(rec['crop_xyxy']);origin=crop[:2]
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contract=dict(id='rgb_local_normal_single_peak_exterior_candidates_v1',frame=frame,car=CAR,
        source_observation=str(WHOLE),source_image=str(source_rgb),source_mask=str(source_mask),crop_xyxy=crop.tolist(),
        proposal='All external SAM contour pixel centers, full CHAIN_APPROX_NONE. SAM is ROI/proposal only; do not assume its boundary is accurate.',
        pca=dict(neighbors=9,curvature='angle between q-p[-4] and p[+4]-q; greater than30deg is geometric unknown',
            outward='mask occupancy at q+/-2px along PCA normal must be one inside and one outside; orient toward outside'),
        photometry=dict(gray='OpenCV RGB2GRAY uint8 then float64; bilinear sampling, no smoothing',
            scan_range_px=[-4,4],scan_step_px=.25,derivative='(gray(s+.5)-gray(s-.5))/1px',
            minimum_main_abs_derivative_intensity_per_px=8,maximum_main_offset_from_SAM_boundary_px=2,
            minimum_independent_peak_separation_px=1,maximum_independent_secondary_to_main_ratio=.7,
            normal_consistency='abs(dot(gray_2d_gradient, outward_normal))/norm(gradient)>=cos(30deg); Cartesian gray gradient uses +/-0.5px central difference',
            peaks='All local maxima of abs derivative, plateau midpoint; include endpoints if strictly above adjacent point. Main is highest peak, ties closest to SAM seed then lower index. Secondary is strongest other peak at least1px away.'),
        limits='Candidates are not automatic semantic truth. Strong shadows can pass these photometric tests; weak/shadow and multi-peak areas remain unknown. Same-scene SAM initialization inherited. No SOURCE correspondence or pose fitting.',
        query_truth_used=False,pose_used=False,used_in_fit=False,threshold_sweep=False)
    contract['inherits_frame27_algorithm']='Same numerical algorithm/thresholds as output_py/attempt2/observe_rgb_edges_f27.py; only frame and output paths parameterized.'
    save_json(ROOT/f'input_json/{RUN}/contract_frame{frame}.json',contract)
    save_json(ROOT/f'input_json/{RUN}/source_hashes_frame{frame}.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),WHOLE,source_rgb,source_mask]})
    seeds,normals,geometry,contour_ids,contour_points=[],[],[],[],[]
    for cid,contour in enumerate(contours):
        points=contour[:,0,:].astype(float)
        if len(points)<9:
            for pid,point in enumerate(points):
                seeds.append(point+origin);normals.append([1.,0.]);geometry.append(dict(valid=False,reasons=['short_external_contour'],turn_deg=None));contour_ids.append(cid);contour_points.append(pid)
            continue
        for pid,point in enumerate(points):
            nearby=points[(pid+np.arange(-4,5))%len(points)]
            centered=nearby-nearby.mean(0)
            eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered/9)
            normal=eigenvectors[:,0]
            before,after=point-nearby[0],nearby[-1]-point
            tangent_norm=np.linalg.norm(before)*np.linalg.norm(after)
            turn=float(np.degrees(np.arccos(np.clip(before@after/tangent_norm,-1,1)))) if tangent_norm>0 else None
            probes=np.array([point+2*normal,point-2*normal])
            inside=map_coordinates(mask.astype(float),[probes[:,1],probes[:,0]],order=0,mode='constant',cval=0)>0.5
            reasons=[]
            if inside[0]==inside[1]:reasons.append('uncertain_mask_normal_orientation')
            elif inside[0]:normal=-normal
            if turn is None:reasons.append('degenerate_local_tangent_spur')
            elif turn>30:reasons.append('high_curvature_or_spur')
            if eigenvalues[1]<=1e-12:reasons.append('degenerate_local_PCA')
            seeds.append(point+origin);normals.append(normal);geometry.append(dict(valid=not reasons,reasons=reasons,turn_deg=turn,pca_eigenvalues=eigenvalues.tolist()));contour_ids.append(cid);contour_points.append(pid)
    seeds,normals=np.array(seeds),np.array(normals)
    offsets=np.arange(-4,4.001,.25)
    coordinates=seeds[:,None,:]+offsets[None,:,None]*normals[:,None,:]
    plus=coordinates+.5*normals[:,None,:];minus=coordinates-.5*normals[:,None,:]
    profiles=sample(gray,coordinates);derivative=sample(gray,plus)-sample(gray,minus)
    color_derivative=np.stack([sample(rgb[:,:,c].astype(float),plus)-sample(rgb[:,:,c].astype(float),minus) for c in range(3)],axis=-1)
    records=[]
    for index,(seed,normal,profile,geo) in enumerate(zip(seeds,normals,derivative,geometry)):
        magnitude=abs(profile)
        found,props=find_peaks(magnitude,plateau_size=True)
        peaks=found.tolist()
        if magnitude[0]>magnitude[1]:peaks.append(0)
        if magnitude[-1]>magnitude[-2]:peaks.append(len(magnitude)-1)
        peaks.sort(key=lambda k:(-magnitude[k],abs(offsets[k]),k))
        main=peaks[0] if peaks else int(np.argmax(magnitude))
        independent=[p for p in peaks[1:] if abs(offsets[p]-offsets[main])>=1]
        secondary=independent[0] if independent else None
        ratio=float(magnitude[secondary]/magnitude[main]) if secondary is not None and magnitude[main]>0 else 0.
        point=coordinates[index,main]
        gx=float(sample(gray,np.array([point+[.5,0]]))[0]-sample(gray,np.array([point-[.5,0]]))[0])
        gy=float(sample(gray,np.array([point+[0,.5]]))[0]-sample(gray,np.array([point-[0,.5]]))[0])
        gradient=np.array([gx,gy]);alignment=float(abs(gradient@normal)/np.linalg.norm(gradient)) if np.linalg.norm(gradient)>0 else 0.
        reasons=list(geo['reasons'])
        if magnitude[main]<8:reasons.append('weak_gray_contrast_shadow_not_identified')
        if abs(offsets[main])>2:reasons.append('strongest_peak_too_far_from_proposal')
        if ratio>.7:reasons.append('competing_independent_gradient_peaks')
        if alignment<np.cos(np.pi/6):reasons.append('gradient_not_normal_consistent')
        records.append(dict(id=index,contour_id=int(contour_ids[index]),contour_point_index=int(contour_points[index]),
            SAM_seed_full_image_xy=seed.tolist(),outward_normal_xy=normal.tolist(),main_peak_full_image_xy=point.tolist(),
            main_peak_offset_px=float(offsets[main]),main_peak_abs_gray_derivative=float(magnitude[main]),main_peak_signed_gray_derivative=float(profile[main]),
            secondary_peak_offset_px=float(offsets[secondary]) if secondary is not None else None,secondary_main_ratio=ratio,
            normal_gradient_abs_cosine=alignment,main_gray_gradient_xy=gradient.tolist(),
            all_peak_indices=peaks,geometry=geo,status='photometric_candidate' if not reasons else 'unknown',reasons=reasons))
    good=np.array([r['status']=='photometric_candidate' for r in records])
    reason_counts=Counter(reason for r in records for reason in r['reasons'])
    angle=np.mod(np.degrees(np.arctan2(normals[good,1],normals[good,0])),180)
    histogram,edges=np.histogram(angle,bins=np.arange(0,181,15))
    save_json(ROOT/f'output_json/{RUN}/frame_{frame}.json',dict(frame=frame,car=CAR,query_truth_used=False,pose_used=False,used_in_fit=False,
        input_contract=f'input_json/{RUN}/contract_frame{frame}.json',contours=len(contours),total_candidates=len(records),
        photometric_candidates=int(good.sum()),unknown=int((~good).sum()),reason_counts=dict(reason_counts),
        candidate_normal_histogram=dict(bin_edges_deg=edges.tolist(),counts=histogram.tolist()),
        profile_file=f'output_npz/{RUN}/frame{frame}_gradient_profiles.npz',records=records))
    (ROOT/f'output_npz/{RUN}').mkdir(parents=True,exist_ok=True)
    with (ROOT/f'output_npz/{RUN}/frame{frame}_gradient_profiles.npz').open('xb') as stream:
        np.savez_compressed(stream,SAM_seed_full_image_xy=seeds,outward_normal_xy=normals,scan_offsets_px=offsets,
            gray_intensity=profiles,signed_gray_derivative=derivative,signed_RGB_derivative=color_derivative,
            contour_id=np.array(contour_ids),contour_point_index=np.array(contour_points),photometric_candidate=good)
    panels=[]
    for mode in ['RGB','accepted','unknown']:
        canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
        if mode!='RGB':
            for record in records:
                seed=record['SAM_seed_full_image_xy']
                draw.point(tuple(seed),fill='#eeeeee')
                if mode=='accepted' and record['status']=='photometric_candidate':
                    point=np.array(record['main_peak_full_image_xy']);n=np.array(record['outward_normal_xy'])
                    color='#28ffb1' if abs(n[0])>=abs(n[1]) else '#52baff'
                    draw.line([tuple(point-2*n),tuple(point+2*n)],fill=color,width=1)
                if mode=='unknown' and record['status']=='unknown':
                    color='#ff5a51' if record['geometry']['reasons'] else ('#f5c84a' if 'competing_independent_gradient_peaks' in record['reasons'] else '#ff72d7')
                    draw.point(tuple(record['SAM_seed_full_image_xy']),fill=color)
        panel=canvas.crop(crop.tolist()).resize((820,470));draw=ImageDraw.Draw(panel)
        draw.rectangle([0,0,820,21],fill='black')
        text={'RGB':f'f{frame} pure RGB: no SOURCE, pose or GT',
            'accepted':f'RGB edge candidates {int(good.sum())}/{len(records)}: green |nx|>=|ny| / blue otherwise; white SAM seed',
            'unknown':'Unknown: red geometry / yellow competing peaks / magenta weak, distant or misaligned gradient'}[mode]
        draw.text((3,4),text,fill='white');panels.append(panel)
        save_png(ROOT/f'output_png/{RUN}/frame{frame}_{mode}.png',panel)
    sheet=Image.new('RGB',(820,1410))
    for i,panel in enumerate(panels):sheet.paste(panel,(0,470*i))
    save_png(ROOT/f'output_png/{RUN}/frame{frame}_rgb_edge_candidates.png',sheet)
    print('FRAME',frame,'RGB_EDGE_CANDIDATES',int(good.sum()),'/',len(records),'UNKNOWN_REASONS',dict(reason_counts),'NORMAL_BINS',histogram.tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,nargs='+',default=[27])
    parser.add_argument('--run',default='attempt2')
    parser.add_argument('--car',type=int,default=1)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    parser.add_argument('--whole',type=Path,default=WHOLE)
    args=parser.parse_args()
    RUN=args.run
    CAR=args.car
    ROOT=args.output_root.resolve()
    if not ROOT.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('Output outside observer purpose')
    WHOLE=args.whole
    for frame in args.frames:observe_frame(frame)
