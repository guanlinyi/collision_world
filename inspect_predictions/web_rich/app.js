
import {deriveDisplayedData,contactText} from './selected_view.js';
import {setupHeadingLines} from './heading_lines.js';
import {buildVehicleGeometry} from './vehicle_geometry.js';
import {valuesAt} from './frame_values.js';
import {setupInspection} from './inspection_panel.js';
import {caseDataUrl,initialView} from './case_routes.js';
import {renderCaseTruth,validateCaseTruth} from './case_truth.js';
import {MOTION_METHOD,motionEvidence,drawMotionEvidence,updateMotionPanel,setupMotionEvidenceLayer} from './motion_evidence.js';
import * as THREE from './input_js/three.module.js';
import { OrbitControls } from './input_js/OrbitControls.js';
const $=id=>document.getElementById(id);
const state={frame:27,version:'candidate',car:1,playing:false,rate:1,serial:0,rendering:false,follow:false};
let data,geometry,renderer,scene,camera,controls,world,pred=[],truth=[],errors=[],timer=null;
const images=new Map();
const tintedImages=new WeakMap();
const rgbBuffer=document.createElement('canvas');rgbBuffer.width=1920;rgbBuffer.height=1080;
const cropBuffer=document.createElement('canvas');cropBuffer.width=640;cropBuffer.height=360;
let updateInspection = () => {};
let structureComparison=null;
let poseExperiments=null,pairGeometry=null,headingLines=null,viewCache=null,motionLayer=null;
let caseTruth=null,caseTruthError='正在读取原始仿真记录。';
function displayedData(){
 const selection=poseExperiments?.selection();
 const key=state.version+':'+(selection?.enabled?selection.method?.id:'baseline');
 if(!viewCache||viewCache.key!==key||viewCache.geometry!==pairGeometry)viewCache={key,geometry:pairGeometry,data:deriveDisplayedData(data,state.version,selection,pairGeometry)};
 return viewCache.data;
}
function displayedPrediction(f,c){return displayedData().frames[f].cars[c].versions.displayed;}
// Only the initial webpage view changes; saved predictions and model defaults do not.
function applyInitialDisplayMethod(method){const baseline=method==='original'||method==='candidate'?method:'candidate';state.version=baseline;$('displayMethod').value=baseline;}

const predictedCenters=[],referenceCenters=[],centerLabels=[];
let recordedCameras=null,sourceCameraMode=false,sourceViewport=null,sourceProjectionRow=null,cameraHelpers=null;
const partNames={front_windshield:'前风挡',rear_window:'后窗',front_left_lamp:'左前灯',front_right_lamp:'右前灯',rear_left_lamp:'左后灯',rear_right_lamp:'右后灯'};
const colors={window:'#48e0f5',left:'#ff64d8',right:'#ffe15c'};
const fmt=(v,n=3)=>v===null||v===undefined?'—':Number(v).toFixed(n);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function failure(error){state.playing=false;clearTimeout(timer);$('fatal').hidden=false;$('fatal').textContent='无法完成同步回放：'+error.message+'\n请确认本地数据、依赖和WebGL可用；本页面不会用二维图替代缺失的三维结果。';$('sync').textContent='载入失败';}
async function json(url){if(window.CWM_OFFLINE){if(url==='/data/case.json')return window.CWM_OFFLINE.data;if(url==='/data/camera.json'){if(!window.CWM_OFFLINE.cameras)throw new Error('离线页面缺少原始相机参数，请重新构建完整页面');return window.CWM_OFFLINE.cameras;}if(url===window.CWM_OFFLINE.data.geometry_url)return window.CWM_OFFLINE.geometry;throw new Error('离线数据未收录 '+url);}const r=await fetch(caseDataUrl(url));if(!r.ok)throw new Error(url+' HTTP '+r.status);return r.json();}
function image(url){if(images.has(url))return images.get(url);const p=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('图像读取失败 '+url));im.src=window.CWM_OFFLINE ? window.CWM_OFFLINE.imageUrls[url] : url;});images.set(url,p);while(images.size>48)images.delete(images.keys().next().value);return p;}
const partPalette={front_windshield:'#38bdf8',rear_window:'#2dd4bf',front_left_lamp:'#facc15',front_right_lamp:'#fb923c',rear_left_lamp:'#f472b6',rear_right_lamp:'#f87171'};
function partColor(name){return partPalette[name]||'#94a3b8';}
function colored(im,color){let variants=tintedImages.get(im);if(!variants){variants=new Map();tintedImages.set(im,variants);}if(variants.has(color))return variants.get(color);const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const x=c.getContext('2d');x.drawImage(im,0,0);x.globalCompositeOperation='source-in';x.fillStyle=color;x.fillRect(0,0,c.width,c.height);variants.set(color,c);return c;}
function toView(p){const o=data.display_origin_lab;return new THREE.Vector3(p[0]-o[0],p[2]-o[2],-(p[1]-o[1]));}
function init3D(){
 const holder=$('threeView');scene=new THREE.Scene();scene.background=new THREE.Color('#101826');camera=new THREE.PerspectiveCamera(42,1,.02,300);renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;holder.appendChild(renderer.domElement);controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.maxDistance=120;controls.minDistance=.8;
 scene.add(new THREE.HemisphereLight(0xd5e8ff,0x354431,2.8));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(6,12,8);scene.add(light);
 world=new THREE.Group();world.rotation.x=-Math.PI/2;scene.add(world);
 motionLayer=setupMotionEvidenceLayer({THREE,world});
 headingLines=setupHeadingLines({THREE,world,origin:data.display_origin_lab,geometry,legendHost:$('headingLegend')});
 const g=buildVehicleGeometry(THREE,geometry);const edge=new THREE.EdgesGeometry(g,13);
 for(let car=0;car<2;car++){
  const group=new THREE.Group();const mat=new THREE.MeshStandardMaterial({color:car===0?0x459dff:0xffa24e,roughness:.6,metalness:.12,transparent:false,opacity:1,side:THREE.DoubleSide});group.add(new THREE.Mesh(g,mat));world.add(group);pred.push(group);
  const ref=new THREE.Group();ref.add(new THREE.LineSegments(edge,new THREE.LineBasicMaterial({color:0x60e5aa,transparent:true,opacity:.85,depthTest:false})));world.add(ref);truth.push(ref);
  for(const reference of [false,true]){
   const color=reference?'#60e5aa':car===0?'#69baff':'#ffb669';
   const markerCanvas=document.createElement('canvas');markerCanvas.width=32;markerCanvas.height=32;
   const ctx=markerCanvas.getContext('2d');ctx.beginPath();ctx.arc(16,16,reference?11:9,0,Math.PI*2);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=4;if(reference)ctx.stroke();else ctx.fill();
   const marker=new THREE.Points(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]),new THREE.PointsMaterial({color:0xffffff,map:new THREE.CanvasTexture(markerCanvas),size:reference?19:13,sizeAttenuation:false,transparent:true,depthTest:false,depthWrite:false}));
   marker.renderOrder=20;world.add(marker);(reference?referenceCenters:predictedCenters).push(marker);
   const label=document.createElement('span');label.className='center-tag '+(reference?'reference-tag':'predicted-tag');label.style.color=color;label.dataset.centerCar=car;label.dataset.centerRole=reference?'reference':'prediction';label.textContent=`车${car}·${reference?'真值':'预测'}中心`;holder.appendChild(label);centerLabels.push({label,marker,reference});
  }
  const line=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xffdf70,depthTest:false}));world.add(line);errors.push(line);
 }
 const grid=new THREE.GridHelper(40,40,0x52637b,0x26364c);grid.rotation.x=Math.PI/2;grid.position.z=-data.display_origin_lab[2];world.add(grid);
 new ResizeObserver(()=>{
  const w=holder.clientWidth,h=holder.clientHeight;
  renderer.setSize(w,h,false);
  if(sourceCameraMode)applySourceCamera();
  else if(sourceProjectionRow)sourceViewport=cameraHelpers.applyRecordedViewport(camera,renderer,sourceProjectionRow,w,h);
  else {camera.aspect=w/h;camera.updateProjectionMatrix();}
 }).observe(holder);
 const animate=()=>{requestAnimationFrame(animate);if(controls.enabled)controls.update();renderer.render(scene,camera);projectCenterLabels();};animate();overview();
}
function projectCenterLabels(){
 const holder=$('threeView');
 for(const {label,marker,reference} of centerLabels){
  const p=marker.getWorldPosition(new THREE.Vector3()).project(camera);
  const visible=marker.visible&&p.z>=-1&&p.z<=1&&Math.abs(p.x)<=1&&Math.abs(p.y)<=1;
  label.hidden=!visible;
  if(visible){const vp=sourceViewport||{x:0,y:0,w:holder.clientWidth,h:holder.clientHeight};label.style.left=(vp.x+(p.x+1)*vp.w/2)+'px';label.style.top=(holder.clientHeight-vp.y-(p.y+1)*vp.h/2)+'px';}
 }
}
function restoreFreeCamera() {
 sourceCameraMode=false;
 sourceViewport=null;
 sourceProjectionRow=null;
 if(renderer){
  const holder=$('threeView');
  if(cameraHelpers)cameraHelpers.restoreFreeViewport(camera,renderer,controls,holder.clientWidth,holder.clientHeight);
  else {
   renderer.setViewport(0,0,holder.clientWidth,holder.clientHeight);
   camera.aspect=holder.clientWidth/holder.clientHeight;
   camera.updateProjectionMatrix();
   controls.enabled=true;
  }
 }
 $('cameraStatus').textContent=recordedCameras?'已载入当前案例160条原始相机参数；自由观察视角，可点击恢复。':'原始相机参数不可用；请查看缺失提示或使用高级导入。';
}
function applySourceCamera() {
 const row=recordedCameras?.get(`${state.frame}:${state.car}`);
 const holder=$('threeView');
 sourceViewport=cameraHelpers.applyRecordedCamera(camera,renderer,controls,row,data.display_origin_lab,holder.clientWidth,holder.clientHeight);
 sourceProjectionRow=row;
 enableOrbitFromRecordedPose();
 state.follow=false;
 $('cameraStatus').dataset.projectionMatrix=JSON.stringify(camera.projectionMatrix.elements);
 $('cameraStatus').dataset.cameraPosition=JSON.stringify(camera.position.toArray());
 $('cameraStatus').dataset.cameraQuaternion=JSON.stringify(camera.quaternion.toArray());
 $('cameraStatus').textContent=`原图相机视角 · 车${state.car}缓存输入 · f${state.frame} · 按原图比例留边；可切总览／近看恢复旋转`;
}
// Fresh controls remove any damping left over from the previous free view.
// Their target lies on the calibrated optical axis, so enabling orbit does not
// move the camera or turn it toward a different point on the first render.
function enableOrbitFromRecordedPose() {
 const position=camera.position.clone();
 const orientation=camera.quaternion.clone();
 const direction=camera.getWorldDirection(new THREE.Vector3());
 const selected=toView(displayedPrediction(state.frame,state.car).pose||data.frames[state.frame].cars[state.car].reference.pose);
 const distance=Math.max(controls.minDistance,Math.min(controls.maxDistance,position.distanceTo(selected)));
 controls.dispose();
 controls=new OrbitControls(camera,renderer.domElement);
 controls.enableDamping=true;
 controls.minDistance=.8;
 controls.maxDistance=120;
 controls.target.copy(position).addScaledVector(direction,distance);
 camera.position.copy(position);
 camera.quaternion.copy(orientation);
 controls.update();
 controls.addEventListener('start',()=>{
  if(!sourceCameraMode)return;
  sourceCameraMode=false;
  state.follow=false;
  $('cameraStatus').textContent='已从原图相机开始自由旋转；点击原图相机可恢复记录视角。';
 });
}
function importCameras(document, activate=true) {
 if(!cameraHelpers)throw new Error('相机模块未加载；请重启本项目服务并刷新页面');
 recordedCameras=cameraHelpers.validateRecordedCameras(document,data);
 sourceCameraMode=activate;
 if(activate&&renderer)applySourceCamera();
 else restoreFreeCamera();
}
async function loadCameraHelpers() {
 // Optional calibration must never block the cached replay module graph.
 if(window.CWM_OFFLINE){
  const helpers=window.CWM_MODULES?.['recorded_camera.js'];
  if(!helpers)throw new Error('离线页面未包含相机模块，请使用完整新版页面');
  return helpers;
 }
 return import('./recorded_camera.js');
}
async function loadCaseCameras() {
 try {
  cameraHelpers=await loadCameraHelpers();
  importCameras(await json('/data/camera.json'),false);
 } catch(error) {
  recordedCameras=null;
  $('cameraStatus').textContent='原图相机暂不可用：'+error.message+'。回放继续；若刚更新代码，请关闭旧服务控制台、重启本项目入口后刷新。';
 }
}
function overview(){restoreFreeCamera();state.follow=false;const box=new THREE.Box3();for(const f of data.frames)for(const c of f.cars)box.expandByPoint(toView(c.versions.original.pose));box.expandByScalar(3);const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());const dist=Math.max(size.x,size.z,8)*1.25;camera.position.copy(center).add(new THREE.Vector3(dist*.7,dist*.8,dist*.95));controls.target.copy(center);controls.update();}
function closeup(){restoreFreeCamera();state.follow=true;const car=data.frames[state.frame].cars[state.car];const p=toView(displayedPrediction(state.frame,state.car).pose||car.reference.pose);controls.target.copy(p);camera.position.copy(p).add(new THREE.Vector3(5,3.5,6));controls.update();}
function update3D(frame,v){
 frame=displayedData().frames[frame.frame];v='displayed';
 const showPrediction=$('prediction').checked;
 if(headingLines)headingLines.update(frame,v,{showPrediction,showReference:$('truth').checked});
 for(const c of frame.cars){const p=c.versions[v];
  if(!Array.isArray(p.pose)){pred[c.car].visible=false;predictedCenters[c.car].visible=false;errors[c.car].visible=false;truth[c.car].position.set(...c.reference.pose.slice(0,3).map((x,i)=>x-data.display_origin_lab[i]));truth[c.car].quaternion.fromArray(c.reference.quaternion_xyzw).normalize();truth[c.car].visible=$('truth').checked;referenceCenters[c.car].position.copy(truth[c.car].position);referenceCenters[c.car].visible=$('centers').checked&&$('truth').checked;continue;}
  pred[c.car].visible=showPrediction;pred[c.car].position.set(...p.pose.slice(0,3).map((x,i)=>x-data.display_origin_lab[i]));pred[c.car].quaternion.fromArray(p.quaternion_xyzw).normalize();truth[c.car].position.set(...c.reference.pose.slice(0,3).map((x,i)=>x-data.display_origin_lab[i]));truth[c.car].quaternion.fromArray(c.reference.quaternion_xyzw).normalize();truth[c.car].visible=$('truth').checked;
  const showCenters=$('centers').checked;
  predictedCenters[c.car].position.copy(pred[c.car].position);referenceCenters[c.car].position.copy(truth[c.car].position);
  predictedCenters[c.car].visible=showCenters&&showPrediction;referenceCenters[c.car].visible=showCenters&&$('truth').checked;
  for(const entry of centerLabels.filter(entry=>Number(entry.label.dataset.centerCar)===c.car)){
   const pose=entry.reference?c.reference.pose:p.pose;entry.label.dataset.centerLab=JSON.stringify(pose.slice(0,3));entry.label.title=`车${c.car} ${entry.reference?'真值':'预测'}中心 [${pose.slice(0,3).map(x=>fmt(x,5)).join(', ')}] m`;
  }

  const pts=[new THREE.Vector3(...pred[c.car].position.toArray()),new THREE.Vector3(...truth[c.car].position.toArray())];errors[c.car].geometry.dispose();errors[c.car].geometry=new THREE.BufferGeometry().setFromPoints(pts);errors[c.car].visible=showPrediction&&$('truth').checked&&$('centers').checked;
 }
 if(motionLayer)motionLayer.update({prediction:frame.cars[state.car].versions[v],vehicle:pred[state.car],enabled:$('motionPoints').checked});
 if(state.follow&&Array.isArray(frame.cars[state.car].versions[v].pose)){const target=toView(frame.cars[state.car].versions[v].pose);const delta=target.clone().sub(controls.target);camera.position.add(delta);controls.target.copy(target);}
 world.updateMatrixWorld(true);if(sourceCameraMode)applySourceCamera();
}

const visiblePartitions=new Map();
let partitionEditor=null;
function partitionLegend(car,frame){
 const r=visiblePartitions.get(`${frame}:${car}`);
 if(!r)return '该帧尚无整车分区草案；请查看第27帧。未覆盖不代表背景。';
 return r.parts.filter(p=>p.pixels>0).map(p=>'<span><i style="background:'+p.color+'"></i>'+esc(p.label)+'</span>').join('')+'<small>'+(r.human_reviewed?'含手工修订；':'待核对：RGB提示点＋SAM；')+'灰色为未知或冲突，占预测车身 '+(100*r.unknown_pixels/r.whole_pixels).toFixed(1)+'%。覆盖完整仅指SAM整车掩码内部；尚未参与三维定位。</small>';
}
async function loadVisiblePartitions(){
 try{
  const doc=window.CWM_OFFLINE?.structure||await json('/data/structure.json');
  for(const r of doc.visible_partition?.records||[]){
   if(r.case!==data.case)continue;
   try{const saved=JSON.parse(localStorage.getItem('cwm_partition_edit:'+`${r.case}:${r.frame}:${r.car}`)||'null');if(saved&&typeof saved.overlay_data_url==='string'&&saved.overlay_data_url.startsWith('data:image/png;base64,')&&Array.isArray(saved.parts)){r.overlay_data_url=saved.overlay_data_url;r.parts=saved.parts;r.source=saved.source;r.human_reviewed='partial';r.unknown_pixels=r.parts.find(p=>p.name==='unknown')?.pixels||0;}}catch(error){console.warn('Stored partition correction unavailable',error);}
   const im=new Image();await new Promise((resolve,reject)=>{im.onload=resolve;im.onerror=reject;im.src=r.overlay_data_url;});
   visiblePartitions.set(`${r.frame}:${r.car}`,{...r,image:im});
  }
  if(visiblePartitions.size){
   $('maskSource').value='partition';
   try{const helpers=await import('./partition_editor.js');partitionEditor=helpers.setupPartitionEditor({canvas:$('rgb'),getRecord:()=>$('maskSource').value==='partition'?visiblePartitions.get(`${state.frame}:${state.car}`):null,redraw:()=>requestFrame(state.frame).catch(failure)});}catch(error){console.warn('Optional partition editor unavailable',error);}
   await requestFrame(state.frame);
  }
 }catch(error){console.warn('Optional visible partition unavailable',error);}
}

function drawMasks(ctx,frame,v,assets,transform){
 if($('maskSource').value==='partition'){
  for(const car of frame.cars){const r=visiblePartitions.get(`${frame.frame}:${car.car}`);if(!r)continue;const [x,y,w,h]=transform(r.crop_xyxy);ctx.globalAlpha=Number($('maskOpacity').value)/100;ctx.drawImage(r.image,x,y,w,h);}
  ctx.globalAlpha=1;return;
 }

 for(const car of frame.cars){const historicalOnly=car.car===0&&(displayedData().display_method===MOTION_METHOD||(data.case==='joint_fresh_07'&&frame.frame>=19&&['rigid_rgb_forward_case07','current_boundary_sets_case07'].includes(displayedData().display_method))||(data.case==='joint_fresh_07'&&frame.frame>=19&&frame.frame<=32&&displayedData().display_method==='case07_far_exterior_bt8')||(data.case==='joint_fresh_11'&&((frame.frame>=23&&frame.frame<=30&&displayedData().display_method==='exact_contour_refresh_bt8')||(frame.frame>=22&&frame.frame<=31&&displayedData().display_method==='contiguous_exact_contour_bt8'))));const b=car.observation.crop_xyxy;for(const p of car.observation.parts){const used=!historicalOnly&&p.participation[v].used;const key=p.mask_url;const fill=colored(assets.get(key),partColor(p.name));const edge=colored(assets.get(p.boundary_url),partColor(p.name));const [x,y,w,h]=transform(b);const alpha=Number($('maskOpacity').value)/100;ctx.globalAlpha=alpha*(used?1:.35);ctx.drawImage(fill,x,y,w,h);ctx.globalAlpha=Math.min(1,alpha*1.5)*(used?1:.35);ctx.drawImage(edge,x,y,w,h);}}ctx.globalAlpha=1;
}
function drawObservations(frame,car,v,rgb,assets){
 const isMotionCar=car.car===0&&displayedData().display_method===MOTION_METHOD;
 const isDiagnosticCar=data.case==='joint_fresh_07'&&car.car===0&&frame.frame>=19&&['rigid_rgb_forward_case07','current_boundary_sets_case07'].includes(displayedData().display_method);
 const isExteriorRepairCar=(data.case==='joint_fresh_11'&&car.car===0&&((frame.frame>=23&&frame.frame<=30&&displayedData().display_method==='exact_contour_refresh_bt8')||(frame.frame>=22&&frame.frame<=31&&displayedData().display_method==='contiguous_exact_contour_bt8')))||(data.case==='joint_fresh_07'&&car.car===0&&frame.frame>=19&&frame.frame<=32&&displayedData().display_method==='case07_far_exterior_bt8');
 const evidence=motionEvidence(displayedPrediction(frame.frame,car.car));
 updateMotionPanel({host:$('motionEvidencePanel'),summary:$('motionEvidenceSummary'),details:$('motionEvidenceDetails')},evidence);
 $('maskLegend').innerHTML=car.observation.parts.map(p=>'<span><i style="background:'+partColor(p.name)+'"></i>'+esc(partNames[p.name]||p.name)+(p.participation[v].used?'':'（本版未用于定位）')+'</span>').join('')+'<small>当前仅有这些部件的缓存掩码，尚非整车无空洞分区；未覆盖处不是背景判定。</small>';
 if($('maskSource').value==='partition')$('maskLegend').innerHTML=partitionLegend(car.car,frame.frame);
 $('maskOpacityValue').textContent=$('maskOpacity').value+'%';$('maskOpacity').disabled=!$('overlay').checked;
 // Compose offscreen, then replace each visible canvas without blanking it.
 const ctx=rgbBuffer.getContext('2d');ctx.globalAlpha=1;ctx.fillStyle='#080d13';ctx.fillRect(0,0,1920,1080);ctx.drawImage(rgb,0,0,1920,1080);if($('overlay').checked)drawMasks(ctx,{frame:frame.frame,cars:[car]},v,assets,b=>[b[0],b[1],b[2]-b[0],b[3]-b[1]]);
 if(structureComparison&&$('imageDebugOverlay').checked)structureComparison.draw(ctx,frame.frame,car.car);
 if(poseExperiments&&$('imageDebugOverlay').checked)poseExperiments.draw(ctx,frame.frame,car.car);
 if($('motionPoints').checked)drawMotionEvidence(ctx,evidence);
 const b=car.observation.crop_xyxy,w=b[2]-b[0],h=b[3]-b[1];
 const scale=Math.min(640/w,360/h),offsetX=(640-w*scale)/2,offsetY=(360-h*scale)/2;
 const cc=cropBuffer.getContext('2d');cc.globalAlpha=1;cc.fillStyle='#080d13';cc.fillRect(0,0,640,360);cc.drawImage(rgb,b[0],b[1],w,h,offsetX,offsetY,w*scale,h*scale);
 if($('overlay').checked)drawMasks(cc,{frame:frame.frame,cars:[car]},v,assets,box=>[offsetX+(box[0]-b[0])*scale,offsetY+(box[1]-b[1])*scale,(box[2]-box[0])*scale,(box[3]-box[1])*scale]);
 if(structureComparison&&$('imageDebugOverlay').checked){cc.save();cc.beginPath();cc.rect(offsetX,offsetY,w*scale,h*scale);cc.clip();cc.translate(offsetX-b[0]*scale,offsetY-b[1]*scale);cc.scale(scale,scale);structureComparison.draw(cc,frame.frame,car.car);cc.restore();}
 if(poseExperiments&&$('imageDebugOverlay').checked){cc.save();cc.beginPath();cc.rect(offsetX,offsetY,w*scale,h*scale);cc.clip();cc.translate(offsetX-b[0]*scale,offsetY-b[1]*scale);cc.scale(scale,scale);poseExperiments.draw(cc,frame.frame,car.car);cc.restore();}
 if($('motionPoints').checked){cc.save();cc.beginPath();cc.rect(offsetX,offsetY,w*scale,h*scale);cc.clip();drawMotionEvidence(cc,evidence,{origin:[b[0],b[1]],scale,offset:[offsetX,offsetY]});cc.restore();}
 $('rgb').getContext('2d').drawImage(rgbBuffer,0,0);$('crop').getContext('2d').drawImage(cropBuffer,0,0);if(partitionEditor)partitionEditor.draw($('rgb').getContext('2d'));
 $('rgbCaption').textContent=`原生1920×1080 · 车${car.car}缓存输入 · f${frame.frame} · 此车SAM／可选结构证据 · 三维双车同步；结构来源见下方`;
 const reason=d=>d.reason==='unresolved_surface_observation'?'无3×3稳定内部':d.reason.includes('visible')?'prior可见边界样本不足':d.reason.includes('empty')?'选中掩码为空':d.reason;
 $('parts').innerHTML=car.observation.parts.map(p=>{const s=p.participation[v],score=p.selected_score<1e-4?p.selected_score.toExponential(2):fmt(p.selected_score,3);return `<div class="partrow"><div><b>${esc(partNames[p.name]||p.name)}</b><div class="partmeta">${p.pixels}观测像素 · SAM index ${p.selected_index}</div></div><div class="${s.used?'used':'excluded'}">${s.used?'使用 · '+s.samples+'边界样本':'排除'}<div class="partmeta">${s.used?'':esc(s.excluded.map(reason).join('；')||'未参与')}</div></div><div class="partmeta">SAM分数 ${score}<br>${p.unavailable?'投影不可观测':'同一缓存选中mask'}</div></div>`;}).join('');
 if($('maskSource').value==='partition'){
  const record=visiblePartitions.get(`${frame.frame}:${car.car}`);
  $('parts').innerHTML=record?record.parts.filter(p=>p.pixels>0).map(p=>'<div class="partrow"><b>'+esc(p.label)+'</b><span>'+p.pixels+' 像素</span><span>待核对，未用于定位</span></div>').join(''):'本帧尚无完整分区草案，请查看第27帧。';
  $('rgbCaption').textContent=`第${frame.frame}帧 · 车${car.car} · RGB提示＋SAM分区草案；灰色为未知；未参与当前三维定位。`;
 }
 $('partsCaption').textContent=isMotionCar?(evidence?.initial_state_only?'本帧采用给定初始位置；后续运动使用像素跟踪＋近似三维深度。':evidence?'当前逐帧定位使用像素跟踪＋近似三维深度。':'本帧沿用旧过滤；此方法后续运动使用像素跟踪＋近似三维深度。')+'窗灯掩码及整车分区仅作历史诊断，不参与此方法的后续运动。':'颜色表示部件；较淡表示该观测未用于当前版本定位。SAM分数不是校准可信概率。掩码来自缓存，版本切换只改变其使用状态。';
 if(isMotionCar||isDiagnosticCar||isExteriorRepairCar){
  $('maskLegend').innerHTML=car.observation.parts.map(p=>'<span><i style="background:'+partColor(p.name)+'"></i>'+esc(partNames[p.name]||p.name)+'（历史诊断）</span>').join('')+'<small>窗灯SAM掩码／整车分区仅供回看，不参与此方法的后续运动。</small>';
  if($('maskSource').value==='partition')$('maskLegend').innerHTML=partitionLegend(car.car,frame.frame)+'<small>此处仅为历史分区诊断，不参与此方法的后续运动。</small>';
  $('parts').innerHTML=car.observation.parts.map(p=>'<div class="partrow"><b>'+esc(partNames[p.name]||p.name)+'</b><span>'+p.pixels+' 缓存像素</span><span>历史诊断 · 未用于后续运动</span></div>').join('');
  $('rgbCaption').textContent=`原生1920×1080 · 车0缓存输入 · f${frame.frame} · ${evidence?.initial_state_only?'给定第19帧初始位置；仅展示初始像素点':evidence?'像素跟踪＋近似三维深度；编号点为当前观测':'本帧沿用旧过滤，无新增运动证据'}。掩码／分区仅作历史诊断。`;
 }

 if(isDiagnosticCar){
  const boundary=displayedData().display_method==='current_boundary_sets_case07'&&frame.frame>=39&&frame.frame<=43;
  const input=frame.frame===19?'沿用07旧过滤19初值，非新估计':boundary?'本帧固定RGB双弧＋当前fresh SOURCE集合；展开所选实验冻结PNG核对':'继承失败的RGB前向轨迹，本帧没有新增边界估计';
  $('partsCaption').textContent='失败诊断，有pose可算但未通过科学验收。'+input+'。这里的窗灯SAM／分区仅为历史证据，不是本轮边界弧输入。';
  $('rgbCaption').textContent=`原生1920×1080 · 07车0 · f${frame.frame} · ${input}。窗灯掩码仅历史回看；无虚构材料点运动。`;
 }
 if(isExteriorRepairCar){
  $('partsCaption').textContent='当前是原外缘方法的数学修复候选：本帧沿用冻结外缘观测和固定anchors，不读取新mask。窗灯SAM／分区仅为历史回看，不是本轮外缘目标；调试点是当前帧观测与实际刷新脚点，不是跨帧材料运动。';
  $('rgbCaption').textContent=`原生1920×1080 · ${data.case==='joint_fresh_07'?'07':'11'}车0 · f${frame.frame} · 固定旧外缘观测＋修复后的当前SOURCE匹配；窗灯掩码仅历史诊断。`;
 }

}
function updateMetrics(frame,v){
 const view=displayedData();frame=view.frames[frame.frame];v='displayed';
 // Keep the disclosure itself mounted so frame/method updates preserve the reader's choice.
 $('caseTruthSummary').innerHTML=renderCaseTruth(caseTruth,data.case,frame.frame,frame.pairs[v],view.versions[v],caseTruthError,'summary');
 $('caseTruthStory').innerHTML=renderCaseTruth(caseTruth,data.case,frame.frame,frame.pairs[v],view.versions[v],caseTruthError,'story');
 $('metrics').innerHTML=frame.cars.map(c=>{const p=c.versions[v],m=valuesAt(view,frame.frame,c.car,v);return `<div class="metriccar"><strong class="${c.car===0?'blue':'orange'}">车${c.car} · ${esc(p.display_source)}</strong><div class="numbers"><div class="number"><small>中心位置误差</small><span>${fmt(m.position,3)} <em>cm</em></span></div><div class="number"><small>速度向量误差${m.velocity_mixed_source?"（方案衔接差分）":""}</small><span>${fmt(m.velocity,4)} <em>m/s</em></span></div><div class="number"><small>整体旋转误差</small><span>${fmt(m.rotation,3)} <em>°</em></span></div></div></div>`;}).join('');
 const pair=frame.pairs[v];$('pair').textContent=`当前画面：${pair.source}。预测${contactText(pair.prediction)} · 参考${contactText(pair.reference)}。三维最近表面欧氏间距：预测 ${fmt(pair.prediction?.distance_m,5)} m，参考 ${fmt(pair.reference?.distance_m,5)} m。${data.event_label}；详细定义和逐帧结果见图像正下方。`;
 const chosen=frame.cars[state.car].versions[v];$('fitDetails').textContent=JSON.stringify({case:data.case,frame:frame.frame,car:state.car,version:view.versions[v],source:chosen.display_source,pose_lab_xyz_rotvec_deg:chosen.pose,samples:chosen.samples,dropped:chosen.dropped,nfev:chosen.nfev,cost:chosen.cost,note:'实验缺记录沿用旧过滤，显式未知不绘制。cost涉及不同残差集合/权重，不能单独据其下降判断改善。'},null,2);
 $('modeLabel').textContent=view.versions[v];
 $('caseNotice').textContent=`参考真值仅用于绿色参考和误差验证，未生成预测。${data.event_label}。${data.case==='joint_fresh_07'?'物理碰撞记录从第41帧开始；几何评分与物理事件分开。':''}数字、中心图、误差标记与主三维采用同一当前画面方案。简化车身不是完整真实车壳；几何接触不等于事故识别。`;
}
function frameImageUrls(car){return [car.image_url,...new Set(car.observation.parts.flatMap(p=>[p.mask_url,p.boundary_url]))];}
function prefetchNextFrames(frameIndex,carId){
 for(let n=frameIndex+1;n<=Math.min(frameIndex+2,data.frames.length-1);n++)
  for(const url of frameImageUrls(data.frames[n].cars[carId]))image(url).catch(()=>{images.delete(url);});
}
async function requestFrame(n){
 const serial=++state.serial;state.rendering=true;
 const frame=data.frames[Math.max(0,Math.min(79,n))],v=state.version,car=frame.cars[state.car];
 // Keep the last complete frame visible while the next frame loads.
 $('sync').textContent='正在准备 f'+frame.frame+'…';
 try{
  const urls=frameImageUrls(car);const loaded=await Promise.all(urls.map(image));
  if(serial!==state.serial)return;
  const assets=new Map(urls.map((url,i)=>[url,loaded[i]]));
  state.frame=frame.frame;
  $('car').value=car.car;$('partCar').value=car.car;
  const observationVersion=poseExperiments?.selection()?.enabled?'candidate':v;
  drawObservations(frame,car,observationVersion,assets.get(car.image_url),assets);
  if(poseExperiments?.selection()?.enabled&&!(car.car===0&&displayedData().display_method===MOTION_METHOD))$('rgbCaption').textContent+=' 图中部件使用状态仅为旧过滤缓存；当前实验线索须看实验图像证据，不冒充已用于新拟合。';
  if(renderer)update3D(frame,v);
  updateMetrics(frame,v);updateInspection(frame.frame,'displayed',displayedData());if(structureComparison)structureComparison.update(frame,state.car);
  if(poseExperiments)poseExperiments.update(frame.frame,state.car);
  $('timeline').value=frame.frame;$('frameLabel').textContent=`帧 ${frame.frame} / 79`;
  $('timeLabel').textContent=`${fmt(frame.relative_time_s,2)} / ${fmt(data.frames[79].relative_time_s,2)} s`;
  $('sync').textContent=`同帧已同步 · f${frame.frame} · ${displayedData().versions.displayed}`;
  prefetchNextFrames(frame.frame,car.car);
 }catch(error){if(serial===state.serial)throw error;}
 finally{if(serial===state.serial)state.rendering=false;}
}
function stop(){state.playing=false;clearTimeout(timer);$('play').textContent='播放';$('play').classList.remove('playing');}
function schedule(){if(!state.playing)return;const next=state.frame+1;if(next>=80){stop();return;}const delay=(data.frames[next].time_s-data.frames[state.frame].time_s)*1000/state.rate;timer=setTimeout(async()=>{try{await requestFrame(next);schedule();}catch(e){stop();failure(e);}},delay);}
function wireControls(){
 $('displayMethod').onchange=()=>{stop();const selected=$('displayMethod').value;if(selected==='original'||selected==='candidate')state.version=selected;requestFrame(state.frame).catch(failure);};
 $('prediction').onchange=()=>{if(renderer)update3D(data.frames[state.frame],state.version);};
 $('otherSchemes3D').onchange=()=>{if(structureComparison)structureComparison.update(data.frames[state.frame],state.car);};
 $('maskSource').onchange=()=>requestFrame(state.frame).catch(failure);
 $('maskOpacity').oninput=()=>{$('maskOpacityValue').textContent=$('maskOpacity').value+'%';requestFrame(state.frame).catch(failure);};
 $('imageDebugOverlay').onchange=()=>requestFrame(state.frame).catch(failure);
 $('motionPoints').onchange=()=>requestFrame(state.frame).catch(failure);
 $('rawImageOnly').onclick=()=>{$('overlay').checked=false;$('imageDebugOverlay').checked=false;$('motionPoints').checked=false;requestFrame(state.frame).catch(failure);};
 $('sourceCamera').onclick=()=>{stop();if(recordedCameras){sourceCameraMode=true;if(renderer)applySourceCamera();}else{$('cameraStatus').textContent='当前案例原始相机参数不可用。请确认完整包包含config/replay_resources.json指定的JSON，或点击高级导入；未使用猜测相机。';}};
 $('cameraImport').onclick=()=>{$('cameraFile').click();};
 $('cameraFile').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>2000000)throw new Error('相机参数JSON不应超过2MB');importCameras(JSON.parse(await file.text()));}catch(error){$('cameraStatus').textContent='相机参数未导入：'+error.message;}finally{e.target.value='';}};
 $('play').onclick=()=>{if(state.playing){stop();return;}state.playing=true;$('play').textContent='暂停';$('play').classList.add('playing');if(state.frame===79){requestFrame(0).then(schedule).catch(failure);}else schedule();};
 const step=d=>{stop();requestFrame(state.frame+d).catch(failure);};$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('timeline').oninput=e=>{stop();requestFrame(Number(e.target.value)).catch(failure);};$('speed').onchange=e=>{state.rate=Number(e.target.value);if(state.playing){clearTimeout(timer);schedule();}};
 document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>{stop();requestFrame(Number(b.dataset.frame)).catch(failure);});
 const chooseCar=e=>{stop();state.car=Number(e.target.value);$('car').value=state.car;$('partCar').value=state.car;requestFrame(state.frame).catch(failure);};$('car').onchange=chooseCar;$('partCar').onchange=chooseCar;$('overlay').onchange=()=>requestFrame(state.frame).catch(failure);$('truth').onchange=()=>{if(renderer)update3D(data.frames[state.frame],state.version);};$('centers').onchange=()=>{if(renderer)update3D(data.frames[state.frame],state.version);};document.querySelector('#centerDiagrams').addEventListener('click',e=>{const button=e.target.closest('[data-center-focus]');if(!button)return;stop();state.car=Number(button.dataset.centerFocus);$('car').value=state.car;requestFrame(state.frame).then(()=>{if(renderer)closeup();}).catch(failure);});$('overview').onclick=()=>{if(renderer)overview();};$('closeup').textContent='近看并跟随';$('closeup').onclick=()=>{if(renderer)closeup();};
 document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}if(e.key==='ArrowRight'){e.preventDefault();step(1);}if(e.code==='Space'){e.preventDefault();$('play').click();}});
}
async function loadStructureComparison() {
 try {
  const helpers=window.CWM_OFFLINE?window.CWM_MODULES?.['structure_comparison.js']:await import('./structure_comparison.js');
  if(!helpers)throw new Error('可选结构模块未包含在离线包');
  structureComparison=helpers.setupStructureComparison({data,geometry,THREE,world,redraw:()=>requestFrame(state.frame).catch(failure)});
  await structureComparison.load();
 } catch(error) {
  structureComparison=null;
  const notice=document.createElement('section');notice.className='panel structure-comparison';notice.id='structureUnavailable';
  notice.textContent='结构比较暂不可用：'+error.message+'。原图、三维回放和原始相机继续可用；若刚更新代码，请重启本项目服务并刷新。';
  document.querySelector('#secondaryPanels').append(notice);
 }
}
async function loadPoseExperiments(){
 if(window.CWM_OFFLINE)return;
 try{
  const doc=await json('/data/pose-experiments.json');if(!doc.pose_experiments?.length)return;
  const helper=await import('./pose_experiments.js');
  poseExperiments=helper.setupPoseExperiments({doc,data,geometry,THREE,world,initialMethod:initialView(location.hash).method,redraw:()=>requestFrame(state.frame).catch(failure),navigate:(f,c)=>{stop();state.car=c;$('car').value=c;$('partCar').value=c;requestFrame(f).catch(failure);}});
  poseExperiments.update(state.frame,state.car);
  await requestFrame(state.frame);
 }catch(error){const note=document.createElement('p');note.className='caption';note.textContent='新增位姿实验暂不可用：'+error.message;document.querySelector('#secondaryPanels').append(note);}
}
async function loadPairGeometry(){try{const doc=await json('/data/pair-geometry.json');if(doc.case&&doc.case!==data.case)throw new Error('几何评分案例不匹配');pairGeometry=doc;await requestFrame(state.frame);}catch(error){console.warn('三维欧氏间距未加载：',error.message);}}
async function loadCaseTruth(){try{caseTruth=validateCaseTruth(await json('/data/case-truth.json'),data.case,data.frame_count);caseTruthError='';}catch(error){caseTruth=null;caseTruthError='原始物理记录暂不可用：'+error.message+'。物理事实保持未知，不以预测几何接触代替。';}await requestFrame(state.frame);}
async function start(){data=await json('/data/case.json');geometry=await json(data.geometry_url);if(data.frames.length!==80||data.car_count!==2)throw new Error('数据不符合冻结范围');const view=initialView(location.hash);applyInitialDisplayMethod(view.method);state.frame=view.frame??data.initial_frame;state.car=view.car??state.car;$('caseSelect').value=data.case==='joint_fresh_07'?'07':'11';$('caseSelect').disabled=!!window.CWM_OFFLINE;$('caseSelect').onchange=e=>{stop();location.assign('/cases/'+e.target.value+'/');};document.querySelector('.scope strong').textContent=data.event_label;document.querySelector('.scope span').textContent='r16 · '+data.case+' · 80帧双车';document.querySelector('.scope span:last-child').textContent=data.warning;document.querySelector('.focus').innerHTML='<span>重点</span>'+data.focus_frames.map(f=>`<button data-frame="${f}">${f}</button>`).join('');try{init3D();}catch(error){renderer=null;$('threeView').textContent='三维显示未启动：'+error.message+'。请在浏览器启用 WebGL；逐帧数值仍可核对。';}wireControls();loadStructureComparison();loadVisiblePartitions();loadPoseExperiments();loadPairGeometry();loadCaseTruth().catch(failure);await loadCaseCameras();updateInspection=setupInspection(data,n=>{stop();requestFrame(n).catch(failure);},geometry);await requestFrame(state.frame);}
start().catch(failure);





