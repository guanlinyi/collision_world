// Display-only evidence. Never reuses a point from a different frame or changes a pose.
export const MOTION_METHOD = 'rigid_rgb_forward';
const finiteVector = (value, n) => Array.isArray(value) && value.length === n && value.every(Number.isFinite);
export function motionEvidence(prediction) {
 return prediction?.display_key === MOTION_METHOD && prediction.motion_evidence?.kind === MOTION_METHOD ? prediction.motion_evidence : null;
}
export function motionPointColor(point) {
 return ({3:'#38bdf8',4:'#22d3ee',17:'#818cf8',18:'#2dd4bf',34:'#facc15',36:'#fb923c'})[point.id] || (point.category === 'front_plate' ? '#facc15' : '#38bdf8');
}
export function motionPixel(xy, {origin = [0,0], scale = 1, offset = [0,0]} = {}) {
 return [offset[0] + (xy[0] - origin[0]) * scale, offset[1] + (xy[1] - origin[1]) * scale];
}
export function drawableMotionPoints(evidence) {
 return (evidence?.points || []).filter(point => point.status !== 'lost' && finiteVector(point.image_xy, 2));
}
export function drawMotionEvidence(ctx, evidence, transform = {}) {
 ctx.save();
 for (const point of drawableMotionPoints(evidence)) {
  const [x,y] = motionPixel(point.image_xy, transform), color = motionPointColor(point);
  ctx.strokeStyle = color;ctx.fillStyle = color;ctx.lineWidth = 1.5;
  if (point.used_for_motion && finiteVector(point.previous_image_xy, 2)) {
   const previous = motionPixel(point.previous_image_xy, transform);
   ctx.beginPath();ctx.moveTo(...previous);ctx.lineTo(x,y);ctx.stroke();
  }
  ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);
  if (point.status === 'depth_unknown') ctx.stroke(); else ctx.fill();
  ctx.font = 'bold 12px sans-serif';ctx.lineWidth = 3;ctx.strokeStyle = '#07101e';ctx.strokeText(String(point.id),x+6,y-5);ctx.fillText(String(point.id),x+6,y-5);
 }
 ctx.restore();
}
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function updateMotionPanel({host, summary, details}, evidence) {
 host.hidden = !evidence;
 if (!evidence) {summary.textContent = '';details.innerHTML = '';return;}
 const points = evidence.points || [], used = points.filter(p => p.used_for_motion).length, lost = points.filter(p => p.status === 'lost').length;
 const unknown = points.filter(p => p.status === 'depth_unknown').length;
 summary.textContent = evidence.initial_state_only ? '第19帧为给定初始位置；图中为初始像素点，无三维落点。' : `本帧使用${used}个点，${lost}个跟踪失效${unknown ? `，${unknown}个深度未知` : ''}。细线是前帧到本帧的实际像素位移；三维小点仅为预测车上的近似落点。`;
 const statuses = {used:'用于本步运动',seed:'初始像素点',lost:'跟踪失效',depth_unknown:'深度未知，未用于运动'};
 details.innerHTML = points.map(p => `<div><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${motionPointColor(p)}"></i> ${escape(p.id)} · ${escape(p.label)}：${escape(statuses[p.status] || '未知')}<small> ${escape(p.reason)}</small></div>`).join('');
}
export function setupMotionEvidenceLayer({THREE, world}) {
 const group = new THREE.Group();group.name = 'motion-evidence-approximate';group.visible = false;world.add(group);
 const clear = () => {
  for (const child of [...group.children]) {group.remove(child);child.geometry.dispose();child.material.dispose();}
  group.visible = false;group.userData = {};
 };
 return {group, clear, update({prediction, vehicle, enabled = true}) {
  clear();
  const evidence = motionEvidence(prediction);
  if (!enabled || !vehicle?.visible || !evidence || evidence.initial_state_only) return;
  // Copy the very same LAB pose minus display origin used by the predicted car.
  // Both groups live under world, whose LAB-to-view rotation is shared.
  group.position.copy(vehicle.position);group.quaternion.copy(vehicle.quaternion);group.scale.copy(vehicle.scale);
  for (const point of evidence.points || []) {
   if (!point.used_for_motion || point.status !== 'used' || !finiteVector(point.local_xyz,3)) continue;
   const marker = new THREE.Points(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]),new THREE.PointsMaterial({color:motionPointColor(point),size:7,sizeAttenuation:false,depthTest:false,depthWrite:false}));
   marker.position.fromArray(point.local_xyz);marker.renderOrder = 25;marker.userData = {pointId:point.id,approximate:true};group.add(marker);
  }
  group.visible = group.children.length > 0;
  group.userData = {sourceFrame:evidence.source_frame,approximate:true};
 }};
}
