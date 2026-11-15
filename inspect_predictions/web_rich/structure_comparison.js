import {caseDataUrl} from './case_routes.js';
import {buildVehicleGeometry} from './vehicle_geometry.js';
// Optional structure evidence viewer. No pose inference and no GT-derived observations.
const COLORS={observed:'#ff6b78',projected:'#bd8cff',match:'#ffd16b',original:'#e8edf5'};
const finite=values=>Array.isArray(values)&&values.every(Number.isFinite);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labelStatus=value=>({accepted:'接受',ok:'成功',success:'成功',rejected:'拒绝',failed:'失败',missing:'缺失',unavailable:'不可用'}[value]||value||'未记录');
export function indexStructureDocument(document,data){
 if(document?.available===false)return {records:new Map(),note:document.reason||'尚未配置结构比较数据',document};
 if(document?.schema!=='cwm_structure_comparison_v1'||!Array.isArray(document.records))throw new Error('结构比较schema/records不符');
 const records=new Map();
 for(const row of document.records){
  if((row.case||document.case)!==data.case)continue;
  if(!Number.isInteger(row.frame)||row.frame<0||row.frame>=data.frame_count||!Number.isInteger(row.car)||row.car<0||row.car>=data.car_count)throw new Error('结构比较帧/车辆越界');
  for(const field of ['observed_edges','projected_edges','matches'])if(row[field]!==undefined&&!Array.isArray(row[field]))throw new Error('结构比较边/匹配列表格式不符');
  if(row.observed_edges?.some(edge=>!edge||!Array.isArray(edge.points_xy))||row.projected_edges?.some(edge=>!edge||!Array.isArray(edge.points_xy)))throw new Error('结构比较边坐标格式不符');
  if(row.matches?.some(match=>!match))throw new Error('结构比较匹配格式不符');
  const key=`${row.frame}:${row.car}`;
  if(records.has(key))throw new Error('结构比较存在重复帧/车辆');
  records.set(key,row);
 }
 return {records,note:`当前案例覆盖 ${records.size} / ${data.frame_count*data.car_count} 个帧车状态；未覆盖处保持缺失。`,document};
}
export function candidateIsUsable(row){
 const c=row?.candidate;
 return !!c&&['accepted','ok','success'].includes(c.status||row.status)&&finite(c.pose)&&c.pose.length>=3&&finite(c.quaternion_xyzw)&&c.quaternion_xyzw.length===4&&Math.abs(Math.hypot(...c.quaternion_xyzw)-1)<.02;
}
export function structureDisplayEdges(row,{otherObserved=false}={}){
 const observed=row?.observed_edges||[],projected=row?.projected_edges||[];
 const isMatched=edge=>Array.isArray(edge.viewer_matched_source_ids)&&edge.viewer_matched_source_ids.length>0;
 const isVisibleFit=edge=>edge.used_in_fit===true&&(edge.prior_visible_fraction??1)>0;
 return {observed:observed.filter(edge=>otherObserved||isMatched(edge)),matchedObserved:observed.filter(isMatched),projected:projected.filter(isVisibleFit),rejected:projected.filter(edge=>!isVisibleFit(edge))};
}
function validPoint(p){return finite(p)&&p.length===2;}
export function fitStructureEdges(row){return (row?.projected_edges||[]).filter(edge=>edge.used_in_fit===true&&Array.isArray(edge.local_points_xyz)&&edge.local_points_xyz.length>=2&&edge.local_points_xyz.every(point=>finite(point)&&point.length===3));}
function metric(value,unit){return Number.isFinite(value)?`${value.toFixed(3)} ${unit}`:'缺失';}
export function buildRoadContextGroup(THREE,context,origin){
 if(!context||context.schema!=='cwm_road_context_v1'||!context.static_camera_verified)return null;
 const group=new THREE.Group();group.name='given_road_and_rgb_lane_candidates';
 const points=context.road?.world_polygon_lab_m;
 if(!Array.isArray(points)||points.length<3||points.some(p=>!finite(p)||p.length!==3))throw new Error('Invalid road polygon');
 const local=points.map(p=>p.map((v,i)=>v-origin[i]));
 const meshGeometry=new THREE.BufferGeometry();meshGeometry.setAttribute('position',new THREE.Float32BufferAttribute(local.flat(),3));
 const indices=[];for(let i=1;i<points.length-1;i++)indices.push(0,i,i+1);meshGeometry.setIndex(indices);meshGeometry.computeVertexNormals();
 const surface=new THREE.Mesh(meshGeometry,new THREE.MeshBasicMaterial({color:0x3b4554,side:THREE.DoubleSide,transparent:true,opacity:.75,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));surface.name='road_surface';group.add(surface);
 const lanes=new THREE.Group();lanes.name='rgb_lane_candidates';group.add(lanes);
 for(const item of context.lanes||[]){
  if(!Array.isArray(item.world_points_lab_m)||item.world_points_lab_m.some(p=>!finite(p)||p.length!==3))continue;
  const xyz=item.world_points_lab_m.map(p=>new THREE.Vector3(p[0]-origin[0],p[1]-origin[1],p[2]-origin[2]+.008));
  const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(xyz),new THREE.LineBasicMaterial({color:item.kind==='yellow'?0xffdc40:0xf4f7ff}));line.userData={source:item.source,reference_frame:context.reference_frame,display_z_offset_m:.008};lanes.add(line);
 }
 return group;
}
export function setupStructureComparison({data,geometry,THREE,world,redraw}){
 let indexed={records:new Map(),note:'正在检查可选结构比较数据…',document:null};
 let roadGroup=null;
 const section=document.createElement('section');section.className='panel structure-comparison';section.id='structureComparison';section.setAttribute('aria-label','RGB结构候选比较');
 section.innerHTML=`<div class="panel-title"><h2>固定历史侧窗结构对照 · 随帧，不随显示方案</h2><strong id="structureFrame"></strong></div>
 <p class="caption">结构实测边来自RGB检测，二维紫线为已知三维资产的配对起点投影；它不是结构候选的重投影。匹配连线只表示关联。真值仅用于误差核查，未作为候选输入。本次紫色新候选已组合灯过滤与侧窗结构；表中同时保留未过滤结构结果。不能只比原始窗灯就声称优于已有方法。上方“当前显示方案”统一控制彩色预测车；其他方案仅在开启三维调试总开关及具体图层后叠加。</p>
  <details open><summary>整车结构覆盖：设计范围不等于已经完成</summary><div id="wholeVehicleScope" class="table-scroll"></div></details>
 <div class="structure-options"><label><input id="roadSurface" type="checkbox" checked> 三维路面（给定标定）</label><label><input id="roadLanes" type="checkbox" checked> 三维道路线候选（来自RGB）</label></div><p id="roadContextStatus" class="caption">道路环境数据尚未载入。</p><div class="structure-options"><label class="structure-observed"><input id="structureObserved" type="checkbox" checked> 匹配使用的RGB边（红）</label><label class="structure-projected"><input id="structureProjected" type="checkbox" checked> 配对起点投影（紫）</label><label class="structure-match"><input id="structureMatches" type="checkbox" checked> 匹配连线（黄）</label><label class="structure-projected"><input id="structureCandidate" type="checkbox"> 结构候选三维（紫）</label><label><input id="structureOriginal" type="checkbox"> 原方案三维对照（白）</label><label class="structure-projected"><input id="structureLines" type="checkbox"> 三维拟合结构（候选紫实线 / 原方案紫虚线）</label></div>
 <details class="structure-advanced"><summary>高级：其他检测线与拒绝参考线</summary><div class="structure-options"><label><input id="structureOtherObserved" type="checkbox"> 显示车crop内其他RGB检测线（含背景，非可靠车身边界）</label><label><input id="structureRejected" type="checkbox"> 显示拒绝／背面参考投影（虚线，非真实可见观测）</label></div><p id="structureRejectedReasons" class="caption"></p></details><p class="caption">匹配线残差为同一批匹配SOURCE边界到RGB线的RMSE（各线等权），不是全车轮廓误差。拒绝候选显示未采用；其尝试及回退数值保留在记录中。</p><p id="structureStatus" class="caption" role="status"></p><p id="structureLoadStatus" class="caption" role="status"></p><p id="structureSources" class="caption"></p><div id="structureRows" class="table-scroll"></div><details><summary>当前车结构记录与失败原因</summary><pre id="structureDetails"></pre></details>`;
 document.querySelector('#secondaryPanels').append(section);
 const $=id=>document.getElementById(id);
 for(const input of section.querySelectorAll('input'))input.addEventListener('change',redraw);
 const candidateGroups=[],originalGroups=[],candidateStructures=[],originalStructures=[];
 if(world&&THREE){
  const mesh=buildVehicleGeometry(THREE,geometry);
  const edges=new THREE.EdgesGeometry(mesh,13);mesh.dispose();
  for(let car=0;car<data.car_count;car++){
   for(const [groups,color] of [[candidateGroups,COLORS.projected],[originalGroups,COLORS.original]]){
    const group=new THREE.Group();const hull=new THREE.LineSegments(edges,new THREE.LineBasicMaterial({color,transparent:true,opacity:.7,depthTest:false}));group.add(hull);group.userData.hull=hull;const structure=new THREE.Group();group.add(structure);(groups===candidateGroups?candidateStructures:originalStructures).push(structure);group.visible=false;world.add(group);groups.push(group);
   }
  }
 }
 function refreshStructure(group,row,reference){
  for(const child of [...group.children]){group.remove(child);child.geometry.dispose();child.material.dispose();}
  group.visible=$('structureLines').checked;
  for(const edge of fitStructureEdges(row)){
   const buffer=new THREE.BufferGeometry().setFromPoints(edge.local_points_xyz.map(point=>new THREE.Vector3(...point)));
   const options={color:COLORS.projected,transparent:true,opacity:reference?.65:1,depthTest:false};
   const material=reference?new THREE.LineDashedMaterial({...options,dashSize:.12,gapSize:.06}):new THREE.LineBasicMaterial(options);
   const line=new THREE.Line(buffer,material);line.userData={edge_id:edge.id,part_id:edge.part_id,source:edge.source,used_in_fit:true};line.renderOrder=8;if(reference)line.computeLineDistances();group.add(line);
  }
 }
 function rowFor(frame,car){return indexed.records.get(`${frame}:${car}`);}
 function draw(ctx,frame,car){
  const row=rowFor(frame,car);if(!row)return;
  const visible=structureDisplayEdges(row,{otherObserved:$('structureOtherObserved').checked});
  ctx.save();ctx.lineWidth=2.5;ctx.lineJoin='round';
  const box=row.crop_xyxy;if(finite(box)&&box.length===4){ctx.beginPath();ctx.rect(box[0],box[1],box[2]-box[0],box[3]-box[1]);ctx.clip();}
  for(const [name,toggle,color] of [['observed_edges','structureObserved',COLORS.observed],['projected_edges','structureProjected',COLORS.projected]]){
   if(!$(toggle).checked)continue;
   ctx.strokeStyle=color;
   for(const edge of name==='observed_edges'?visible.observed:visible.projected){const points=edge.points_xy;if(points.length<2||!points.every(validPoint))continue;ctx.beginPath();ctx.moveTo(...points[0]);for(const pt of points.slice(1))ctx.lineTo(...pt);ctx.stroke();}
  }
  if($('structureProjected').checked&&$('structureRejected').checked){ctx.strokeStyle=COLORS.projected;ctx.globalAlpha=.4;ctx.setLineDash([7,6]);for(const edge of visible.rejected){const points=edge.points_xy;if(!Array.isArray(points)||points.length<2||!points.every(validPoint))continue;ctx.beginPath();ctx.moveTo(...points[0]);for(const pt of points.slice(1))ctx.lineTo(...pt);ctx.stroke();}ctx.setLineDash([]);ctx.globalAlpha=1;}
  if($('structureMatches').checked){ctx.strokeStyle=COLORS.match;ctx.fillStyle=COLORS.match;ctx.lineWidth=1.5;ctx.setLineDash([5,4]);for(const match of row.matches||[]){if(!validPoint(match.observed_xy)||!validPoint(match.projected_xy))continue;ctx.beginPath();ctx.moveTo(...match.observed_xy);ctx.lineTo(...match.projected_xy);ctx.stroke();ctx.beginPath();ctx.arc(...match.observed_xy,3,0,2*Math.PI);ctx.fill();}}
  ctx.restore();
 }
 function update(frame,selectedCar){
  const showOtherSchemes=$('otherSchemes3D')?.checked===true;
  if(roadGroup){roadGroup.getObjectByName('road_surface').visible=$('roadSurface').checked;roadGroup.getObjectByName('rgb_lane_candidates').visible=$('roadLanes').checked;}
  $('structureFrame').textContent=`f${frame.frame} · 图像车${selectedCar}`;
  const provenance=indexed.document?.provenance;
  $('structureLoadStatus').textContent=indexed.note+(provenance?' 数据说明：'+(typeof provenance==='string'?provenance:JSON.stringify(provenance)):'');
  const selected=rowFor(frame.frame,selectedCar);
  const reason=selected?.failure_reason||selected?.reason||'';
  const reasons={fewer_than_3_unambiguous_RGB_lines:'无歧义RGB线少于3条',insufficient_image_span:'图像覆盖范围不足',insufficient_line_orientation_diversity:'线方向种类不足',no_successful_structure_fit:'没有成功的结构拟合'};
  $('structureStatus').textContent=selected?`当前图像车${selectedCar}结构候选：${labelStatus(selected.candidate?.status||selected.status)}。${reason?'原因：'+String(reason).split(';').map(item=>reasons[item.trim()]||item.trim()).join('；')+'。':''}${candidateIsUsable(selected)?'候选三维可显示。':'候选未采用；紫色新候选三维保持隐藏，误差表不把回退值当作成功候选。'}`:'当前图像车辆结构数据缺失。';
  const sourceNames=name=>[...new Set((selected?.[name]||[]).map(edge=>edge.source).filter(Boolean).map(s=>typeof s==='string'?s:JSON.stringify(s)))].join('；')||'缺失';
  $('structureSources').textContent=selected?`红边来源：${sourceNames('observed_edges')}。紫线来源：${sourceNames('projected_edges')}。${selected.matches?.length||0}条匹配。`:'当前帧/车没有结构记录；原图、原三维与原有误差照常显示。';
  const display=structureDisplayEdges(selected,{otherObserved:$('structureOtherObserved').checked});
  if(selected)$('structureSources').textContent+=` 默认关联RGB线 ${display.matchedObserved.length}/${selected.observed_edges?.length||0}；可见且参与的资产投影 ${display.projected.length}/${selected.projected_edges?.length||0}。其他检测线和拒绝／背面投影默认隐藏；全部结构叠加限于车crop。`;
  $('structureRejectedReasons').textContent=selected?display.rejected.map(edge=>`${edge.id}：${edge.rejection_reason||edge.match_status||'未参与'}；prior可见比例 ${Number.isFinite(edge.prior_visible_fraction)?edge.prior_visible_fraction.toFixed(3):'未注明'}`).join('； '):'当前帧无结构记录。';
  const poseRole=selected?.projection_pose_role||selected?.projected_edges?.[0]?.pose_role||'未注明';
  if(selected)$('structureSources').textContent+=` 二维投影姿态来源：${poseRole}（配对参考，不能当作新候选的重投影）；三维紫实线则随已接受候选pose摆放，紫虚线随原方案pose摆放。`;
  const structureEdges=fitStructureEdges(selected);
  $('structureSources').textContent+=structureEdges.length?` 三维拟合结构：${structureEdges.length}条真实SOURCE局部折线（紫实线候选 / 紫虚线原方案）。`:' 三维拟合结构缺失：未提供used_in_fit=true且具备真实local_points_xyz的边；未从二维反造三维。';
  const headings=[['center_error_m','中心 / m'],['heading_error_deg','车头方向 / °'],['boundary_error_px','匹配线残差 / px'],['velocity_error_mps','运动向量 / m/s']];
  let html='<table><thead><tr><th>车辆 / 方案</th><th>状态</th>'+headings.map(([,name])=>`<th>${name}</th>`).join('')+'</tr></thead><tbody>';
  for(const car of frame.cars){
   const row=rowFor(frame.frame,car.car);
   for(const [key,name] of [['original','原始窗灯'],['lamp_filtered','已有灯过滤'],['unfiltered_structure','侧窗结构（未滤灯）'],['candidate','灯过滤＋侧窗结构']]){
    const lamp=car.versions.candidate||{};
    const metrics=key==='lamp_filtered'?{center_error_m:lamp.center_error_m,heading_error_deg:lamp.rotation_error_deg,velocity_error_mps:lamp.velocity_error_mps}:row?.metrics?.[key];
    const status=key==='lamp_filtered'?'已有缓存结果':!row?'未覆盖 / 缺失':key==='candidate'?labelStatus(row.candidate?.status||row.status):labelStatus(row.original_status||'缓存结果');
    const adopted=key==='unfiltered_structure'?candidateIsUsable({candidate:row?.unfiltered_structure_candidate}):key!=='candidate'||candidateIsUsable(row);
    html+=`<tr><th>车${car.car} · ${name}</th><td>${esc(status)}</td>`+headings.map(([key])=>`<td>${row&&!adopted?'未采用':metric(metrics?.[key],'')}</td>`).join('')+'</tr>';
   }
   if(candidateGroups[car.car]){
    const group=candidateGroups[car.car];group.visible=showOtherSchemes&&$('structureCandidate').checked&&candidateIsUsable(row);
    if(group.visible){group.position.set(...row.candidate.pose.slice(0,3).map((x,i)=>x-data.display_origin_lab[i]));group.quaternion.fromArray(row.candidate.quaternion_xyzw).normalize();}
    refreshStructure(candidateStructures[car.car],row,false);
    const original=originalGroups[car.car],pose=car.versions.original;original.visible=showOtherSchemes&&($('structureOriginal').checked||($('structureLines').checked&&fitStructureEdges(row).length>0));original.userData.hull.visible=$('structureOriginal').checked;original.position.set(...pose.pose.slice(0,3).map((x,i)=>x-data.display_origin_lab[i]));original.quaternion.fromArray(pose.quaternion_xyzw).normalize();refreshStructure(originalStructures[car.car],row,true);
   }
  }
  $('structureRows').innerHTML=html+'</tbody></table>';
  $('structureDetails').textContent=JSON.stringify(selected||{frame:frame.frame,car:selectedCar,status:'missing',reason:indexed.note},null,2);
 }
 async function load(){
  try{
   let doc;
   if(window.CWM_OFFLINE)doc=window.CWM_OFFLINE.structure||{available:false,reason:'离线包未收录结构比较数据；原回放继续。'};
   else {const response=await fetch(caseDataUrl('/data/structure.json'));if(!response.ok)throw new Error(`HTTP ${response.status}`);doc=await response.json();}
   indexed=indexStructureDocument(doc,data);
   const scope=doc.vehicle_scope?.regions||[];
   $('wholeVehicleScope').innerHTML=scope.length?'<table><thead><tr><th>部位</th><th>目前状态</th><th>要利用的依据</th></tr></thead><tbody>'+scope.map(r=>`<tr><th>${esc(r.name)}</th><td>${esc(r.status)}</td><td>${esc(r.basis)}</td></tr>`).join('')+'</tbody></table>':'暂无完整结构清单；不代表这些部位已经实现。';
   const road=doc.road_context;
   $('roadContextStatus').textContent=road?`路面来自既有标定；${road.lanes.length}段标线候选（同一标线可有重复检测）来自第${road.reference_frame}帧RGB颜色／直线检测候选投到给定路面，固定相机下作为环境参照显示。不是道路真值，不延伸遮挡段，尚未加入本次车辆拟合。`:'缺少道路显示数据。';
   if(world&&THREE&&road){if(roadGroup)world.remove(roadGroup);roadGroup=buildRoadContextGroup(THREE,road,data.display_origin_lab);if(roadGroup)world.add(roadGroup);}
  }catch(error){indexed={records:new Map(),note:'结构比较暂不可用：'+error.message+'。原回放继续。',document:null};}
  redraw();
 }
 return {draw,update,load};
}








