// Read-only experimental poses; never changes the selected default replay version.
import {indexRouteAgreement,renderRouteAgreement} from './route_agreement.js';
import {methodDescription,renderMethodStory,renderPriorityMethods} from './method_catalog.js';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=v=>Array.isArray(v)&&v.every(Number.isFinite);
export function indexPoseExperiments(doc,data){
 if(doc.case&&doc.case!==data.case)throw new Error('实验案例与回放不符');
 const methods=doc.pose_experiments||[];
 if(!Array.isArray(methods))throw new Error('pose_experiments必须是数组');
 const identities=new Set();
 return methods.map(method=>{
  if(!Array.isArray(method.records)||typeof method.id!=='string'||!method.id||identities.has(method.id))throw new Error('实验位姿列表缺少记录或编号重复');identities.add(method.id);
  const records=new Map();
  for(const r of method.records){
   if(!Number.isInteger(r.frame)||r.frame<0||r.frame>=data.frame_count||!Number.isInteger(r.car)||r.car<0||r.car>=data.car_count)throw new Error('实验位姿帧车越界');
   const unknownPair=method.render_scope==='pair'&&r.pose_status==='unknown'&&r.pose===null&&r.quaternion_xyzw===null;
   if(!unknownPair&&(!finite(r.pose)||r.pose.length!==6||!finite(r.quaternion_xyzw)||r.quaternion_xyzw.length!==4||Math.abs(Math.hypot(...r.quaternion_xyzw)-1)>.02))throw new Error('实验位姿或旋转无效');
   const key=`${r.frame}:${r.car}`;if(records.has(key))throw new Error('实验重复帧车记录');records.set(key,r);
  }
  return {...method,records};
 });
}
export function poseExperimentRenderRecords(method,frame,car){
 const rows=method.render_scope==='pair'?[...method.records.values()].filter(r=>r.frame===frame):[method.records.get(`${frame}:${car}`)];
 return rows.filter(r=>r&&finite(r.pose)&&r.pose.length===6).sort((a,b)=>a.car-b.car);
}
export function displayMethodGuidance(caseId,id){
 if(caseId==='joint_fresh_07'&&id==='case07_far_exterior_bt8')return "自身07整车外缘连续19～32帧，原10＋预定新增4；其他146个车／帧严格自身旧过滤。无11、失败前向或guarded20混源；原baseline新选anchors＋R/BT8，无平滑、无GT选范围、不varproj。24/27/28位置红格和24朝向红格清除，内部19～32速度无红，但18→19、32→33新增速度红格。19帧图像目标继续下降时位置反而由更准的早轮偏开，不能按GT改选轮；33旧Y偏差更明显。41首接触原本正确，这里只是保留正确，不是新救回。不是全片运动已解决，不改默认／优先。";
 if(caseId==='joint_fresh_11'&&id==='contiguous_exact_contour_bt8')return '连续22～31帧修复候选：22/31同原初值新求BT8，23～30八份R完整缓存；其余150个车／帧严格继承actual008，不是filtered或009。全80双车位置／方向／速度当前无红，但不是零误差、每帧最优或泛化证明；23/26/29中心和28速度退化，31→32约0.290599m/s接近0.3。原22/31三轮→BT8已披露，无平滑、无GT摘帧、不varproj；默认／优先不变。';
 if(caseId==='joint_fresh_11'&&id==='exact_contour_refresh_bt8')return '新增外缘数学修复候选，非默认/全片最佳：仅23～30新估计，其余旧filtered，不继承009。无平滑/无GT摘帧；位置5好3差，20/22/23/31速度仍红。展开当前方案说明看具体比较和衔接。';
 if(caseId==='joint_fresh_07'&&id==='rigid_rgb_forward_case07')return '失败诊断，非默认：19旧windshield_and_lamps初值，20～79前向；速度红格消失但累计位置/朝向偏差，41起接触漏报。0～18和车1旧过滤；有pose可算不等于通过验收。';
 if(caseId==='joint_fresh_07'&&id==='current_boundary_sets_case07')return '失败诊断，非默认：仅39～43有当前双弧纠偏，39复用原停止结果；其他19～79继承失败父轨迹。40误报、41～43恢复、44切回再漏，38→39/43→44速度误差约0.689/0.841m/s；窗灯SAM不是新弧输入。';
 if(id==='rigid_rgb_forward'&&caseId==='joint_fresh_11')return '19为给定008初值，不是新估计；20～79车0同一规则前向，0～18和车1旧过滤。速度红格4→0，但26/38中心及部分方向变差，初值/近似深度/已知标定依赖和累计漂移仍在，无事后平滑，不是每帧最优或默认替换。';
 if(id==='far_car_exterior_consensus'&&caseId==='joint_fresh_11')return '保留的008历史提示一致性控制：比007少一个速度红格（20帧），但21中心和22速度变差，31/32回退。20帧位置也回退，速度改善来自19→20差分。仍有22/23/25/31速度红格，使用27未来RGB模板，非在线或默认替换。';
 if(id==='far_car_exterior_extended'&&caseId==='joint_fresh_11')return '保留的007历史离线组合：车0第23～30帧采用两批冻结外缘结果，其余车／帧旧过滤，不是默认替换。位置红格为0，但22→23衔接速度变差、30→31旧风险仍在；还没把全段运动和碰撞判断都修好。';
 if(id==='far_car_exterior'&&caseId==='joint_fresh_11')return '这是保留的首批局部改进：仅车0第27～30帧，未处理24帧；30→31衔接速度风险已在这批出现。最新23～30组合另列，不覆盖这批历史。';
 if(id==='contact_reference')return '这是按真值摆放的参考答案，只能用来核查，不能当成预测效果。';
 if(id==='contact_window_exterior')return '这是保留的历史失败对照，不推荐作为当前结果：07案例第40帧提前报接触。仅覆盖39～43帧，不会修复27、28帧的大误差。';
 if(id==='original')return '原版是旧方法对照。全片优先看灯过滤；最新统一前向结果在11案例清除了速度红格，但仍有单帧中心／方向退化和累计漂移，不是默认替换。';
 if(id==='candidate')return '全片核对优先看已有灯过滤对照。11案例另有19起始统一前向结果，速度红格消失但仍有单帧退化和条件依赖；07历史外缘在第40帧误报，不推荐。';
 return '历史实验或窗口对照：保留它用于分析收益和失败，不代表当前推荐或全片最优。只覆盖选项标明的帧，其他位置沿用旧过滤。';
}
export function setupPoseExperiments({doc,data,geometry,THREE,world,redraw,navigate,initialMethod}){
 const methods=indexPoseExperiments(doc,data);if(!methods.length)return {update(){},draw(){},selection(){return null;},disable(){}};
 const priorityId=methods.some(m=>m.id==='rigid_rgb_forward')?'rigid_rgb_forward':methods.some(m=>m.id==='far_car_exterior_consensus')?'far_car_exterior_consensus':methods.some(m=>m.id==='far_car_exterior_extended')?'far_car_exterior_extended':methods.some(m=>m.id==='far_car_exterior')?'far_car_exterior':'rigid_rgb_forward';
 const agreement=indexRouteAgreement(doc.route_agreement,data.case);
 const section=document.createElement('section');section.className='panel';section.id='poseExperiments';
  section.innerHTML='<details id="historicalExperimentDetails"><summary>历史与控制实验 · 随帧／车，不随显示方案改写</summary><p>这里保留已经保存的结果和失败。多个灯过滤窗口是同一方法的对照；参考几何是真值评分，不是预测算法。运行时求解成功不代表通过科学验收。在上方选择历史实验后，主三维、逐帧数值、中心图与误差标记一起切换；无实验记录沿用旧过滤并明确标注。</p><label><input id="poseExperimentEvidence" type="checkbox" checked> 显示当前所选实验图像证据（还需开启上方调试线）</label><p id="poseExperimentConclusion"></p><div id="poseExperimentMetrics"></div></details>';
 document.querySelector('#secondaryPanels').append(section);
  const select=document.getElementById('displayMethod'),evidence=section.querySelector('#poseExperimentEvidence');
 const displayControls=document.querySelector('#displayControls');
 const contextCard=document.createElement('section');contextCard.id='methodContext';contextCard.className='panel';contextCard.innerHTML='<details id="methodStoryDetails"><summary>当前方案说明 · 故事、时间与优缺点</summary><div id="currentMethodStory"></div></details><details id="methodComparisonDetails"><summary>方案对照表 · 适用范围与已知问题</summary><div id="priorityMethodComparison"></div></details>';displayControls.after(contextCard);
 contextCard.querySelector('#priorityMethodComparison').innerHTML=renderPriorityMethods(data.case,priorityId);
 const historyLabel=document.createElement('label'),historyToggle=document.createElement('input');historyToggle.id='showHistoricalMethods';historyToggle.type='checkbox';historyToggle.checked=false;historyLabel.append(historyToggle,' 显示历史与控制实验选项');displayControls.append(historyLabel);
 const historyStatus=document.createElement('small');historyStatus.id='methodHistoryStatus';historyStatus.style.cssText='flex-basis:100%';displayControls.append(historyStatus);
 for(const control of [...section.querySelectorAll(':scope > label')])displayControls.append(control);
 const selectionLabel=document.createElement('strong');displayControls.prepend(selectionLabel);
 for(const option of select.options||[]){const entry=methodDescription(option.value);if(entry){option.textContent=entry.name;option.title=entry.how+' 范围：'+entry.scope+' 版本：'+entry.versionDate+' 结果：'+entry.resultDate;}}
 const groups=new Map(),groupOf=m=>m.id===priorityId?'最新前向结果（非默认替换）':'历史与控制实验（含真值评分）';
 for(const name of ['最新前向结果（非默认替换）','历史与控制实验（含真值评分）']){
  if(!methods.some(m=>groupOf(m)===name))continue;
  const group=document.createElement('optgroup');group.label=name;select.append(group);groups.set(name,group);
 }
 for(const m of methods){
  const groupName=groupOf(m);
  const option=document.createElement('option');option.value=m.id;
  const frames=[...new Set([...m.records.values()].map(r=>r.frame))].sort((a,b)=>a-b),cars=[...new Set([...m.records.values()].map(r=>r.car))].sort();
  const range=frames.length===1?`${frames[0]}帧`:frames.length===frames.at(-1)-frames[0]+1?`${frames[0]}～${frames.at(-1)}帧`:`${frames.join('、')}帧`;
  const entry=methodDescription(m.id);
  option.textContent=(entry?.name||m.label)+`｜${range}`;
  option.title=entry?entry.how+' 好处：'+entry.benefits+' 问题：'+entry.issues.map(([kind,text])=>kind+'：'+text).join('；')+' 范围：'+entry.scope+' 版本：'+entry.versionDate+' 结果：'+entry.resultDate:m.conclusion;
  if(!['far_car_exterior','far_car_exterior_extended','far_car_exterior_consensus','rigid_rgb_forward'].includes(m.id)&&cars.length===1)option.textContent+=`｜车${cars[0]}`;
  groups.get(groupName).append(option);
 }
 const guide=document.createElement('p');guide.id='displayMethodGuide';guide.className='caption';guide.style.cssText='flex-basis:100%;margin:2px 0';displayControls.append(guide);
 if(data.case==='joint_fresh_07'){const latest=document.createElement('a');latest.href='/cases/11/#frame=25&car=0&method=rigid_rgb_forward';latest.textContent='去11案例看统一前向结果：车0第25帧';displayControls.append(latest);}
  if(methods.some(m=>m.id===priorityId)&&navigate){const focusFrame=priorityId==='rigid_rgb_forward'?25:priorityId==='far_car_exterior_consensus'?20:priorityId==='far_car_exterior_extended'?24:27;const jump=document.createElement('button');jump.textContent='看最新前向结果：车0 · '+focusFrame+'帧';jump.onclick=()=>{select.value=priorityId;navigate(focusFrame,0);displayControls.scrollIntoView({block:'nearest',behavior:'smooth'});};displayControls.append(jump);}
  if(initialMethod&&(['original','candidate'].includes(initialMethod)||methods.some(m=>m.id===initialMethod)))select.value=initialMethod;
  const enabled=()=>methods.some(m=>m.id===select.value);
 const selectedMethod=()=>methods.find(m=>m.id===select.value)||null;
 const historical=()=>!['candidate',priorityId].includes(select.value);
 function syncHistoryOptions(){
  if(historical())historyToggle.checked=true;
  const show=historyToggle.checked;
  for(const option of select.options||[])if(option.value==='original'){option.hidden=!show;option.disabled=!show;}
  const group=groups.get('历史与控制实验（含真值评分）');if(group){group.hidden=!show;group.disabled=!show;}
  const details=section.querySelector('#historicalExperimentDetails');if(historical())details.open=true;
  historyStatus.textContent=historical()?'当前正在显示历史／控制方案，已显式保留该选项；若要收起，先选择灯过滤或最新局部改进。':'当前优先项已选中；历史选项可按需展开，收起不会更换当前方案。';
 }
 historyToggle.onchange=()=>{syncHistoryOptions();if(!historyToggle.checked)section.querySelector('#historicalExperimentDetails').open=false;};
 let frame=27,car=1;
 const metric=(r,key,scale=1,unit='')=>Number.isFinite(r?.metrics?.[key])?(r.metrics[key]*scale).toFixed(3)+unit:'—';
 function update(f,c){
   frame=f;car=c;selectionLabel.textContent=enabled()?'当前画面：'+selectedMethod().label+'（两车同步核查）':'当前画面：'+(select.value==='candidate'?'灯过滤 · 全片核对优先':'原版 · 旧对照');const method=selectedMethod();const row=method?.records.get(`${f}:${c}`);
  syncHistoryOptions();contextCard.querySelector('#currentMethodStory').innerHTML=renderMethodStory(select.value);
  guide.textContent=displayMethodGuidance(data.case,select.value);
  const renderRows=method?poseExperimentRenderRecords(method,f,c):[];
  evidence.disabled=!renderRows.some(r=>r.evidence?.wheel_xy?.length||r.evidence?.observed_xy?.length)&&!row?.evidence?.frozen_overlay_png_data_url;
  const statusText={seed_case07:'07旧过滤19初值（非新估计）',failed_forward_diagnostic:'前向失败诊断（pose可算，非验收成功）',boundary_diagnostic:'边界失败诊断（pose可算，非验收成功）',seed_initial:'沿用008的19初值（非新估计）',accepted_new:'新位置',fallback_old:'回退旧基线（非新位置）',unknown:'未知',baseline_filtered:'旧过滤对照',reference:'参考（仅评分）'};
  const pairStatus=method?.render_scope==='pair'?[...method.records.values()].filter(r=>r.frame===f).sort((a,b)=>a.car-b.car).map(r=>`car${r.car}：${statusText[r.pose_status]||'未知'}`).join('；'):'';
  section.querySelector('#poseExperimentConclusion').textContent=method?method.conclusion+(row?'':' 当前帧／车辆无此实验记录。')+(pairStatus?' 本帧双车：'+pairStatus+'。':row?.pose_status?' 当前所选车：'+statusText[row.pose_status]+'。':''):'当前显示为全片基线；下表只是固定历史对照。当前未选择实验，不显示所选实验图像证据或双车结果。';
  let table='<h3>固定历史实验对照 · 随帧／车，不随显示方案</h3><table><thead><tr><th>历史方案（当前所选车）</th><th>中心误差</th><th>旋转误差</th><th>最大车身边界误差</th><th>2D-SAT间隙误差</th><th>实际3D共同松弛</th><th>3D松弛误差</th><th>3D接触</th><th>位置来源</th></tr></thead><tbody>';
  for(const m of methods){const r=m.records.get(`${f}:${c}`);const contact=typeof r?.metrics?.contact_3d==='boolean'?(r.metrics.contact_3d?'接触':'未接触'):'未知';table+=`<tr><td>${esc(m.label)}</td><td>${metric(r,'center_error_m',100,' cm')}</td><td>${metric(r,'rotation_error_deg',1,'°')}</td><td>${metric(r,'max_boundary_error_m',100,' cm')}</td><td>${metric(r,'pair_gap_error_m',100,' cm')}</td><td>${metric(r,'source_convex_3d_slack_m',100,' cm')}</td><td>${metric(r,'source_convex_3d_slack_absolute_error_m',100,' cm')}</td><td>${contact}</td><td>${esc(statusText[r?.pose_status]||'无记录')}</td></tr>`;}
  const selectedRows=method?[...method.records.values()].filter(r=>r.frame===f).sort((a,b)=>a.car-b.car):[];let pairTable='';
  if(method?.render_scope==='pair'&&selectedRows.length){pairTable='<h3>所选实验：双车与参考证据</h3><p>参考3D共同松弛 '+metric(selectedRows[0],'reference_source_convex_3d_slack_m',100,' cm')+'；参考状态：'+(typeof selectedRows[0].metrics?.reference_contact_3d==='boolean'?(selectedRows[0].metrics.reference_contact_3d?'接触':'未接触'):'未知')+'。</p><table><thead><tr><th>车辆</th><th>中心误差</th><th>旋转误差</th><th>最大边界误差</th><th>位置来源</th></tr></thead><tbody>'+selectedRows.map(r=>`<tr><td>车${r.car}</td><td>${metric(r,'center_error_m',100,' cm')}</td><td>${metric(r,'rotation_error_deg',1,'°')}</td><td>${metric(r,'max_boundary_error_m',100,' cm')}</td><td>${esc(statusText[r.pose_status]||'未知')} ${esc(r.pose_status_detail||'')}</td></tr>`).join('')+'</tbody></table><p>评分来源：'+esc(method.source||'原冻结独立评分')+'</p>';}
  const frozenImage=row?.evidence?.frozen_overlay_png_data_url;
  const frozenEvidence=evidence.checked&&typeof frozenImage==='string'&&frozenImage.startsWith('data:image/png;base64,')?'<details class="frozenBoundaryEvidence"><summary>'+esc(row.evidence.frozen_overlay_summary||'查看本帧冻结双弧与最终SOURCE证据')+'</summary><p>'+esc(row.evidence.caption)+'</p><img style="max-width:100%;height:auto" alt="'+esc(row.evidence.caption)+'" src="'+frozenImage+'"><p>原件：'+esc(row.evidence.source_project_path)+'；SHA256：'+esc(row.evidence.source_sha256)+'</p></details>':'';
  section.querySelector('#poseExperimentMetrics').innerHTML=table+'</tbody></table>'+pairTable+frozenEvidence+renderRouteAgreement(agreement,f)+'<p>缺失保持未知，不等于未接触。旧2D-SAT是地面投影指标；3D共同半空间松弛≤0判SOURCE凸包采样接触，非欧氏间距或穿透深度，也不是事故识别。回退非新位置；绿色参考仅评分。尚无已校准概率，不显示百分比。窗口不能证明全片首碰或泛化。</p>';

 }
  evidence.onchange=()=>{update(frame,car);redraw();};
 function draw(ctx,f,c){
   if(!enabled()||!evidence.checked)return;const m=selectedMethod();
  for(const r of poseExperimentRenderRecords(m,f,c)){
  ctx.save();ctx.strokeStyle=m.color||'#32ffd2';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;
  const line=r.evidence?.wheel_xy;if(finite(line?.flat())&&line.length===2){ctx.beginPath();ctx.moveTo(...line[0]);ctx.lineTo(...line[1]);ctx.stroke();}
  for(const p of r.evidence?.observed_xy||[]){if(finite(p)&&p.length===2)ctx.fillRect(p[0]-1,p[1]-1,2,2);}
  if(r.evidence?.projected_xy?.length){
   const observed=r.evidence.observed_xy||[];ctx.fillStyle='#ffcc33';
   for(const p of observed)if(finite(p)&&p.length===2)ctx.fillRect(p[0]-1,p[1]-1,2,2);
   ctx.fillStyle=m.color||'#32ffd2';
   for(const p of r.evidence.projected_xy)if(finite(p)&&p.length===2)ctx.fillRect(p[0]-1,p[1]-1,2,2);
  }
  ctx.restore();
  }
 }
  update(frame,car);return {update,draw,selection:()=>({enabled:enabled(),method:selectedMethod()}),disable:()=>{select.value='original';}};
}
