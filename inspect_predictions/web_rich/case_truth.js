// Recorded simulation evidence, separate from the selected prediction method.
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=(value,digits=3)=>Number.isFinite(value)?value.toFixed(digits):'未知';
const contact=measure=>typeof measure?.contact==='boolean'?(measure.contact?'简化车身接触':'简化车身未接触'):'未知（未计算或缺测）';
export function validateCaseTruth(doc,caseId,frameCount=80){
 if(doc?.schema!=='cwm_recorded_case_truth_v1'||doc.case!==caseId||doc.frame_count!==frameCount||!Array.isArray(doc.records)||doc.records.length!==frameCount)throw new Error('原始仿真记录案例／帧数不匹配');
 const seen=new Set();
 for(const row of doc.records){if(!Number.isInteger(row.frame)||row.frame<0||row.frame>=frameCount||seen.has(row.frame))throw new Error('原始仿真记录帧号无效或重复');seen.add(row.frame);}
 return doc;
}
export function renderCaseTruth(doc,caseId,frame,pair,label,unavailableReason='',mode='full'){
 const row=doc?.case===caseId?doc.records.find(r=>r.frame===frame):null;
 const current=`<div data-truth-current-prediction><strong>当前预测 · ${esc(label)}</strong><p>${contact(pair?.prediction)} · 最近表面间距 ${Number.isFinite(pair?.prediction?.distance_m)?number(pair.prediction.distance_m*100)+' cm':'未知'}</p></div>`;
 if(!row){
  if(mode==='story')return `<p>${esc(unavailableReason||'该帧缺少原始仿真记录，不能根据预测接触替代物理事实。')}</p>`;
  return `<h2>物理事实与当前预测</h2><div data-recorded-truth><strong>${esc(caseId)} · 第${frame}帧 · 原始物理记录未知</strong>${mode==='summary'?'':`<p>${esc(unavailableReason||'该帧缺少原始仿真记录，不能根据预测接触替代物理事实。')}</p>`}</div>${current}`;
 }
 const first=doc.physical_outcome==='no_collision_recorded'?'全程没有记录到物理碰撞':Number.isInteger(doc.physical_first_collision_frame)?`首次物理碰撞：第${doc.physical_first_collision_frame}帧`:'首次物理碰撞帧未知';
 const event=typeof row.physical_event_this_frame==='boolean'?(row.physical_event_this_frame?'本帧有物理碰撞事件':'本帧没有新增物理碰撞事件'):'本帧物理事件未知';
 const seen=typeof row.physical_event_seen==='boolean'?(row.physical_event_seen?'截至本帧已经发生物理碰撞':'截至本帧尚未发生物理碰撞'):'截至本帧物理事件未知';
 if(mode==='summary')return `<h2>物理事实与当前预测</h2><div data-recorded-truth data-truth-frame="${frame}"><strong>${esc(caseId)} · 第${frame}帧 · ${first}</strong><p>${seen}。</p></div>${current}`;
 const brakes=(doc.brake_controls||[]).map((control,car)=>`车${car}：${Number.isInteger(control.brake_frame)?'第'+control.brake_frame+'帧开始刹车':'刹车帧未知'}`).join('；');
 const speed=[0,1].map(car=>`车${car} ${number(row.native_instant_speed_mps?.[car],5)} m/s`).join('；');
 const refDistance=Number.isFinite(row.reference_geometry_distance_m)?number(row.reference_geometry_distance_m*100)+' cm':'未知';
 const refContact=typeof row.reference_geometry_contact==='boolean'?(row.reference_geometry_contact?'接触':'未接触'):'未知';
 return `<h2>这段仿真实际发生了什么</h2><div data-recorded-truth data-truth-frame="${frame}"><strong>${esc(caseId)} · 第${frame}帧 · ${first}</strong><p>${esc(doc.narrative)}</p><p>${seen}。${esc(brakes)}</p><details><summary>本帧物理事件回调</summary><p>${event}。已发生的物理碰撞不会因后续某帧没有新增回调而取消。</p></details><p>原生瞬时速率：${speed}。这是仿真器当刻记录的速率，不同于下方按前一帧位移计算的差分速度。</p><p>按参考真值摆放的简化车身：间距 ${refDistance} · ${refContact}。这是同一SOURCE凸包的几何评分，不是精确物理车壳或物理碰撞传感器。</p><p class="caption">物理事实只随案例／帧变化，不随当前预测方案改变；物理碰撞来自原始仿真事件记录，和下面当前预测并列核对。</p></div>${current}`;
}
