const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function indexRouteAgreement(document,caseId){
 if(!document)return null;
 if(document.schema!=='frozen_route_agreement_v1'||document.case!==caseId)throw new Error('两路诊断案例或规则版本不符');
 const rows=new Map();
 for(const r of document.records){if(r.case_id!==caseId||rows.has(r.frame))throw new Error('两路诊断重复或混入其他案例');rows.set(r.frame,r);}
 return {document,rows};
}
export function renderRouteAgreement(indexed,frame){
 const r=indexed?.rows.get(frame);if(!r)return '<h3>固定历史两路对照 · 随帧，不随显示方案</h3><p>至少一路缺测，接触证据不足。</p>';
 const status=v=>typeof v==='boolean'?(v?'接触':'分离'):'未知';const names={filtered_parts:'原过滤',historical_exterior:'历史外缘（含误报）'};
 return '<h3>固定历史两路对照 · 随帧，不随显示方案</h3><p><strong>'+esc(r.display_text)+'</strong></p><table><thead><tr><th>路线</th><th>3D共同松弛</th><th>当帧接触</th><th>来源／缺失原因</th></tr></thead><tbody>'+r.routes.map(route=>'<tr><td>'+names[route.route_id]+'</td><td>'+(Number.isFinite(route.gap_common_halfspace_slack_m)?(route.gap_common_halfspace_slack_m*100).toFixed(3)+' cm':'未知')+'</td><td>'+status(route.contact)+'</td><td>'+esc(route.unavailable_reason||route.source_path)+'<br><small>'+esc(route.contact_rule_id)+' · '+esc(route.source_hash)+'</small></td></tr>').join('')+'</tbody></table><p>冻结窗口同意覆盖9/10、未定1/10；这是同意覆盖，不是准确率或置信概率。原过滤10/10，与它相比没有更优。两路共享输入，可能共同出错；一致不证明真实状态。仅比较当帧输出，参考只作评分，不参与此映射。</p><p>这是透明诊断，非概率、融合模型或碰撞算法修复；07/40历史确定性误报仍保留，未定不解释成视角差或已修好。</p>';
}
