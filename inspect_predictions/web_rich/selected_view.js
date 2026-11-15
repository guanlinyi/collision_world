// Recorded display selection only. No inference or truth-based selection.
export const poseStatusLabel = status => ({inherited_actual008:'继承已发布008（非本轮新估计）',seed_case07:'07旧过滤19初值（非新估计）',failed_forward_diagnostic:'前向失败诊断（有效pose，非验收成功）',boundary_diagnostic:'边界失败诊断（有效pose，非验收成功）',accepted_new:'新实验位置',seed_initial:'已有第19帧初始位置（非新估计）',fallback_old:'回退旧过滤（非新位置）',unknown:'未知 / 无有效位置',baseline_filtered:'旧过滤对照',reference:'参考自查（仅评分，非预测）'}[status] || '实验记录');
const finitePose = row => Array.isArray(row?.pose) && row.pose.length === 6 && row.pose.every(Number.isFinite) && Array.isArray(row.quaternion_xyzw);
export function deriveDisplayedData(data, baseline = 'original', selection = null, pairGeometry = null) {
  const method = selection?.enabled ? selection.method : null;
  const fallback = method ? 'candidate' : baseline;
  const label = method ? method.label : data.versions[baseline];
  const coveredFrames = method ? [...new Set([...method.records.values()].map(row => row.frame))].sort((a,b)=>a-b) : [];
  const coverageText = method ? method.display_coverage || `本实验有记录 ${coveredFrames.length} 帧（${coveredFrames.join('、')}）；其他帧／车沿用旧过滤，非新成绩。色条和排名包含这些旧过滤记录，不能作为新方案全片成绩。` : '全片使用所选原有基线。';
  const frames = data.frames.map(frame => {
    const cars = frame.cars.map(car => {
      const record = method?.records.get(`${frame.frame}:${car.car}`);
      const base = car.versions[fallback];
      const displayed = record ? {...record, solver_acceptable:finitePose(record) && record.pose_status !== 'unknown',
        display_source:`${method.label} · ${poseStatusLabel(record.pose_status)}`, display_key:method.id,
        center_error_m:record.metrics?.center_error_m, rotation_error_deg:record.metrics?.rotation_error_deg}
        : {...base, display_key:fallback, display_source:method ? '此实验该车/帧无记录，沿用旧过滤，非新成绩' : data.versions[baseline]};
      return {...car, versions:{...car.versions, displayed}};
    });
    const hasRecords = !!method && cars.some(car => method.records.has(`${frame.frame}:${car.car}`));
    const geometryRow = pairGeometry?.records?.find(row => row.frame === frame.frame);
    const measure = method && hasRecords ? geometryRow?.methods?.[method.id] : geometryRow?.baselines?.[fallback];
    const reference = geometryRow?.reference;
    const sourceRow = method && [...method.records.values()].find(row => row.frame === frame.frame);
    const pair = {prediction:measure || null, reference:reference || null,
      source:method ? hasRecords ? method.label : '无实验记录，沿用旧过滤' : data.versions[baseline],
      source_metrics:sourceRow?.metrics || null,
      projection_pair: !hasRecords ? frame.pairs[fallback] : null};
    return {...frame,cars,pairs:{...frame.pairs,displayed:pair}};
  });
  return {...data,versions:{...data.versions,displayed:label},frames,display_method:method?.id || null,display_coverage:coverageText};
}
export function contactText(measure) {
  if (!measure || typeof measure.contact !== 'boolean') return '未知（未计算或缺测）';
  return measure.contact ? '简化车身接触' : '简化车身未接触';
}
