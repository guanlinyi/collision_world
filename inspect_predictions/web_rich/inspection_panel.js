import {valuesAt, buildIndex, rankFrames, compareMetricToCandidate, metricInfo, thresholdsDefault, projectedFootprint, rotateVector, headingDegrees, bodyRelativeOffset} from './frame_values.js';
import {contactText} from './selected_view.js';
const $ = id => document.getElementById(id);
const fmt = (value, digits = 5) => !Number.isFinite(value) ? '—' : value.toFixed(digits);
const signed = value => value === null ? '—' : `${value > 0 ? '+' : ''}${fmt(value)}`;
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const frameRanges = frames => {
  if (!frames.length) return '无';
  const ranges = [];
  for (let i = 0; i < frames.length; i++) {
    const start = frames[i];
    while (i + 1 < frames.length && frames[i + 1] === frames[i] + 1) i++;
    ranges.push(start === frames[i] ? String(start) : `${start}–${frames[i]}`);
  }
  return ranges.join('、');
};
export function setupInspection(initialData, onJump, geometry) {
  let data=initialData;
  let version = 'original', frame = 27, index = [], thresholds = {...thresholdsDefault};
  const maximumOffsetCm=Math.max(...data.frames.flatMap(f=>f.cars.flatMap(c=>Object.values(c.versions).flatMap(v=>v.pose.slice(0,3).map((x,i)=>Math.abs(x-c.reference.pose[i])*100)))));
  const diagramExtentCm=Math.max(10,Math.ceil(maximumOffsetCm/5)*5);
  const footprintExtentM=Math.ceil((Math.max(...geometry.vertices.map(v=>Math.hypot(...v)))+maximumOffsetCm/100)*2)/2;
  function footprintSvg(car,color){
    const record=data.frames[frame].cars[car],prediction=record.versions[version],reference=record.reference;
    if(!Array.isArray(prediction.pose))return '<p class=caption>当前所选方案没有有效位姿，轮廓和中心图不绘制。</p>';
    const scale=110/footprintExtentM,point=p=>`${150+p[0]*scale},${150-p[1]*scale}`;
    const referenceHull=projectedFootprint(geometry.vertices,reference.pose,reference.quaternion_xyzw,reference.pose);
    const predictedHull=projectedFootprint(geometry.vertices,prediction.pose,prediction.quaternion_xyzw,reference.pose);
    const delta=prediction.pose.slice(0,3).map((value,i)=>value-reference.pose[i]);
    const arrow=(offset,q,stroke,id,dashed)=>{const forward=rotateVector([1.8,0,0],q);return `<line class="heading-arrow" x1="${150+offset[0]*scale}" y1="${150-offset[1]*scale}" x2="${150+(offset[0]+forward[0])*scale}" y2="${150-(offset[1]+forward[1])*scale}" stroke="${stroke}" ${dashed?'stroke-dasharray="5 3"':''} marker-end="url(#${id})"/>`;};
    const marker=(id,fill)=>`<marker id="${id}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="${fill}"/></marker>`;
    const predHeading=headingDegrees(prediction.quaternion_xyzw),refHeading=headingDegrees(reference.quaternion_xyzw),angle=((predHeading-refHeading+540)%360)-180;
    const body=bodyRelativeOffset(prediction.pose,reference.pose,reference.quaternion_xyzw);
    const predId=`pred-arrow-${car}`,refId=`ref-arrow-${car}`;
    return `<div class="footprint-block" data-heading-error="${angle}" data-left-cm="${body.leftCm}" data-forward-cm="${body.forwardCm}"><h4>整车俯视 · 轮廓与车头方向</h4><svg class="footprint-svg" viewBox="0 0 300 300" role="img" aria-label="车${car}预测与真值的地面投影轮廓和车头方向，朝向差${fmt(angle,3)}度"><defs>${marker(predId,color)}${marker(refId,'#60e5aa')}</defs><g class="center-grid"><path d="M40 150H260M150 40V260"/></g><text x="268" y="155">+X</text><text x="154" y="29">+Y</text><polygon class="reference-footprint" points="${referenceHull.map(point).join(' ')}" stroke="#60e5aa"/><polygon class="prediction-footprint" points="${predictedHull.map(point).join(' ')}" stroke="${color}" fill="${color}"/>${arrow([0,0,0],reference.quaternion_xyzw,'#60e5aa',refId,true)}${arrow(delta,prediction.quaternion_xyzw,color,predId,false)}<line class="center-offset" x1="150" y1="150" x2="${150+delta[0]*scale}" y2="${150-delta[1]*scale}"/><circle cx="150" cy="150" r="5" class="truth-center-dot"/><circle cx="${150+delta[0]*scale}" cy="${150-delta[1]*scale}" r="3" fill="${color}"/><text x="40" y="280">刻度范围 ±${footprintExtentM} m</text></svg><p class="caption">彩色实线＝预测；绿色虚线＝真值。箭头指向车头（资产局部 +X）。轮廓为当前三维凸包沿Z方向投影，无角度或偏移夸大。</p><p class="heading-values">预测朝向 ${fmt(predHeading,3)}° · 真值 ${fmt(refHeading,3)}°<br>有符号朝向差 ${signed(angle)}°（正值＝图中逆时针）<br>按真值车头方向：${body.leftCm>=0?'左':'右'}偏 ${fmt(Math.abs(body.leftCm),3)} cm，${body.forwardCm>=0?'前':'后'}偏 ${fmt(Math.abs(body.forwardCm),3)} cm</p></div>`;
  }
  function renderCenters(){
    $('centerFrame').textContent=`第 ${frame} 帧 · ${data.versions[version]}`;
    const svgCenter=150,pixelsPerCm=110/diagramExtentCm;
    $('centerDiagrams').innerHTML=[0,1].map(car=>{
      const v=valuesAt(data,frame,car,version);
      if(v.missing)return `<article class="center-card"><h3>车${car} · 位置未知</h3><p>${v.source}</p></article>`;
      const delta=v.poseDelta.slice(0,3).map(x=>x*100);
      const x=svgCenter+delta[0]*pixelsPerCm,y=svgCenter-delta[1]*pixelsPerCm;
      const color=car===0?'#69baff':'#ffb669';
      const labelX=x>svgCenter?x-8:x+8,labelY=y<svgCenter?y-10:y+18;
      const grid=[-1,-.5,0,.5,1].map(t=>{const p=svgCenter+t*110;return `<line x1="${p}" y1="40" x2="${p}" y2="260"/><line x1="40" y1="${p}" x2="260" y2="${p}"/>`;}).join('');
      return `<article class="center-card" data-center-car="${car}" data-delta-cm="${delta.join(',')}" data-error-cm="${v.position}"><div class="panel-title"><h3 class="${car===0?'blue':'orange'}">车${car} · 中心误差 ${fmt(v.position,3)} cm</h3><button data-center-focus="${car}">三维近看车${car}</button></div>
      <div class="center-views">${footprintSvg(car,color)}<div class="center-zoom-block"><h4>中心放大 · 厘米刻度</h4><div class="center-card-body"><svg viewBox="0 0 300 300" role="img" aria-label="车${car}中心偏移俯视示意，X差${fmt(delta[0],3)}厘米，Y差${fmt(delta[1],3)}厘米，刻度范围正负${diagramExtentCm}厘米"><g class="center-grid">${grid}</g><g class="center-axes"><path d="M40 150H265 M150 260V35"/></g><text x="270" y="155">+X</text><text x="154" y="28">+Y</text><text x="40" y="280">−${diagramExtentCm} cm</text><text x="223" y="280">+${diagramExtentCm} cm</text><line class="center-offset" x1="150" y1="150" x2="${x}" y2="${y}"/><circle class="truth-center-dot" cx="150" cy="150" r="6"/><circle class="pred-center-dot" cx="${x}" cy="${y}" r="4" fill="${color}"/><text x="158" y="169" class="truth-center-label">真值中心 (0,0)</text><text x="${labelX}" y="${labelY}" text-anchor="${x>svgCenter?'end':'start'}" fill="${color}">预测中心</text></svg>
      <div class="center-description"><p>● 彩色：预测中心<br><span class="green">○ 绿色：真值中心</span><br><span class="yellow">黄线：两中心的俯视连线</span></p><p class="center-deltas">ΔX：${signed(delta[0])} cm<br>ΔY：${signed(delta[1])} cm<br>ΔZ：${signed(delta[2])} cm</p><p class="caption">总误差 √(ΔX² + ΔY² + ΔZ²)<br>＝${fmt(v.position,3)} cm<br>两车／所有帧使用相同厘米刻度。<br>X/Y 为 LAB 世界坐标，不代表车头／车尾方向。</p></div></div></div></div></article>`;
    }).join('');
  }

  const row = (label, unit, predicted, reference) => `<tr><th scope="row">${label} <small>${unit}</small></th><td>${fmt(predicted)}</td><td>${fmt(reference)}</td><td>${signed(predicted === null || reference === null ? null : predicted - reference)}</td></tr>`;
  function renderValues() {
    renderCenters();
    $('valueFrame').textContent = `第 ${frame} 帧 · ${fmt(data.frames[frame].relative_time_s, 2)} 秒 · ${data.versions[version]}`;
    $('vehicleValues').innerHTML = [0, 1].map(car => {
      const v = valuesAt(data, frame, car, version), flagged = index[frame].cars[car];
      const summary = flagged.reasons.map(key => metricInfo[key].label).join('、');
      let rows = ['X', 'Y', 'Z'].map((axis, i) => row(`中心 ${axis}`, 'm', v.predictedPose[i], v.referencePose[i])).join('');
      rows += ['X', 'Y', 'Z'].map((axis, i) => row(`旋转向量 ${axis}`, '°', v.predictedPose[i + 3], v.referencePose[i + 3])).join('');
      rows += ['X', 'Y', 'Z'].map((axis, i) => row(`速度 ${axis}`, 'm/s', v.predictedVelocity?.[i] ?? null, v.referenceVelocity?.[i] ?? null)).join('');
      rows += row('速率（速度大小）', 'm/s', v.predictedSpeed, v.referenceSpeed);
      return `<article class="vehicle-values ${flagged.flagged ? 'flagged' : ''}" data-car="${car}"><div class="panel-title"><h3 class="${car === 0 ? 'blue' : 'orange'}">车${car}</h3><span>${v.source||data.versions[version]}</span><span class="flag-state">${v.missing ? '缺测：不填零、不判未接触' : !v.accepted ? '求解未接受' : summary ? '超过查看阈值：' + summary : '未超过查看阈值'}</span></div><div class="error-totals"><span>中心位置 <b>${fmt(v.position, 3)} cm</b></span><span>整体朝向 <b>${fmt(v.rotation, 3)}°</b></span><span>速度向量 <b>${fmt(v.velocity, 4)} m/s</b></span></div><p class="caption" data-velocity-source="${v.velocity_mixed_source?'mixed':'same'}">速度来源：${escapeHtml(v.velocity_source)}${frame>0?' · '+(frame-1)+'→'+frame+'帧':''}。有效显示位置都会按实际时间差计算；衔接也保留，不代表物理运动事实或接触证据。</p><div class="table-scroll"><table><thead><tr><th>量 / 单位</th><th>当前预测</th><th>参考真值</th><th>差值（预测 − 真值）</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
    }).join('');
    const p = data.frames[frame].pairs[version];
    const pred=p.prediction,ref=p.reference;
    const slack=p.source_metrics;
    $('distanceValues').innerHTML = `<table><thead><tr><th>两车关系</th><th>当前画面方案</th><th>按真值摆放的同一简化车身</th><th>差值</th></tr></thead><tbody>${row('三维最近表面欧氏间距','m',pred?.distance_m??null,ref?.distance_m??null)}<tr><th>简化车身几何接触</th><td>${contactText(pred)}</td><td>${contactText(ref)}</td><td>${typeof pred?.contact==='boolean'&&typeof ref?.contact==='boolean'?(pred.contact===ref.contact?'一致':'不一致'):'—'}</td></tr></tbody></table><details><summary>距离与接触的定义、旧投影指标</summary><p>三维最近表面欧氏间距始终非负；相交时为0，0本身不能区分刚接触与重叠。几何接触单独读取SOURCE凸包评分；不等于真实车壳间距或物理事故。缺测／未计算显示“—”，不能据二维投影认定三维接触。</p><p>当前方案三维共同半空间松弛：${fmt(slack?.source_convex_3d_slack_m)} m；参考：${fmt(slack?.reference_source_convex_3d_slack_m)} m。此有符号松弛不是欧氏间距，也不是穿透深度。</p><p>旧2D-SAT地面投影间隙：预测 ${fmt(p.projection_pair?.prediction?.distance_m)} m；参考 ${fmt(p.projection_pair?.reference?.distance_m)} m。仅在当前画面确实沿用对应基线时列出。</p></details>`;
    $('frameContactSummary').innerHTML=`<strong>${data.case} · 第 ${frame} 帧（帧号从0开始） · ${data.versions[version]}</strong><p>当前预测：<b>${contactText(pred)}</b> · 三维最近表面间距 ${fmt(pred?.distance_m)} m<br>真值摆放：<b>${contactText(ref)}</b> · 三维最近表面间距 ${fmt(ref?.distance_m)} m</p><p>${p.source||''}。${data.display_coverage||''}${!pred?'此帧该方案尚无有效三维距离评分；不以旧投影距离代替。':''}${data.case==='joint_fresh_07'?'物理碰撞记录从第41帧开始；第40帧不能提前作为真实碰撞。':data.event_label+'。'}</p>`;
    document.querySelectorAll('.frame-cell').forEach(button => {
      const active = Number(button.dataset.jump) === frame;
      button.classList.toggle('current', active);
      if (active) button.setAttribute('aria-current', 'true'); else button.removeAttribute('aria-current');
    });
    const reasons = index[frame].cars.filter(car => car.flagged).map(car => `车${car.car}：${car.reasons.map(k => metricInfo[k].label).join('、') || '求解未接受'}`);
    $('currentFlag').textContent = reasons.length ? `本帧已标记 · ${reasons.join('；')}` : '本帧未超过查看阈值';
    $('currentFlag').classList.toggle('is-flagged', reasons.length > 0);
  }
  function rebuild() {
    index = buildIndex(data, version, thresholds);
    const metric = $('rankMetric').value;
    const info = metricInfo[metric];
    const flagged = index.filter(row => row.cars.some(car => car.flagged));
    $('flagCount').textContent = `当前方案：${data.versions[version]} ｜ 下划线：${flagged.length} / ${data.frame_count} 帧（任一项超阈值或求解未接受／缺测） ｜ 红格：${info.label} > ${thresholds[metric]} ${info.unit}`;
    const comparison = compareMetricToCandidate(data, version, metric, thresholds[metric]);
    $('markComparison').innerHTML = comparison.map(row => `<p><strong class="${row.car === 0 ? 'blue' : 'orange'}">车${row.car}</strong> · 当前红格 <b>${row.currentHighFrames.length}</b>（${frameRanges(row.currentHighFrames)}） · 旧过滤 ${row.baselineHighFrames.length} · 消除：${frameRanges(row.clearedFrames)} · 新增：${frameRanges(row.addedFrames)} · 数值变化 ${row.changedFrames.length} 帧（${frameRanges(row.changedFrames)}）${row.incomparableFrames.length ? ` · 无可比数值 ${row.incomparableFrames.length} 帧（${frameRanges(row.incomparableFrames)}）` : ''}</p>`).join('') + `<p class="caption">以上均相对旧过滤，仅比较${escapeHtml(info.label)}。数值变化不一定跨过红色阈值，红格相同也可能已有变化；位置缺测／无前邻帧速度不按0计算，也不算“消除”。跨方案的有效位置仍计算差分，并标“方案衔接差分”。${escapeHtml(data.display_coverage || '全片使用所选原有基线。')} 新增／消除只表示相对旧过滤的查看阈值变化，不代表本轮新增、科学验收或碰撞判断。</p>`;
    $('errorTracks').innerHTML = [0, 1].map(car => `<div class="error-track"><strong class="${car === 0 ? 'blue' : 'orange'}">车${car}</strong><div class="frame-cells">${index.map(row => {
      const entry = row.cars[car], value = entry.values[metric];
      const level = value === null ? 'missing' : value > thresholds[metric] ? 'high' : 'low';
      const title = `车${car} · 帧${row.frame} · ${entry.values.source||data.versions[version]} · 位置 ${fmt(entry.values.position, 3)}cm · 速度 ${fmt(entry.values.velocity, 4)}m/s（${entry.values.velocity_source}） · 朝向 ${fmt(entry.values.rotation, 3)}°`;
      return `<button class="frame-cell ${level} ${entry.flagged ? 'marked' : ''}" data-jump="${row.frame}" title="${title}" aria-label="${title}${entry.flagged ? '，已标记' : ''}">${row.frame % 10 === 0 || row.frame === 79 ? row.frame : ''}</button>`;
    }).join('')}</div></div>`).join('');
    $('errorRanking').innerHTML = [0, 1].map(car => `<div class="ranking-car"><strong class="${car === 0 ? 'blue' : 'orange'}">车${car} · ${escapeHtml(data.versions[version])} · ${info.label}最大的5帧</strong><div>${rankFrames(index, car, metric).map(entry => `<button title="${escapeHtml(entry.values.source||data.versions[version])}${metric==='velocity'?' · '+escapeHtml(entry.values.velocity_source):''}${!entry.values.accepted?' · 求解未接受，另看综合标记':''}" data-jump="${entry.frame}" class="rank-jump ${entry.values[metric]!==null&&entry.values[metric]>thresholds[metric] ? 'marked' : ''}">f${entry.frame} · ${fmt(entry.values[metric], 3)} ${info.unit}${metric==='velocity'&&entry.values.velocity_mixed_source?' · 方案衔接差分':''}</button>`).join('')}</div></div>`).join('');
    const boundaries=index.flatMap(row=>row.cars.filter(car=>car.values.velocity_mixed_source).map(car=>({frame:row.frame,car:car.car,current:car.values.velocity,baseline:valuesAt(data,row.frame,car.car,'candidate').velocity})));
    $('velocityBoundaryComparison').innerHTML=`<details${boundaries.length?' open':''}><summary>方案衔接差分：${boundaries.length} 个帧车段 · 与灯过滤对照</summary><p>以下按当前显示位置直接差分，不能因跨方案而藏成缺测。比较基准始终是灯过滤；相对它新增的红格可能在旧组合中已存在，不自动算本轮新增。差分不等于物理真值，也不是接触证据。</p>${boundaries.length?'<table><thead><tr><th>车辆／相邻帧</th><th>当前速度误差 m/s</th><th>灯过滤 m/s</th><th>相对灯过滤</th></tr></thead><tbody>'+boundaries.map(r=>`<tr><td>车${r.car} · ${r.frame-1}→${r.frame}</td><td>${fmt(r.current,6)}</td><td>${fmt(r.baseline,6)}</td><td>${r.current>thresholds.velocity?(r.baseline>thresholds.velocity?'两者均超速度阈值':'新增速度红格'):r.baseline>thresholds.velocity?'消除速度红格':'两者未超速度阈值'}</td></tr>`).join('')+'</tbody></table>':'<p>当前方案无跨来源的有效位置衔接段。</p>'}</details>`;
    renderValues();
  }
  $('inspection').addEventListener('click', event => {
    const button = event.target.closest('[data-jump]');
    if (button) onJump(Number(button.dataset.jump));
  });
  for (const key of Object.keys(thresholds)) {
    const element = $(`${key}Threshold`);
    element.value = thresholds[key];
    element.addEventListener('change', () => {
      const value = element.valueAsNumber;
      if (!Number.isFinite(value) || value <= 0) { element.value = thresholds[key]; return; }
      thresholds[key] = value; rebuild();
    });
  }
  $('rankMetric').onchange = rebuild;
  const jumpMarked = direction => {
    const marked = index.filter(row => row.cars.some(car => car.flagged)).map(row => row.frame);
    const target = direction > 0 ? marked.find(value => value > frame) : marked.slice().reverse().find(value => value < frame);
    if (target !== undefined) onJump(target);
    else $('currentFlag').textContent = direction > 0 ? '后面没有其他标记帧' : '前面没有其他标记帧';
  };
  $('previousFlag').onclick = () => jumpMarked(-1);
  $('nextFlag').onclick = () => jumpMarked(1);
  rebuild();
  return (newFrame, newVersion, newData = data) => {
    const changed = data !== newData; data = newData;
    frame = newFrame;
    if (newVersion !== version || changed) { version = newVersion; rebuild(); } else renderValues();
  };
}
