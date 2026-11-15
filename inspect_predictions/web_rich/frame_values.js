// Pure calculations on recorded poses. No image access, inference or smoothing.
export const thresholdsDefault = {position: 5, velocity: 0.3, rotation: 2};
export const metricInfo = {
  position: {label: '中心位置误差', unit: 'cm'},
  velocity: {label: '速度向量误差', unit: 'm/s'},
  rotation: {label: '整体朝向误差', unit: '°'},
};
const length = values => Math.hypot(...values);
const difference = (a, b) => a.map((x, i) => x - b[i]);
export function valuesAt(data, frameIndex, carId, version) {
  const frame = data.frames[frameIndex], car = frame.cars[carId];
  const prediction = car.versions[version], reference = car.reference;
  if (!Array.isArray(prediction.pose) || !Array.isArray(prediction.quaternion_xyzw)) return {
    predictedPose:[null,null,null,null,null,null],referencePose:reference.pose,poseDelta:[null,null,null,null,null,null],
    predictedVelocity:null,referenceVelocity:null,velocityDelta:null,predictedSpeed:null,referenceSpeed:null,
    position:null,velocity:null,rotation:null,accepted:false,source:prediction.display_source,missing:true,
    velocity_source:'位置缺测，差分速度未知',velocity_mixed_source:false,
  };
  const poseDelta = difference(prediction.pose, reference.pose);
  let predictedVelocity = null, referenceVelocity = null, velocityDelta = null;
  let velocityMixedSource=false,velocitySource='第0帧无前邻帧，差分速度未知';
  if (frameIndex > 0) {
    const previous = data.frames[frameIndex - 1], dt = frame.time_s - previous.time_s;
    if (!(dt > 0)) throw new Error('相邻帧时间必须递增');
    const oldPrediction = previous.cars[carId].versions[version];
    velocityMixedSource=!!prediction.display_key&&prediction.display_key!==oldPrediction.display_key;
    if (Array.isArray(oldPrediction.pose)) predictedVelocity = difference(prediction.pose.slice(0, 3), oldPrediction.pose.slice(0, 3)).map(x => x / dt);
    velocitySource=predictedVelocity?(velocityMixedSource?'方案衔接差分':'相邻显示位置差分'):'前一帧位置缺测，差分速度未知';
    referenceVelocity = difference(reference.pose.slice(0, 3), previous.cars[carId].reference.pose.slice(0, 3)).map(x => x / dt);
    if (predictedVelocity) velocityDelta = difference(predictedVelocity, referenceVelocity);
  }
  const q = prediction.quaternion_xyzw, r = reference.quaternion_xyzw;
  const cosine = Math.min(1, Math.abs(q.reduce((sum, x, i) => sum + x * r[i], 0)) / (length(q) * length(r)));
  return {
    predictedPose: prediction.pose, referencePose: reference.pose, poseDelta,
    predictedVelocity, referenceVelocity, velocityDelta,
    predictedSpeed: predictedVelocity === null ? null : length(predictedVelocity),
    referenceSpeed: referenceVelocity === null ? null : length(referenceVelocity),
    position: length(poseDelta.slice(0, 3)) * 100,
    velocity: velocityDelta === null ? null : length(velocityDelta),
    rotation: 2 * Math.acos(cosine) * 180 / Math.PI,
    accepted: prediction.solver_acceptable,
    source:prediction.display_source,missing:false,
    velocity_source:velocitySource,velocity_mixed_source:velocityMixedSource&&predictedVelocity!==null,
  };
}
export function buildIndex(data, version, thresholds) {
  return data.frames.map((frame, index) => ({
    frame: index,
    cars: frame.cars.map(car => {
      const values = valuesAt(data, index, car.car, version);
      const reasons = Object.keys(metricInfo).filter(key => values[key] !== null && values[key] > thresholds[key]);
      return {car: car.car, values, reasons, flagged: reasons.length > 0 || !values.accepted};
    }),
  }));
}
export function rankFrames(index, car, metric, count = 5) {
  return index.map(row => ({frame: row.frame, ...row.cars[car]}))
    .filter(row => row.values[metric] !== null)
    .sort((a, b) => b.values[metric] - a.values[metric] || a.frame - b.frame).slice(0, count);
}

// Compare only the selected metric. Missing values remain unknown; a missing
// current value must never be counted as a cleared baseline threshold crossing.
export function compareMetricToCandidate(data, version, metric, threshold) {
  return Array.from({length:data.car_count}, (_, car) => {
    const result = {car, currentHighFrames:[], baselineHighFrames:[], clearedFrames:[], addedFrames:[], changedFrames:[], incomparableFrames:[]};
    for (let frame = 0; frame < data.frame_count; frame++) {
      const current = valuesAt(data, frame, car, version)[metric];
      const baseline = valuesAt(data, frame, car, 'candidate')[metric];
      if (Number.isFinite(current) && current > threshold) result.currentHighFrames.push(frame);
      if (Number.isFinite(baseline) && baseline > threshold) result.baselineHighFrames.push(frame);
      if (!Number.isFinite(current) || !Number.isFinite(baseline)) {
        result.incomparableFrames.push(frame);
        continue;
      }
      if (baseline > threshold && current <= threshold) result.clearedFrames.push(frame);
      if (current > threshold && baseline <= threshold) result.addedFrames.push(frame);
      // Ignore storage rounding far below the UI precision; this does not alter
      // values, thresholds, ranking, or the actual red cells.
      if (Math.abs(current - baseline) > 1e-6) result.changedFrames.push(frame);
    }
    return result;
  });
}

export function rotateVector(vector, quaternion) {
  const norm=Math.hypot(...quaternion);
  const [x,y,z,w]=quaternion.map(value=>value/norm),[a,b,c]=vector;
  const u=[y*c-z*b,z*a-x*c,x*b-y*a];
  const v=[y*u[2]-z*u[1],z*u[0]-x*u[2],x*u[1]-y*u[0]];
  return vector.map((value,i)=>value+2*(w*u[i]+v[i]));
}

export function headingDegrees(quaternion) {
  const forward=rotateVector([1,0,0],quaternion);
  return Math.atan2(forward[1],forward[0])*180/Math.PI;
}

export function convexHull2D(points) {
  const sorted=points.map(point=>point.slice(0,2)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const half=items=>{const result=[];for(const point of items){while(result.length>=2&&cross(result[result.length-2],result[result.length-1],point)<=0)result.pop();result.push(point);}result.pop();return result;};
  return [...half(sorted),...half(sorted.slice().reverse())];
}

export function projectedFootprint(vertices, pose, quaternion, referenceCenter) {
  // Orthographic projection onto LAB XY, after the full recorded 3D rotation.
  return convexHull2D(vertices.map(vertex=>rotateVector(vertex,quaternion).map((value,i)=>value+pose[i]-referenceCenter[i])));
}

export function bodyRelativeOffset(predictedCenter, referenceCenter, referenceQuaternion) {
  const heading=headingDegrees(referenceQuaternion)*Math.PI/180;
  const delta=predictedCenter.map((value,i)=>(value-referenceCenter[i])*100);
  return {forwardCm:delta[0]*Math.cos(heading)+delta[1]*Math.sin(heading),
          leftCm:-delta[0]*Math.sin(heading)+delta[1]*Math.cos(heading)};
}
