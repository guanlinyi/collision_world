/** Recorded calibration only. LAB metres -> Three (x,z,-y); no inferred camera. */
import * as THREE from './input_js/three.module.js';

export function validateRecordedCameras(document, data) {
  if (document.schema !== 'cwm_recorded_camera_v1' || document.case !== data.case ||
      document.records?.length !== data.frame_count * data.car_count) {
    throw new Error('需要当前案例完整的160条原始相机参数');
  }
  const records = new Map();
  for (const row of document.records) {
    if (!Number.isInteger(row.frame) || row.frame < 0 || row.frame >= data.frame_count ||
        !Number.isInteger(row.car) || row.car < 0 || row.car >= data.car_count) {
      throw new Error('相机帧号或车辆身份不符');
    }
    const matrix = row.camera_matrix;
    const intrinsics = row.intrinsics;
    if (matrix?.length !== 4 || matrix.some(line => line.length !== 4 || line.some(x => !Number.isFinite(x))) ||
        !intrinsics || !['fx', 'fy', 'cx', 'cy'].every(key => Number.isFinite(intrinsics[key])) ||
        intrinsics.fx <= 0 || intrinsics.fy <= 0) {
      throw new Error('相机矩阵或内参不完整');
    }
    if (row.image_width !== data.image_dimensions[0] || row.image_height !== data.image_dimensions[1] ||
        !Number.isFinite(row.time_s) || Math.abs(row.time_s - data.frames[row.frame].time_s) > 1e-5) {
      throw new Error('相机分辨率或时间与当前案例不符');
    }
    for (let first = 0; first < 3; first++) {
      for (let last = 0; last < 3; last++) {
        const dot = matrix[first].slice(0, 3).reduce((sum, x, column) => sum + x * matrix[last][column], 0);
        if (Math.abs(dot - Number(first === last)) > 1e-4) throw new Error('相机旋转矩阵不满足正交条件');
      }
    }
    const key = `${row.frame}:${row.car}`;
    if (records.has(key)) throw new Error('相机记录重复');
    records.set(key, row);
  }
  return records;
}

export function applyRecordedCamera(camera, renderer, controls, row, origin, width, height) {
  if (!row) throw new Error('当前帧／车辆没有原始相机参数');
  const matrix = row.camera_matrix;
  const k = row.intrinsics;
  const imageWidth = row.image_width;
  const imageHeight = row.image_height;
  // Same frozen S*C*P calibration as LAB. Three looks along -Z with +Y up.
  const labAxes = matrix.slice(0, 3).map((line, axis) =>
    [line[1], -line[2], line[0]].map(value => value * (axis === 1 ? -1 : 1)));
  const orientation = new THREE.Matrix4().set(
    labAxes[0][0], -labAxes[0][1], -labAxes[0][2], 0,
    labAxes[1][0], -labAxes[1][1], -labAxes[1][2], 0,
    labAxes[2][0], -labAxes[2][1], -labAxes[2][2], 0,
    0, 0, 0, 1);
  camera.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeRotationX(-Math.PI / 2).multiply(orientation));
  const labPosition = [matrix[0][3], -matrix[1][3], matrix[2][3]];
  camera.position.set(labPosition[0] - origin[0], labPosition[2] - origin[2], -(labPosition[1] - origin[1]));
  const viewport = applyRecordedViewport(camera, renderer, row, width, height);
  controls.enabled = false;
  return viewport;
}

export function restoreFreeViewport(camera, renderer, controls, width, height) {
  renderer.setViewport(0, 0, width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  controls.enabled = true;
}

/** Resize the recorded projection without moving a freely orbiting camera. */
export function applyRecordedViewport(camera, renderer, row, width, height) {
  const k = row.intrinsics;
  const imageWidth = row.image_width;
  const imageHeight = row.image_height;
  const scale = Math.min(width / imageWidth, height / imageHeight);
  const viewport = {x: (width - imageWidth * scale) / 2, y: (height - imageHeight * scale) / 2,
                    w: imageWidth * scale, h: imageHeight * scale};
  renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h);
  const near = camera.near;
  const far = camera.far;
  // Integer image samples are pixel centres; +.5 maps to the viewport edge convention.
  camera.projectionMatrix.set(
    2 * k.fx / imageWidth, 0, 1 - 2 * (k.cx + .5) / imageWidth, 0,
    0, 2 * k.fy / imageHeight, 2 * (k.cy + .5) / imageHeight - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.updateMatrixWorld(true);
  return viewport;
}
