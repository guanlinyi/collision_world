import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
S=np.diag([1.,-1.,1.]);P=np.array([[0.,0.,1.],[1.,0.,0.],[0.,-1.,0.]])
def base_camera(row):
    C = np.asarray(row['camera_matrix'], float)
    return (S @ C[:3, :3] @ P, S @ C[:3, 3])

def camera(row):
    q,t=base_camera(row);k=row['intrinsics'];return q,t,np.array([k['fx'],k['fy']]),np.array([k['cx'],k['cy']])
a=SimpleNamespace(camera=base_camera)
def rot(p):
    return Rotation.from_rotvec(np.asarray(p)[3:], degrees=True).as_matrix()

class Geometry:

    def full(self, x):
        return np.array([x[0], x[1], (x[:2] - self.origin) @ self.plane[:2] + self.plane[2] + self.height, 0.0, 0.0, x[2]])

    def project(self, loc, pose, row):
        q, t = a.camera(row)
        cam = (np.asarray(loc) @ rot(pose).T + pose[:3] - t) @ np.linalg.inv(q.T)
        return cam[:, :2] / cam[:, 2, None] * 320 + [319.5, 179.5]
def pose_row(row, pose, origin):
    S = np.diag([1.0, -1.0, 1.0])
    R = rot(pose)
    T = np.eye(4)
    T[:3, :3] = S @ R @ S
    T[:3, 3] = S @ (np.array(pose[:3]) - R @ origin)
    return {**{k: row[k] for k in ['image_height', 'image_width', 'intrinsics', 'camera_matrix']}, 'actor_matrix': T.tolist()}
def project(xyz, pose, row):
    q, t, f, c = camera(row)
    cp = (xyz @ rot(pose).T + pose[:3] - t) @ np.linalg.inv(q.T)
    return (cp[:, :2] / cp[:, 2, None] * f + c, cp[:, 2])
def geometry_from_meta(meta, module):
    geo = module.Geometry.__new__(module.Geometry)
    geo.height = meta['height_m']
    geo.plane = np.array(meta['plane_z_axbyc'], dtype=np.float64)
    geo.origin = np.array(meta['origin_xy'], dtype=np.float64)
    return geo

def legacy():
    return SimpleNamespace(camera=camera,rot=rot,project=project,Geometry=Geometry,pose_row=pose_row)
def load_problems():raise RuntimeError('Use the explicit portable observation bundle')
class FrozenProblem:pass
