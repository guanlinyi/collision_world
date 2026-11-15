import hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt
from audit_silhouette import ProjectedUnion,pixel_square_boundary
from continuous_depth_attempt import coverage
from tangent_kernel import certify
from portable_io import ROOT,read,save_json,save_npz
from portable_geometry import legacy,load_problems,FrozenProblem
AUDIT=ROOT/'geometry_source'
ATTEMPT=''
class GeometryPool:

    def __init__(self, out_root=None, attempt=None):
        self.out_root = ROOT if out_root is None else Path(out_root)
        self.attempt = ATTEMPT if attempt is None else attempt
        self.module = legacy()
        with np.load(AUDIT / 'input_npz/source_mesh.npz') as a:
            self.triangles = a['triangles_local_m']
        with np.load(AUDIT / 'input_npz/edge_samples_frame27.npz') as a:
            self.vertices = a['source_exact_vertices_local_m']
            self.edges = a['source_edges_exact_vertex_ids']
            self.faces = a['original_triangle_exact_vertex_ids']
        self.endpoints = self.vertices[self.edges]
        self.length = float(np.ptp(self.triangles[:, :, 0]))
        self.midpoint = float((self.triangles[:, :, 0].max() + self.triangles[:, :, 0].min()) / 2)
        self.kernel_hash = hashlib.sha256((Path(__file__).with_name('tangent_kernel.py')).read_bytes()).hexdigest()
        self.cache = {}

    def build(self, pose, row, label):
        key = tuple(np.asarray(pose).tolist()) + tuple(np.asarray(row['camera_matrix']).ravel().tolist())
        if key in self.cache:
            return self.cache[key]
        archive_path = self.out_root / 'output_npz' / self.attempt / f'{label}_source_pool.npz'
        certificate_path = self.out_root / 'output_json' / self.attempt / f'{label}_source_certificate.json'
        if archive_path.exists() and certificate_path.exists():
            report = read(certificate_path)
            assert np.array_equal(np.array(report['pose']), pose)
            assert report['tangent_kernel_sha256'] == self.kernel_hash
            with np.load(archive_path) as saved:
                arrays = {name: saved[name] for name in saved.files}
            keys = [('v', int(self.edges[e, 0])) if t == 0 else ('v', int(self.edges[e, 1])) if t == 1 else ('s', i) for i, (e, t) in enumerate(zip(arrays['edge_id'], arrays['edge_t']))]
            data = dict(pose=arrays['pose'], local=arrays['local_xyz_m'], xy=arrays['projected_xy'], edge_ids=arrays['edge_id'], t=arrays['edge_t'], normal=arrays['outward_normal_xy'], accepted=arrays['accepted'], segment_ids=arrays['segment_sample_ids'], arc_ids=arrays['arc_id'], status=arrays['status'], status_labels=report['status_labels'], admitted=report['admitted'], coverage=report['coverage'], unique_point_keys=keys)
            self.cache[key] = data
            return data
        m = self.module
        cam = m.camera(row)
        cp = (self.triangles @ m.rot(pose).T + pose[:3] - cam[1]) @ np.linalg.inv(cam[0].T)
        if np.any(cp[:, :, 2] <= 0):
            raise ValueError('Unsupported SOURCE near clipping')
        uv = cp[:, :, :2] / cp[:, :, 2, None] * cam[2] + cam[3]
        shape = (row['image_height'], row['image_width'])
        if uv[:, :, 0].min() < 0 or uv[:, :, 1].min() < 0 or uv[:, :, 0].max() >= shape[1] or (uv[:, :, 1].max() >= shape[0]):
            raise ValueError('SOURCE image clipping')
        projected = ProjectedUnion(uv, cp[:, :, 2], shape)
        depth, _ = projected.raster()
        foreground = np.isfinite(depth)
        raster = pixel_square_boundary(foreground)
        distance = distance_transform_edt(~(foreground & ~binary_erosion(foreground)))
        image, z = m.project(self.endpoints.reshape(-1, 3), pose, row)
        image, z = (image.reshape(-1, 2, 2), z.reshape(-1, 2))
        local_list, xy_list, edge_list, t_list, normal_list = ([], [], [], [], [])
        for eid, (ends, pixels, zs) in enumerate(zip(self.endpoints, image, z)):
            vector = pixels[1] - pixels[0]
            length = float(np.linalg.norm(vector))
            if length <= 1e-12:
                continue
            s = np.linspace(0, 1, max(2, int(np.ceil(length)) + 1))
            t = s * zs[0] / ((1 - s) * zs[1] + s * zs[0])
            local = ends[0] + t[:, None] * (ends[1] - ends[0])
            xy, _ = m.project(local, pose, row)
            local_list.extend(local)
            xy_list.extend(xy)
            edge_list.extend([eid] * len(t))
            t_list.extend(t)
            normal_list.extend([np.array([-vector[1], vector[0]]) / length] * len(t))
        local, xy, edge_ids, t, normals = (np.array(local_list), np.array(xy_list), np.array(edge_list), np.array(t_list), np.array(normal_list))
        rounding = np.rint(xy).astype(int)
        coarse = distance[rounding[:, 1], rounding[:, 0]] <= 1.5
        accepted = np.zeros(len(local), bool)
        outward = np.full_like(xy, np.nan)
        status = np.full(len(local), 1, np.int16)
        evidence = []
        counts = Counter()
        labels = {1: 'coarse_gate', 2: 'finite_offset_not_proposal'}
        for index in np.where(coarse)[0]:
            side = []
            for epsilon in [0.05, 0.025]:
                side.append([projected.test(xy[index] + sign * epsilon * normals[index]) for sign in [1, -1]])
            if any((ambiguous for scale in side for inside, ambiguous in scale)) or any((sum((v[0] for v in scale)) != 1 for scale in side)) or side[0][0][0] != side[1][0][0]:
                status[index] = 2
                continue
            camera_z = float(((local[index] @ m.rot(pose).T + pose[:3] - cam[1]) @ np.linalg.inv(cam[0].T))[2])
            result, proof = certify(projected, xy[index], camera_z, self.edges[edge_ids[index]], t[index], self.faces, exact_incident_depth=True)
            if result not in labels.values():
                labels[max(labels) + 1] = result
            code = next((k for k, v in labels.items() if v == result))
            status[index] = code
            accepted[index] = result == 'accepted'
            counts[result] += 1
            if accepted[index]:
                outward[index] = -normals[index] if side[1][0][0] else normals[index]
            proof.update(sample_id=int(index), edge_id=int(edge_ids[index]), status=result)
            evidence.append(proof)
        cov, segments, forward, reverse, runs = coverage(xy, edge_ids, accepted, raster)
        admitted = cov['fraction_within_1_5px'] >= 0.99 and cov['max_uncovered_connected_boundary_length_px'] <= 3 and (cov['reverse_fraction_within_1_5px'] == 1)
        segment_ids = np.array([[i, i + 1] for i in range(len(local) - 1) if accepted[i] and accepted[i + 1] and (edge_ids[i] == edge_ids[i + 1])], int)
        keys = []
        for i in range(len(local)):
            keys.append(('v', int(self.edges[edge_ids[i], 0])) if t[i] == 0 else ('v', int(self.edges[edge_ids[i], 1])) if t[i] == 1 else ('s', i))
        parent = {}
        sizes = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b in segment_ids:
            ra, rb = (find(keys[a]), find(keys[b]))
            if ra != rb:
                parent[rb] = ra
        arc_map = {}
        sample_arc = np.full(len(local), -1, int)
        for i in np.unique(segment_ids):
            root = find(keys[i])
            arc_map.setdefault(root, len(arc_map))
            sample_arc[i] = arc_map[root]
        data = dict(pose=np.array(pose), local=local, xy=xy, edge_ids=edge_ids, t=t, normal=outward, accepted=accepted, segment_ids=segment_ids, arc_ids=sample_arc, status=status, status_labels=labels, admitted=admitted, coverage=cov, unique_point_keys=keys, projected=projected)
        save_npz(archive_path, dict(pose=pose, local_xyz_m=local, projected_xy=xy, edge_id=edge_ids, edge_t=t, outward_normal_xy=outward, accepted=accepted, status=status, segment_sample_ids=segment_ids, arc_id=sample_arc, raster_coverage_distance_px=forward, reverse_raster_distance_px=reverse))
        save_json(certificate_path, dict(query_truth_used=False, pose=np.asarray(pose).tolist(), status_labels=labels, certificate_counts=dict(counts), accepted_samples=int(accepted.sum()), coverage=cov, admitted=bool(admitted), local_boundary_and_depth_evidence=evidence, geometric_source=str(AUDIT / 'input_npz/source_mesh.npz'), tangent_kernel=str(Path(__file__).with_name('tangent_kernel.py')), tangent_kernel_sha256=self.kernel_hash, structural_incident_depth=True))
        print('SOURCE_POOL', label, int(accepted.sum()), 'admitted', admitted, 'coverage', cov['fraction_within_1_5px'], flush=True)
        self.cache[key] = data
        return data
