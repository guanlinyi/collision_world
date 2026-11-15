import numpy as np
from pathlib import Path
from collections import Counter,defaultdict
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from portable_io import ROOT,read,save_json,save_npz
SOFT=ROOT
WHOLE=None
from audit_silhouette import ProjectedUnion,distance_to_segments
def query_depth(projected, point, restrict=None, closed=False):
    ids = projected.bins.get(tuple(np.floor(point / projected.bin_px).astype(int)), np.empty(0, int))
    if restrict is not None:
        ids = np.intersect1d(ids, restrict)
    bary = projected.barycentric(ids, point)
    if closed:
        opposite = projected.edge_lengths[ids][:, [1, 2, 0]]
        altitude = abs(projected.den[ids])[:, None] / np.maximum(opposite, 1e-300)
        keep = np.all(bary * altitude >= -1e-07, axis=1)
    else:
        keep = np.all(bary >= 0, axis=1)
    ids, bary = (ids[keep], bary[keep])
    if not len(ids):
        return (None, None, [], [])
    z = 1 / np.sum(bary / projected.z[ids], axis=1)
    if not np.isfinite(z).all() or np.any(z <= 0):
        return (None, None, ids.tolist(), bary.tolist())
    nearest = int(np.argmin(z))
    return (float(z[nearest]), int(ids[nearest]), ids.tolist(), bary.tolist())

def tolerance(a, b):
    return float(32 * np.finfo(np.float64).eps * max(1, abs(a), abs(b)))

def coverage(image_points, edge_ids, accepted, raster):
    segments = np.array([image_points[[i, i + 1]] for i in range(len(image_points) - 1) if accepted[i] and accepted[i + 1] and (edge_ids[i] == edge_ids[i + 1])]).reshape(-1, 2, 2)
    distance = distance_to_segments(raster.mean(1), segments)
    reverse = distance_to_segments(image_points[accepted], raster)
    missing = np.where(distance > 1.5)[0]
    adjacency = defaultdict(list)
    for i in missing:
        for endpoint in raster[i]:
            adjacency[tuple(np.rint(endpoint * 2).astype(int))].append(int(i))
    unseen, components = (set(missing.tolist()), [])
    while unseen:
        stack, component = ([unseen.pop()], [])
        while stack:
            index = stack.pop()
            component.append(index)
            for endpoint in raster[index]:
                for other in adjacency[tuple(np.rint(endpoint * 2).astype(int))]:
                    if other in unseen:
                        unseen.remove(other)
                        stack.append(other)
        components.append(component)
    report = dict(fraction_within_1_5px=float(np.mean(distance <= 1.5)), max_uncovered_connected_boundary_length_px=max(map(len, components), default=0), uncovered_component_lengths_px=sorted([len(c) for c in components], reverse=True), reverse_fraction_within_1_5px=float(np.mean(reverse <= 1.5)), certified_contiguous_segments=len(segments))
    return (report, segments, distance, reverse, components)
