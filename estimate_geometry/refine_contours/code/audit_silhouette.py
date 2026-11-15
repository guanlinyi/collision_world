import numpy as np
from pathlib import Path
from collections import Counter,defaultdict
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from portable_io import ROOT,read,save_json,save_npz
SOFT=ROOT
WHOLE=None
class ProjectedUnion:

    def __init__(self, triangles, depth, shape, bin_px=16):
        self.triangles, self.z, self.shape = (triangles, depth, shape)
        self.bin_px = bin_px
        self.a, self.b, self.c = (triangles[:, 0], triangles[:, 1], triangles[:, 2])
        self.den = (self.b[:, 1] - self.c[:, 1]) * (self.a[:, 0] - self.c[:, 0]) + (self.c[:, 0] - self.b[:, 0]) * (self.a[:, 1] - self.c[:, 1])
        self.valid = abs(self.den) > 1e-10
        self.lo, self.hi = (triangles.min(1), triangles.max(1))
        self.edge_lengths = np.linalg.norm(np.roll(triangles, -1, axis=1) - triangles, axis=2)
        self.bins = defaultdict(list)
        for index in np.where(self.valid)[0]:
            lo, hi = (np.floor(self.lo[index] / bin_px).astype(int), np.floor(self.hi[index] / bin_px).astype(int))
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    self.bins[x, y].append(index)
        self.bins = {key: np.array(value, int) for key, value in self.bins.items()}

    def barycentric(self, ids, points):
        points = np.asarray(points)
        a, b, c, den = (self.a[ids], self.b[ids], self.c[ids], self.den[ids])
        u = ((b[..., 1] - c[..., 1]) * (points[..., 0] - c[..., 0]) + (c[..., 0] - b[..., 0]) * (points[..., 1] - c[..., 1])) / den
        v = ((c[..., 1] - a[..., 1]) * (points[..., 0] - c[..., 0]) + (a[..., 0] - c[..., 0]) * (points[..., 1] - c[..., 1])) / den
        return np.stack([u, v, 1 - u - v], axis=-1)

    def test(self, point):
        ids = self.bins.get(tuple(np.floor(point / self.bin_px).astype(int)), np.empty(0, int))
        ids = ids[np.all(point >= self.lo[ids] - 1e-07, axis=1) & np.all(point <= self.hi[ids] + 1e-07, axis=1)]
        if not len(ids):
            return (False, False)
        bary = self.barycentric(ids, point)
        inside = np.all(bary >= 0, axis=1)
        opposite = self.edge_lengths[ids][:, [1, 2, 0]]
        altitude = abs(self.den[ids])[:, None] / np.maximum(opposite, 1e-300)
        signed_px = bary * altitude
        near = np.all(signed_px >= -1e-07, axis=1) & np.any(abs(signed_px) <= 1e-07, axis=1)
        interior = np.any(np.all(signed_px > 1e-07, axis=1))
        return (bool(np.any(inside)), bool(np.any(near) and (not interior)))

    def raster(self):
        h, w = self.shape
        depth = np.full((h, w), np.inf)
        triangle_ids = np.full((h, w), -1, np.int32)
        for index in np.where(self.valid)[0]:
            lo = np.maximum(np.ceil(self.lo[index]).astype(int), 0)
            hi = np.minimum(np.floor(self.hi[index]).astype(int), [w - 1, h - 1])
            if np.any(lo > hi):
                continue
            yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            points = np.stack([xx, yy], -1)
            bary = self.barycentric(index, points)
            inside = np.all(bary >= -1e-09, axis=-1)
            zz = 1 / np.maximum(np.sum(bary / self.z[index], axis=-1), 1e-12)
            target = depth[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            ids = triangle_ids[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            keep = inside & (zz < target)
            target[keep] = zz[keep]
            ids[keep] = index
        return (depth, triangle_ids)

    def face_pixel_visibility(self, point, faces, depth):
        center = np.rint(point).astype(int)
        pixels = np.array([(center[0] + x, center[1] + y) for y in range(-2, 3) for x in range(-2, 3)])
        pixels = pixels[np.argsort(np.linalg.norm(pixels - point, axis=1), kind='stable')]
        pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < self.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.shape[0])]
        nearest_delta = np.inf
        for face in faces:
            if not self.valid[face]:
                continue
            bary = self.barycentric(face, pixels)
            inside = np.all(bary >= -1e-09, axis=1)
            for pixel, weights in zip(pixels[inside], bary[inside]):
                predicted = 1 / np.maximum(np.sum(weights / self.z[face]), 1e-12)
                buffered = depth[pixel[1], pixel[0]]
                tolerance = 32 * np.finfo(depth.dtype).eps * max(1, abs(buffered))
                delta = abs(predicted - buffered)
                nearest_delta = min(nearest_delta, delta)
                if np.isfinite(buffered) and delta <= tolerance:
                    return (int(face), pixel, float(predicted), float(buffered), float(tolerance))
        return (-1, np.array([-1, -1]), float(nearest_delta), np.nan, np.nan)

def pixel_square_boundary(mask):
    padded = np.pad(mask, 1)
    sides = [mask & ~padded[:-2, 1:-1], mask & ~padded[2:, 1:-1], mask & ~padded[1:-1, :-2], mask & ~padded[1:-1, 2:]]
    offsets = [([-0.5, -0.5], [0.5, -0.5]), ([-0.5, 0.5], [0.5, 0.5]), ([-0.5, -0.5], [-0.5, 0.5]), ([0.5, -0.5], [0.5, 0.5])]
    segments = []
    for side, (a, b) in zip(sides, offsets):
        points = np.argwhere(side)[:, ::-1]
        segments.extend(np.stack([points + a, points + b], 1))
    return np.array(segments)

def distance_to_segments(points, segments, radius=1.5):
    if not len(segments):
        return np.full(len(points), np.inf)
    midpoint = segments.mean(1)
    half_length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) / 2
    tree = cKDTree(midpoint)
    result = np.full(len(points), np.inf)
    for index, (point, ids) in enumerate(zip(points, tree.query_ball_point(points, radius + half_length.max()))):
        if not ids:
            continue
        endpoints = segments[ids]
        vector = endpoints[:, 1] - endpoints[:, 0]
        t = np.clip(np.sum((point - endpoints[:, 0]) * vector, axis=1) / np.maximum(np.sum(vector * vector, axis=1), 1e-300), 0, 1)
        result[index] = np.min(np.linalg.norm(point - (endpoints[:, 0] + t[:, None] * vector), axis=1))
    return result
