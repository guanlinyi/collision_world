"""Current full projected SOURCE union arc index; no smoothing or mesh welding.

This changes only the topology used by refresh competition.  It never filters
competitors by depth certification.  Original best-point certification remains
the caller's responsibility.  Unknown competitors must not be silently dropped.
"""
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.errors import GEOSException
from tangent_kernel import PIXEL_TOL


def _rings(geometry):
    if geometry.geom_type == 'Polygon':
        return [geometry.exterior, *geometry.interiors]
    if hasattr(geometry, 'geoms'):
        return [ring for part in geometry.geoms for ring in _rings(part)]
    return []


class ExactUnionIndex:
    def __init__(self, builder, pool, problem):
        self.builder, self.pool, self.problem = builder, pool, problem
        self.bindings = {}
        self.rings = []
        self.construction_unknown = None
        self.union_type = None
        self.triangle_count = len(builder.triangles)
        self.zero_area_count = 0
        self.unknown_reasons = {}
        module = builder.module
        pose = np.asarray(pool['pose'], dtype=float)
        camera = module.camera(problem.row)
        cp = (builder.triangles @ module.rot(pose).T + pose[:3] - camera[1]) @ np.linalg.inv(camera[0].T)
        if not np.isfinite(cp).all() or np.any(cp[:, :, 2] <= 0):
            self.construction_unknown = 'nonfinite_or_nonpositive_camera_depth'
            return
        uv = cp[:, :, :2] / cp[:, :, 2, None] * camera[2] + camera[3]
        polygons = shapely.polygons(uv)
        areas = shapely.area(polygons)
        positive = areas > 0
        self.zero_area_count = int((areas == 0).sum())
        if not np.isfinite(areas).all() or not np.all(shapely.is_valid(polygons[positive])):
            self.construction_unknown = 'invalid_positive_area_projected_triangle'
            return
        if not np.any(positive):
            self.construction_unknown = 'empty_positive_area_projection'
            return
        # Exact double-precision GEOS overlay: no snap, grid_size or buffer repair.
        try:
            union = shapely.union_all(polygons[positive])
        except GEOSException:
            self.construction_unknown = 'projected_union_topology_exception'
            return
        self.union_type = union.geom_type
        if union.is_empty or not union.is_valid:
            self.construction_unknown = 'empty_or_invalid_projected_union'
            return
        for ring_id, ring in enumerate(_rings(union)):
            line = LineString(ring.coords)
            xy = np.asarray(line.coords, dtype=float)
            vector = xy[1:] - xy[:-1]
            lengths = np.linalg.norm(vector, axis=1)
            nonzero = lengths > 0
            # Removing zero-length duplicate edges does not move any boundary point.
            starts = xy[:-1][nonzero]
            vectors = vector[nonzero]
            lengths = lengths[nonzero]
            self.rings.append(dict(
                ring_id=ring_id, closed_simple=bool(line.is_ring),
                starts=starts, vectors=vectors, lengths=lengths,
                cumulative=np.r_[0., np.cumsum(lengths)],
                original_segment_ids=np.flatnonzero(nonzero),
                length=float(lengths.sum()),
            ))
        if not self.rings:
            self.construction_unknown = 'no_polygon_boundary_ring'

    def _unknown(self, reason, **witness):
        self.unknown_reasons[reason] = self.unknown_reasons.get(reason, 0) + 1
        return dict(valid=False, ring_id=None, coordinate=None, length=None,
                    unknown_reason=reason, ring_distance=None, **witness)

    def locate(self, segment_id, s):
        """Bind the candidate's certified-later 3D foot to one 2D boundary arc.

        s is the image segment fraction, already clamped by the caller.  The
        original perspective conversion is essential; a 2D fraction is not a
        linear 3D fraction.  No SOURCE depth certification is performed here.
        """
        key = (int(segment_id), float(s))
        if key in self.bindings:
            return self.bindings[key]
        if self.construction_unknown:
            result = self._unknown(self.construction_unknown)
        elif not np.isfinite(s) or not 0 <= s <= 1:
            result = self._unknown('foot_fraction_outside_closed_segment')
        else:
            result = self._locate_foot(int(segment_id), float(s))
        self.bindings[key] = result
        return result

    def _locate_foot(self, segment_id, s):
        module = self.builder.module
        pair = self.pool['segment_ids'][segment_id]
        ends = np.asarray(self.pool['local'][pair], dtype=float)
        _, z = module.project(ends, self.pool['pose'], self.problem.row)
        z = np.asarray(z)
        if not np.isfinite(z).all() or np.any(z <= 0):
            return self._unknown('nonpositive_candidate_endpoint_depth')
        t = float(s * z[0] / ((1-s) * z[1] + s * z[0]))
        local = ends[0] + t * (ends[1] - ends[0])
        xy, depth = module.project(local[None], self.pool['pose'], self.problem.row)
        q = xy[0]
        if not np.isfinite(q).all() or not np.isfinite(depth).all() or np.any(np.asarray(depth) <= 0):
            return self._unknown('nonfinite_candidate_foot_projection')
        witness = dict(segment_id=segment_id, image_fraction=s,
                       perspective_local_fraction=t, q_xy=q.tolist(),
                       local_xyz=local.tolist(), camera_z=float(depth[0]))
        result = self.locate_xy(q)
        result.update(witness)
        return result

    def locate_xy(self, q):
        """Pure boundary binding, exposed for independent synthetic checks."""
        if self.construction_unknown:
            return self._unknown(self.construction_unknown)
        q = np.asarray(q, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            return self._unknown('nonfinite_or_invalid_xy')
        witness = dict(q_xy=q.tolist())
        matches = []
        closest_ring_distance = float('inf')
        for ring in self.rings:
            if not len(ring['lengths']):
                continue
            u = np.clip(np.sum((q-ring['starts']) * ring['vectors'], axis=1)
                        / ring['lengths']**2, 0, 1)
            feet = ring['starts'] + u[:, None] * ring['vectors']
            distance = np.linalg.norm(feet-q, axis=1)
            nearest = float(distance.min())
            closest_ring_distance = min(closest_ring_distance, nearest)
            near = np.flatnonzero(distance <= PIXEL_TOL)
            if len(near):
                matches.append((ring, u, distance, near))
        witness['closest_ring_distance'] = closest_ring_distance if np.isfinite(closest_ring_distance) else None
        witness['candidate_ring_ids'] = [r[0]['ring_id'] for r in matches]
        if not matches:
            return self._unknown('no_boundary_ring_within_original_PIXEL_TOL', **witness)
        if len(matches) != 1:
            return self._unknown('multiple_boundary_rings_at_foot', **witness)
        ring, u, distance, near = matches[0]
        if not ring['closed_simple']:
            return self._unknown('boundary_ring_not_closed_simple', **witness)
        flags = np.zeros(len(distance), dtype=bool)
        flags[near] = True
        # One cyclic contiguous cluster includes ordinary shared-vertex corners.
        # Separate clusters on the same ring indicate numerically ambiguous arc
        # positions; never let argmin silently select one near a self-contact.
        clusters = 1 if flags.all() else int(np.sum(flags & ~np.roll(flags, 1)))
        cluster_witness = dict(
            cluster_count=clusters,
            boundary_segment_ids=ring['original_segment_ids'][near].tolist(),
            cyclic_compact_segment_ids=near.tolist(),
            distances_px=distance[near].tolist(),
            fractions=u[near].tolist(),
            ring_segment_count=len(distance),
        )
        near_coordinates = (ring['cumulative'][near] + u[near]*ring['lengths'][near]) % ring['length']
        ordered = np.sort(near_coordinates)
        circular_gaps = np.diff(np.r_[ordered, ordered[0]+ring['length']])
        coordinate_span = max(0., float(ring['length'] - np.max(circular_gaps)))
        cluster_witness.update(
            candidate_arc_coordinates_px=near_coordinates.tolist(),
            minimum_covering_arc_span_px=coordinate_span,
            maximum_equivalent_arc_span_px=float(2*PIXEL_TOL),
        )
        witness['adjacent_cluster_witness'] = cluster_witness
        if flags.all():
            return self._unknown('entire_boundary_ring_within_pixel_tolerance', **witness)
        if clusters != 1:
            return self._unknown('multiple_nonadjacent_boundary_arc_clusters', **witness)
        # Adjacent sides of a very thin V can have almost identical image
        # positions but distant arc coordinates.  Adjacency alone is not a
        # unique binding.  Ordinary shared vertices/seams have span zero.
        if coordinate_span > 2*PIXEL_TOL:
            return self._unknown('adjacent_boundary_feet_have_distinct_arc_coordinates', **witness)
        index = int(near[np.argmin(distance[near])])
        coordinate = float((ring['cumulative'][index] + u[index]*ring['lengths'][index]) % ring['length'])
        return dict(valid=True, ring_id=ring['ring_id'], coordinate=coordinate,
                    length=ring['length'], unknown_reason=None,
                    ring_distance=float(distance[index]), **witness)

    def arc_distance(self, left, s, right, t):
        """Original 10px competition threshold remains entirely with the caller."""
        a, b = self.locate(left, s), self.locate(right, t)
        if not a['valid'] or not b['valid']:
            return float('nan')
        if a['ring_id'] != b['ring_id']:
            return float('inf')
        delta = abs(a['coordinate']-b['coordinate'])
        return float(min(delta, a['length']-delta))

    def summary(self):
        return dict(algorithm='full_projected_positive_area_triangle_union_closed_boundary_arcs',
                    shapely_version=shapely.__version__, pixel_tolerance=float(PIXEL_TOL),
                    precision_grid=None, snap_or_buffer=False,
                    triangle_count=self.triangle_count, zero_area_triangle_count=self.zero_area_count,
                    union_type=self.union_type, ring_count=len(self.rings),
                    closed_simple_ring_count=sum(r['closed_simple'] for r in self.rings),
                    construction_unknown=self.construction_unknown,
                    cached_foot_bindings=len(self.bindings),
                    unknown_reasons=dict(self.unknown_reasons),
                    depth_certificate_filter_applied=False)


def get_index(builder, pool, problem):
    """Cache only within this pose-specific pool; do not share across poses."""
    key = '_exact_projected_union_arc_index'
    if key not in pool:
        pool[key] = ExactUnionIndex(builder, pool, problem)
    return pool[key]
