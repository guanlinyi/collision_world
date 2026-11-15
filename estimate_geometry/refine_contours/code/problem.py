import numpy as np
from portable_geometry import geometry_from_meta
class GeometryOnlyProblem:
    """Only the fields required by the existing exterior solver and admission."""

    def __init__(self, safe, filtered, support, module):
        self.geo = geometry_from_meta(support['ground_support'], module)
        self.start = np.array(safe['old_predicted_prior'], float)[[0, 1, 5]]
        self.bound = np.array([0.8, 0.8, 12.0])
        baseline = np.array(filtered['pose'], float)
        self.row = module.pose_row(safe['calibration'], baseline, np.array(support['source_asset_origin_lab_m']))
        self.meta = dict(key=safe['key'], case=safe['case'], frame=safe['frame'], car=safe['car'], prior=safe['old_predicted_prior'], baseline_pose=baseline.tolist(), baseline_cost=filtered['cost'])
