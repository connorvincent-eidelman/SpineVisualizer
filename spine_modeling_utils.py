# === File: spine_modeling_utils.py ===
import numpy as np
from scipy.interpolate import splprep, splev
from collections import defaultdict

class LandmarkSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.history = {}

    def smooth(self, lid, point):
        if lid not in self.history:
            self.history[lid] = point
        else:
            self.history[lid] = self.alpha * np.array(point) + (1 - self.alpha) * np.array(self.history[lid])
        return self.history[lid]

def get_spine_points(triangulated):
    spine_ids = [0, 11, 12, 23, 24]  # NOSE, L/Shoulder, R/Shoulder, L/Hip, R/Hip
    points = []
    for lid in spine_ids:
        if lid in triangulated:
            points.append(triangulated[lid])
    return points if len(points) >= 2 else None

def fit_spine_curve(points):
    points = np.array(points).T
    m = points.shape[1]
    if m < 2:
        return None
    k = min(3, m - 1)  # Ensure m > k
    try:
        tck, _ = splprep(points, s=0, k=k)
        u_fine = np.linspace(0, 1, 50)
        x_fine, y_fine, z_fine = splev(u_fine, tck)
        curve_points = np.vstack((x_fine, y_fine, z_fine)).T
        return curve_points
    except Exception as e:
        print(f"[Spline Fit Error]: {e}")
        return None

def compute_lateral_deviation(curve_points):
    if len(curve_points) < 2:
        return 0.0
    midline = np.array([
        curve_points[0],
        curve_points[-1]
    ])
    vec = midline[1] - midline[0]
    vec = vec / np.linalg.norm(vec)
    total_dev = 0
    for pt in curve_points[1:-1]:
        proj_len = np.dot(pt - midline[0], vec)
        proj_point = midline[0] + proj_len * vec
        total_dev += np.linalg.norm(pt - proj_point)
    return total_dev / (len(curve_points) - 2)

def project_curve_to_image(curve_3d, proj_mat):
    pts_2d = []
    for pt3d in curve_3d:
        pt = np.append(pt3d / 100.0, 1)  # cm to m
        proj = proj_mat @ pt
        proj /= proj[2]
        pts_2d.append((int(proj[0]), int(proj[1])))
    return pts_2d
