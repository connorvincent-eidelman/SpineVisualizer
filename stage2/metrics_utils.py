import numpy as np

def compute_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def compute_angle(ptA, ptB, ptC):
    a = np.array(ptA) - np.array(ptB)
    c = np.array(ptC) - np.array(ptB)
    cos_angle = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle_rad)
