import numpy as np
import cv2

def triangulate_landmarks(landmarks_per_cam, proj_mats, landmark_ids):
    triangulated = {}

    for landmark_id in landmark_ids:
        points_2d = []
        proj_used = []

        for cam_idx, landmarks in landmarks_per_cam.items():
            if landmark_id < len(landmarks):
                lm = landmarks[landmark_id]
                if lm.visibility > 0.5:  # Optional: Only triangulate if confident
                    x, y = lm.x, lm.y
                    points_2d.append([x, y])
                    proj_used.append(proj_mats[cam_idx])

        if len(points_2d) >= 2:
            point_3d = dlt_triangulation(points_2d, proj_used)
            triangulated[landmark_id] = point_3d

    return triangulated

def dlt_triangulation(pts, proj_mats):
    A = []
    for pt, P in zip(pts, proj_mats):
        x, y = pt
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]) * 100  # Convert from meters to centimeters if needed
