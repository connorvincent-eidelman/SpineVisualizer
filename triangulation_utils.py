import numpy as np
import cv2

def triangulate_point(pt1, pt2, P1, P2):
    A = np.array([
        pt1[0]*P1[2,:] - P1[0,:],
        pt1[1]*P1[2,:] - P1[1,:],
        pt2[0]*P2[2,:] - P2[0,:],
        pt2[1]*P2[2,:] - P2[1,:],
    ])
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]

def triangulate_landmarks(landmarks_per_cam, proj_matrices, landmark_indices):
    triangulated = {}
    for idx in landmark_indices:
        points = []
        projections = []
        for cam_id, lm_list in landmarks_per_cam.items():
            if lm_list[idx].visibility > 0.5:
                points.append([lm_list[idx].x, lm_list[idx].y])
                projections.append(proj_matrices[cam_id])
        if len(points) >= 2:
            pts3d = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    pt3d = triangulate_point(points[i], points[j], projections[i], projections[j])
                    pts3d.append(pt3d)
            triangulated[idx] = np.mean(pts3d, axis=0)
    return triangulated
