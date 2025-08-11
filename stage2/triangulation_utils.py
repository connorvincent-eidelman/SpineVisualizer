import numpy as np
import cv2

def triangulate_landmarks(landmarks_per_cam, proj_mats, landmark_ids, frame_shapes):
    triangulated = {}
    confidences = {}

    for landmark_id in landmark_ids:
        points_2d = []
        proj_used = []
        cam_idxs = []

        for cam_idx, landmarks in landmarks_per_cam.items():
            if landmark_id < len(landmarks):
                lm = landmarks[landmark_id]
                if lm.visibility > 0.5 and 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                    width, height = frame_shapes[cam_idx]
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    points_2d.append([x_px, y_px])
                    proj_used.append(proj_mats[cam_idx])
                    cam_idxs.append(cam_idx)

        if len(points_2d) >= 2:
            pt3d = dlt_triangulation(points_2d, proj_used)

            # Reproject and compute error
            errors = []
            for cam_idx, pt2d, P in zip(cam_idxs, points_2d, proj_used):
                pt_h = np.append(pt3d / 100.0, 1)  # convert to meters
                proj = P @ pt_h
                proj = proj[:2] / proj[2]
                error = np.linalg.norm(proj - np.array(pt2d))
                errors.append(error)

            mean_error = np.mean(errors)
            confidence = np.exp(-mean_error / 20.0) * (len(points_2d) / len(proj_mats))  # decay + coverage

            triangulated[landmark_id] = pt3d
            confidences[landmark_id] = confidence

    return triangulated, confidences



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

def geometric_confidence_score(pt3d, proj_mats, orig_2d_pts):
    """
    Compute average back-projection error to assess geometric confidence.

    Args:
        pt3d: 3D point as a NumPy array (x, y, z)
        proj_mats: list of 3x4 projection matrices used in triangulation
        orig_2d_pts: list of corresponding 2D points from each camera

    Returns:
        float: Mean reprojection error in pixels
    """
    pt3d_hom = np.append(pt3d / 100.0, 1.0)  # to meters and homogeneous
    errors = []

    for P, pt2d in zip(proj_mats, orig_2d_pts):
        proj = P @ pt3d_hom
        proj /= proj[2]
        reproj_pt = (proj[0], proj[1])
        error = np.linalg.norm(np.array(reproj_pt) - np.array(pt2d))
        errors.append(error)

    return np.mean(errors)