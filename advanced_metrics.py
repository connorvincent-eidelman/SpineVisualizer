import numpy as np
from metrics_utils import compute_distance

def compute_advanced_metrics(triangulated, curve):
    metrics = {}

    # Shoulder Tilt
    if 11 in triangulated and 12 in triangulated:
        shoulder_vec = np.array(triangulated[12]) - np.array(triangulated[11])
        tilt_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
        metrics["Shoulder Tilt"] = tilt_angle

    # Pelvic Tilt
    if 23 in triangulated and 24 in triangulated:
        hip_vec = np.array(triangulated[24]) - np.array(triangulated[23])
        pelvic_tilt = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
        metrics["Pelvic Tilt"] = pelvic_tilt

    # Forward Head Offset
    if all(k in triangulated for k in [7, 8, 11, 12]):
        head_mid = (np.array(triangulated[7]) + np.array(triangulated[8])) / 2
        shoulder_mid = (np.array(triangulated[11]) + np.array(triangulated[12])) / 2
        forward_offset = head_mid[2] - shoulder_mid[2]  # z-axis
        metrics["Forward Head Offset"] = forward_offset

    # Torso Length
    if all(k in triangulated for k in [0, 23, 24]):
        pelvis_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
        torso_len = compute_distance(triangulated[0], pelvis_mid)
        metrics["Torso Length"] = torso_len

    # Spine Curvature Ratio
    if curve is not None and len(curve) > 1:
        spine_len = np.sum([np.linalg.norm(curve[i] - curve[i-1]) for i in range(1, len(curve))])
        spine_height = np.linalg.norm(curve[-1] - curve[0])
        if spine_height > 0:
            curvature_ratio = spine_len / spine_height
            metrics["Spine Curvature Ratio"] = curvature_ratio

    # Neck Tilt
    if 0 in triangulated and 1 in triangulated:
        neck_vec = np.array(triangulated[0]) - np.array(triangulated[1])
        norm = np.linalg.norm(neck_vec)
        if norm > 0:
            neck_tilt = np.degrees(np.arccos(np.dot(neck_vec, [0,1,0]) / norm))
            metrics["Neck Tilt"] = neck_tilt

    # Body Lean
    if all(k in triangulated for k in [11, 12, 23, 24]):
        shoulder_mid = (np.array(triangulated[11]) + np.array(triangulated[12])) / 2
        hip_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
        torso_vec = shoulder_mid - hip_mid
        norm = np.linalg.norm(torso_vec)
        if norm > 0:
            torso_angle = np.degrees(np.arccos(np.dot(torso_vec, [0,1,0]) / norm))
            metrics["Body Lean"] = torso_angle

    return metrics
