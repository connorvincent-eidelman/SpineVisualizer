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

        # Pelvic Obliquity (frontal plane)
        pelvic_obliquity = np.degrees(np.arctan2(hip_vec[1], hip_vec[2]))
        metrics["Pelvic Obliquity"] = pelvic_obliquity

    # Forward Head Offset
    if all(k in triangulated for k in [7, 8, 11, 12]):
        head_mid = (np.array(triangulated[7]) + np.array(triangulated[8])) / 2
        shoulder_mid = (np.array(triangulated[11]) + np.array(triangulated[12])) / 2
        forward_offset = head_mid[2] - shoulder_mid[2]
        metrics["Forward Head Offset"] = forward_offset

    # Torso Length
    if all(k in triangulated for k in [0, 23, 24]):
        pelvis_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
        torso_len = compute_distance(triangulated[0], pelvis_mid)
        metrics["Torso Length"] = torso_len

    # Head-to-Hip Alignment
    if all(k in triangulated for k in [0, 23, 24]):
        pelvis_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
        head = np.array(triangulated[0])
        head_hip_vec = head - pelvis_mid
        norm = np.linalg.norm(head_hip_vec)
        if norm > 0:
            angle = np.degrees(np.arccos(np.clip(np.dot(head_hip_vec, [0, 1, 0]) / norm, -1.0, 1.0)))
            metrics["Head-Hip Alignment Angle"] = angle

    # Neck Tilt
    if 0 in triangulated and 1 in triangulated:
        neck_vec = np.array(triangulated[0]) - np.array(triangulated[1])
        norm = np.linalg.norm(neck_vec)
        if norm > 0:
            neck_tilt = np.degrees(np.arccos(np.dot(neck_vec, [0, 1, 0]) / norm))
            metrics["Neck Tilt"] = neck_tilt

    # Body Lean
    if all(k in triangulated for k in [11, 12, 23, 24]):
        shoulder_mid = (np.array(triangulated[11]) + np.array(triangulated[12])) / 2
        hip_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
        torso_vec = shoulder_mid - hip_mid
        norm = np.linalg.norm(torso_vec)
        if norm > 0:
            torso_angle = np.degrees(np.arccos(np.dot(torso_vec, [0, 1, 0]) / norm))
            metrics["Body Lean"] = torso_angle

    # Spine Curvature Ratio
    if curve is not None and len(curve) > 1:
        spine_len = np.sum([np.linalg.norm(curve[i] - curve[i-1]) for i in range(1, len(curve))])
        spine_height = np.linalg.norm(curve[-1] - curve[0])
        if spine_height > 0:
            curvature_ratio = spine_len / spine_height
            metrics["Spine Curvature Ratio"] = curvature_ratio

        # Thoracic vs Lumbar Curvature
        mid_idx = len(curve) // 2
        thoracic_len = np.sum([np.linalg.norm(curve[i] - curve[i-1]) for i in range(1, mid_idx)])
        lumbar_len = np.sum([np.linalg.norm(curve[i] - curve[i-1]) for i in range(mid_idx, len(curve))])
        if lumbar_len > 0:
            thoraco_lumbar_ratio = thoracic_len / lumbar_len
            metrics["Thoracic/Lumbar Curve Ratio"] = thoraco_lumbar_ratio

        # Spinal Segment Angles
        angles = []
        for i in range(1, len(curve) - 1):
            vec1 = curve[i] - curve[i - 1]
            vec2 = curve[i + 1] - curve[i]
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))
                angles.append(angle)
        if angles:
            metrics["Avg Intersegment Angle"] = np.mean(angles)

        # Sagittal Alignment Offsets
        sagittal_offsets = [pt[2] - curve[0][2] for pt in curve]
        metrics["Sagittal Offset Range"] = np.max(sagittal_offsets) - np.min(sagittal_offsets)

        # Postural Centerline Deviation (X deviation from vertical axis)
        centerline_deviation = [abs(pt[0] - curve[0][0]) for pt in curve]
        metrics["Max Centerline Deviation (X)"] = np.max(centerline_deviation)

        # Spinal Symmetry Index (mirror difference along X)
        mirrored = [np.array([2 * curve[0][0] - pt[0], pt[1], pt[2]]) for pt in curve]
        symmetry_errors = [np.linalg.norm(pt - mpt) for pt, mpt in zip(curve, mirrored)]
        metrics["Spinal Symmetry Index"] = np.mean(symmetry_errors)

    return metrics
