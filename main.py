import cv2
import mediapipe as mp
import numpy as np
import time

from config import CAMERA_IDS, SPINE_LANDMARKS
from calibration_utils import (
    find_chessboard_corners,
    calibrate_individual_cameras,
    stereo_calibrate_all,
    build_projection_matrices,
    capture_frames
)
from triangulation_utils import triangulate_landmarks
from metrics_utils import compute_distance, compute_angle
from spine_modeling_utils import LandmarkSmoother, get_spine_points, fit_spine_curve, compute_lateral_deviation, project_curve_to_image, compute_lateral_offsets

# Initialize video capture
caps = [cv2.VideoCapture(cid) for cid in CAMERA_IDS]

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Calibration
print("Starting calibration...")
objpoints, imgpoints = find_chessboard_corners(caps)
intrinsics = calibrate_individual_cameras(objpoints, imgpoints, caps)
gray_shape = cv2.cvtColor(caps[0].read()[1], cv2.COLOR_BGR2GRAY).shape[::-1]
extrinsics = stereo_calibrate_all(objpoints, imgpoints, intrinsics, gray_shape)
proj_mats = build_projection_matrices(intrinsics, extrinsics, reference_cam=0)

# Smoothing
smoother = LandmarkSmoother(alpha=0.3)

frame_shapes = {}
for i, cap in enumerate(caps):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_shapes[i] = (width, height)
    print(f"Camera {i} resolution: {width}x{height}")

# Heatmap initialization
heatmaps = [np.zeros((h, w), dtype=np.float32) for i, (w, h) in frame_shapes.items()]

# Function to stack frames in a grid
def stack_frames_grid(frames, grid_shape):
    target_height = max(f.shape[0] for f in frames)
    target_width = max(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (target_width, target_height)) for f in frames]
    rows = []
    for i in range(0, len(resized), grid_shape[1]):
        row = cv2.hconcat(resized[i:i + grid_shape[1]])
        rows.append(row)
    return cv2.vconcat(rows)


# Function to overlay large metric text
def draw_metrics_overlay(img, metrics, line_height=50, margin=20):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 255, 255)  # Yellow

    for i, text in enumerate(metrics):
        y = margin + i * line_height
        cv2.putText(img, text, (margin, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)

# Main loop
while True:
    frames = capture_frames(caps)

    if any(f is None for f in frames):
        print("⚠️ Warning: One or more camera frames could not be read.")
        continue

    landmarks_per_cam = {}

    for i, frame in enumerate(frames):
        if frame is None:
            continue
        mtx, dist = intrinsics[i]
        frame = cv2.undistort(frame.copy(), mtx, dist)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks_per_cam[i] = results.pose_landmarks.landmark

    triangulated = {}
    confidences_raw = {}
    metrics = ["Lateral Deviation: --", "Shoulder Distance: --", "Spine Angle: --"]

    if len(landmarks_per_cam) >= 2:
        triangulated_raw, raw_confidences = triangulate_landmarks(
            landmarks_per_cam,
            proj_mats,
            [lm.value for lm in SPINE_LANDMARKS],
            frame_shapes
        )

        for lid, pt3d in triangulated_raw.items():
            triangulated[lid] = smoother.smooth(lid, pt3d)
            confidences_raw[lid] = smoother.smooth_confidence(lid, raw_confidences.get(lid, 0.0))

        spine_pts = get_spine_points(triangulated)
        curve = fit_spine_curve(spine_pts) if spine_pts else None
        lateral_dev = compute_lateral_deviation(curve) if curve is not None else None

        if lateral_dev is not None:
            metrics[0] = f"Lateral Deviation: {lateral_dev:.2f} cm"

        if 11 in triangulated and 12 in triangulated:
            dist = compute_distance(triangulated[11], triangulated[12])
            metrics[1] = f"Shoulder Distance: {dist:.1f} cm"

        if 23 in triangulated and 24 in triangulated and 0 in triangulated:
            hip_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
            nose = np.array(triangulated[0])
            spine_vec = nose - hip_mid
            spine_vec /= np.linalg.norm(spine_vec)
            vertical = np.array([0, 1, 0])
            angle_rad = np.arccos(np.clip(np.dot(spine_vec, vertical), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            metrics[2] = f"Spine Angle: {angle_deg:.1f}°"

        for i, frame in enumerate(frames):
            proj_mat = proj_mats[i]
            h, w = frame.shape[:2]

            def project_point(point3d):
                pt = np.append(point3d / 100.0, 1)  # cm to meters
                proj = proj_mat @ pt
                proj /= proj[2]
                return int(proj[0]), int(proj[1])

            def draw_line(pt1, pt2, label, color=(0, 255, 255)):
                cv2.line(frame, pt1, pt2, color, 2)
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2 - 10
                cv2.putText(frame, label, (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for lid, pt3d in triangulated.items():
                conf = confidences_raw.get(lid, 0.0)
                pt2d = project_point(pt3d)

                if 0 <= pt2d[0] < w and 0 <= pt2d[1] < h:
                    heatmaps[i][pt2d[1], pt2d[0]] += conf

                radius = int(4 + 8 * conf)
                color = (
                    int(255 * (1 - conf)),
                    int(255 * conf),
                    0
                )
                cv2.circle(frame, pt2d, radius, color, -1)
                
                cv2.putText(
                    frame, f"{conf:.2f}", (pt2d[0] + 6, pt2d[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )   

            if 11 in triangulated and 12 in triangulated:
                p11 = project_point(triangulated[11])
                p12 = project_point(triangulated[12])
                dist = compute_distance(triangulated[11], triangulated[12])
                draw_line(p11, p12, f"{dist:.1f} cm", color=(0, 255, 0))

            if 23 in triangulated and 24 in triangulated and 0 in triangulated:
                hip_mid = (np.array(triangulated[23]) + np.array(triangulated[24])) / 2
                nose = np.array(triangulated[0])
                p_hip = project_point(hip_mid)
                p_nose = project_point(nose)

                vec_y_screen = np.array([0, -100])
                vertical_end = (p_hip[0] + vec_y_screen[0], p_hip[1] + vec_y_screen[1])
                cv2.line(frame, p_hip, vertical_end, (255, 255, 0), 2)

                cv2.line(frame, p_hip, p_nose, (255, 200, 200), 4)
                label_x = (p_hip[0] + p_nose[0]) // 2
                label_y = (p_hip[1] + p_nose[1]) // 2 - 10
                cv2.putText(frame, f"{angle_deg:.1f}°", (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 0, 200), 3)

            if curve is not None:
                curve_2d = project_curve_to_image(curve, proj_mat)
                lateral_offsets = compute_lateral_offsets(curve)
                for j in range(1, len(curve_2d) - 1):
                    pt = curve[j]
                    base_pt = curve[0]
                    vec = curve[-1] - base_pt
                    vec = vec / np.linalg.norm(vec)
                    proj_len = np.dot(pt - base_pt, vec)
                    proj_point = base_pt + proj_len * vec
                    offset = pt - proj_point
                    color = (100, 100, 255)
                    start = project_point(proj_point)
                    end = project_point(pt)
                    cv2.line(frame, start, end, color, 2)

    for i in range(len(frames)):
        normalized = cv2.normalize(heatmaps[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frames[i], 0.8, colored, 0.4, 0)
        frames[i] = overlay
    if confidences_raw:
        avg_conf = sum(confidences_raw.values()) / len(confidences_raw)
        metrics.append(f"Avg Confidence: {avg_conf:.2f}")
    else:
        metrics.append("Avg Confidence: --")

    combined = stack_frames_grid(frames, grid_shape=(1, len(frames)))
    draw_metrics_overlay(combined, metrics)
    cv2.imshow("Multi-Cam View", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
