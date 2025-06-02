# === File: main.py ===
import cv2
import mediapipe as mp
import numpy as np

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
from spine_modeling_utils import LandmarkSmoother, get_spine_points, fit_spine_curve, compute_lateral_deviation, project_curve_to_image

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

# Function to stack frames in a grid
def stack_frames_grid(frames, grid_shape):
    rows = []
    for i in range(0, len(frames), grid_shape[1]):
        row = cv2.hconcat(frames[i:i + grid_shape[1]])
        rows.append(row)
    return cv2.vconcat(rows)

# Main loop
while True:
    frames = capture_frames(caps)
    landmarks_per_cam = {}

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks_per_cam[i] = results.pose_landmarks.landmark

    triangulated = {}
    if len(landmarks_per_cam) >= 2:
        triangulated_raw = triangulate_landmarks(
            landmarks_per_cam,
            proj_mats,
            [lm.value for lm in SPINE_LANDMARKS]
        )

        # Apply smoothing
        for lid, pt3d in triangulated_raw.items():
            triangulated[lid] = smoother.smooth(lid, pt3d)

        # Fit spine curve and compute deviation
        spine_pts = get_spine_points(triangulated)
        curve = fit_spine_curve(spine_pts) if spine_pts else None
        lateral_dev = compute_lateral_deviation(curve) if curve is not None else None
        if lateral_dev is not None:
            print(f"Lateral Deviation: {lateral_dev:.2f} cm")

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

            # Draw projected landmarks
            for lid, pt3d in triangulated.items():
                pt2d = project_point(pt3d)
                cv2.circle(frame, pt2d, 4, (255, 255, 0), -1)

            # Shoulder distance
            if 11 in triangulated and 12 in triangulated:
                p11 = project_point(triangulated[11])
                p12 = project_point(triangulated[12])
                dist = compute_distance(triangulated[11], triangulated[12])
                draw_line(p11, p12, f"{dist:.1f} cm", color=(0, 255, 0))

            # Spine angle
            if 23 in triangulated and 24 in triangulated and 0 in triangulated:
                p23 = project_point(triangulated[23])
                p24 = project_point(triangulated[24])
                p0 = project_point(triangulated[0])
                angle = compute_angle(triangulated[23], triangulated[24], triangulated[0])
                draw_line(p23, p24, f"{angle:.1f}°", color=(255, 0, 0))

            # Draw spine curve on body
            if curve is not None:
                curve_2d = project_curve_to_image(curve, proj_mat)
                for j in range(len(curve_2d) - 1):
                    cv2.line(frame, curve_2d[j], curve_2d[j + 1], (0, 0, 255), 2)

    # Combine all views into one window
    resized_frames = [cv2.resize(f, (640, 480)) for f in frames]
    combined = stack_frames_grid(resized_frames, grid_shape=(1, len(resized_frames)))
    cv2.imshow("Multi-Cam View", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
