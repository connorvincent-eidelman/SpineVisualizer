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
        triangulated = triangulate_landmarks(
            landmarks_per_cam,
            proj_mats,
            [lm.value for lm in SPINE_LANDMARKS]
        )

        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            proj_mat = proj_mats[i]

            def project_point(point3d):
                pt = np.append(point3d / 100.0, 1)  # convert cm to meters if needed
                proj = proj_mat @ pt
                proj /= proj[2]
                return int(proj[0]), int(proj[1])

            def draw_line(pt1, pt2, label, anchor=None):
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                pos = anchor if anchor else ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 - 10)
                cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if triangulated:
            # Shoulder distance
                if 11 in triangulated and 12 in triangulated:
                    p11 = project_point(triangulated[11])
                    p12 = project_point(triangulated[12])
                    dist = compute_distance(triangulated[11], triangulated[12])
                    draw_line(p11, p12, f"{dist:.1f} cm")
                    cv2.circle(frame, p11, 5, (0, 255, 0), -1)
                    cv2.circle(frame, p12, 5, (0, 255, 0), -1)

            # Spine angle (between hips and nose)
                if 23 in triangulated and 24 in triangulated and 0 in triangulated:
                    p23 = project_point(triangulated[23])
                    p24 = project_point(triangulated[24])
                    p0 = project_point(triangulated[0])
                    angle = compute_angle(triangulated[23], triangulated[24], triangulated[0])
                    draw_line(p23, p24, f"{angle:.1f}°", anchor=p0)
                    cv2.circle(frame, p23, 5, (255, 0, 0), -1)
                    cv2.circle(frame, p24, 5, (255, 0, 0), -1)
                    cv2.circle(frame, p0, 5, (255, 0, 0), -1)

            cv2.imshow(f"Camera {i}", cv2.resize(frame, (640, 480)))


    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
