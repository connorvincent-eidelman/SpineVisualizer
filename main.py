import cv2
import mediapipe as mp
import numpy as np

from config import CAMERA_IDS, SPINE_LANDMARKS
from calibration_utils import find_chessboard_corners, calibrate_individual_cameras, capture_frames
from triangulation_utils import triangulate_landmarks
from metrics_utils import compute_distance, compute_angle

caps = [cv2.VideoCapture(cid) for cid in CAMERA_IDS]
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Calibration
print("Starting calibration...")
objpoints, imgpoints = find_chessboard_corners(caps)
calibrations = calibrate_individual_cameras(objpoints, imgpoints, caps)

# Stereo projection matrices
proj_mats = []
for mtx, dist, rvecs, tvecs in calibrations:
    R, _ = cv2.Rodrigues(rvecs[0])
    t = tvecs[0]
    P = mtx @ np.hstack((R, t))
    proj_mats.append(P)

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

        if triangulated:
            # Draw lines between key points
            def draw_line(pt1, pt2, label):
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]), int(pt2[1]))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)
                cv2.putText(frame, label, (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Example: Shoulder width
            if 11 in triangulated and 12 in triangulated:
                dist = compute_distance(triangulated[11], triangulated[12])
                draw_line((w//2 - 100, 30), (w//2 + 100, 30), f"Shoulder Width: {dist:.1f} cm")

            # Spine tilt
            if 23 in triangulated and 24 in triangulated and 0 in triangulated:
                angle = compute_angle(triangulated[23], triangulated[24], triangulated[0])
                cv2.putText(frame, f"Spine Angle: {angle:.1f}°", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(f"Camera {i}", cv2.resize(frame, (640, 480)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
