import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ---------- CONFIG ----------
CAMERA_IDS = [0, "http://192.168.1.143:4747/video", "http://192.168.1.174:4747/video"]
CHECKERBOARD = (6, 9)
CALIBRATION_SAMPLES = 15

# ---------- INIT ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# ---------- FUNCTIONS ----------
def capture_frames_from_all_cams():
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Frame not received from camera {i}")
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))  # placeholder
        else:
            frames.append(frame)
    return frames

def find_chessboard_corners(frames):
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = [[] for _ in range(len(frames))]

    count = 0
    print("Press 'c' to capture calibration sample, or ESC to skip.")

    while count < CALIBRATION_SAMPLES:
        frames = capture_frames_from_all_cams()
        display_frames = []

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            status_text = f"Camera {i}: {'Found' if ret else 'Not Found'}"

            if ret:
                # Refine corners and draw
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                status_text += f" | Sample {len(imgpoints[i])}/{CALIBRATION_SAMPLES}"

            # Overlay text info
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if ret else (0, 0, 255), 2)

            display_frames.append(cv2.resize(frame, (640, 480)))

        # Show stacked camera views
        stacked = cv2.vconcat(display_frames)
        cv2.imshow("Calibration - Press 'c' to capture, ESC to skip", stacked)

        key = cv2.waitKey(1)
        if key == ord('c'):
            found_all = True
            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    if count == 0:
                        objpoints.append(objp)
                    imgpoints[i].append(corners2)
                else:
                    found_all = False

            if found_all:
                print(f"Sample {count+1} captured.")
                count += 1
            else:
                print("Checkerboard not found in all cameras. Try again.")

        elif key == 27:  # ESC
            print("Calibration canceled/skipped.")
            break

    cv2.destroyWindow("Calibration - Press 'c' to capture, ESC to skip")
    return objpoints, imgpoints

    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = [[] for _ in range(len(frames))]

    count = 0
    print("Press 'c' to capture calibration sample or ESC to skip calibration.")
    while count < CALIBRATION_SAMPLES:
        key = cv2.waitKey(0)
        if key == ord('c'):
            frames = capture_frames_from_all_cams()
            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    imgpoints[i].append(corners2)
                    print(f"Captured corners from camera {i}")
            count += 1
        elif key == 27:  # ESC
            print("Calibration skipped.")
            break

    return objpoints, imgpoints

def calibrate_cameras(objpoints, imgpoints, frames):
    calibrations = []
    for i in range(len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints[i], gray.shape[::-1], None, None)
        calibrations.append((mtx, dist))
    return calibrations

def get_shoulder_data(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        try:
            l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            x1, y1 = int(l_sh.x * w), int(l_sh.y * h)
            x2, y2 = int(r_sh.x * w), int(r_sh.y * h)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            offset = x2 - x1

            return (x1, y1), (x2, y2), angle, offset
        except IndexError:
            pass
    return None, None, None, None

# ---------- MAIN ----------
print("Opening cameras...")
caps = [cv2.VideoCapture(cid) for cid in CAMERA_IDS]
for i, cap in enumerate(caps):
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {CAMERA_IDS[i]}")

print("Starting calibration...")
objpoints, imgpoints = find_chessboard_corners(capture_frames_from_all_cams())
calibrations = calibrate_cameras(objpoints, imgpoints, capture_frames_from_all_cams())
print("Calibration complete. Starting live posture tracking...")

while True:
    frames = capture_frames_from_all_cams()
    display = []

    for i, frame in enumerate(frames):
        p1, p2, angle, offset = get_shoulder_data(frame)

        if p1 and p2:
            cv2.circle(frame, p1, 6, (255, 0, 0), -1)
            cv2.circle(frame, p2, 6, (0, 255, 0), -1)
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
            cv2.putText(frame, f"Angle: {angle:.1f} deg", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Offset: {offset:.1f}px", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        display.append(frame)

    stacked = cv2.vconcat([cv2.resize(f, (640, 480)) for f in display])
    cv2.imshow("Posture View - Shoulders", stacked)
    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
