import cv2
import numpy as np
import mediapipe as mp

# ---------- CONFIG ----------
CAMERA_IDS = [0, "http://192.168.1.143:4747/video", "http://192.168.1.174:4747/video"]
CHECKERBOARD = (6, 9)
CALIBRATION_SAMPLES = 15

# ---------- INIT ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

print("Opening cameras...")
caps = [cv2.VideoCapture(cid) for cid in CAMERA_IDS]
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------- FUNCTIONS ----------
def capture_frames_from_all_cams():
    return [cap.read()[1] for cap in caps]

def find_chessboard_corners(frames):
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = [[] for _ in range(len(frames))]

    count = 0
    print("Calibration mode: Press [c] to capture, [Esc] to cancel.")

    while count < CALIBRATION_SAMPLES:
        frames = capture_frames_from_all_cams()
        all_detected = True
        detected_corners = []

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                detected_corners.append(corners2)
            else:
                all_detected = False
                detected_corners.append(None)
                cv2.putText(frame, "Checkerboard Not Found", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"Cam {i}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames[i] = frame

        # Show all frames stacked vertically
        stacked = cv2.vconcat([cv2.resize(f, (640, 480)) for f in frames])
        cv2.imshow("Calibration View", stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if all_detected:
                print(f"Captured sample {count + 1}/{CALIBRATION_SAMPLES}")
                objpoints.append(objp)
                for i in range(len(frames)):
                    imgpoints[i].append(detected_corners[i])
                count += 1
            else:
                print("Calibration failed: checkerboard not found on all cameras.")
        elif key == 27:
            print("Calibration canceled.")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()
    return objpoints, imgpoints

def calibrate_cameras(objpoints, imgpoints, frames):
    calibrations = []
    for i in range(len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints[i], gray.shape[::-1], None, None)
        calibrations.append((mtx, dist))
        print(f"Camera {i} calibrated. Reprojection error: {ret:.4f}")
    return calibrations

def get_shoulder_data(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        x1, y1 = int(l_sh.x * w), int(l_sh.y * h)
        x2, y2 = int(r_sh.x * w), int(r_sh.y * h)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        offset = x2 - x1

        return (x1, y1), (x2, y2), angle, offset
    return None, None, None, None

# ---------- MAIN ----------
print("Starting calibration...")
initial_frames = capture_frames_from_all_cams()
objpoints, imgpoints = find_chessboard_corners(initial_frames)
calibrations = calibrate_cameras(objpoints, imgpoints, initial_frames)
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
            cv2.putText(frame, f"Angle: {angle:.1f} deg", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Offset: {offset:.1f}px", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        display.append(frame)

    stacked = cv2.vconcat([cv2.resize(f, (640, 480)) for f in display])
    cv2.imshow("Posture View - Shoulders", stacked)
    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
