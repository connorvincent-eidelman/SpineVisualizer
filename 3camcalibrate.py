import cv2
import numpy as np
import mediapipe as mp
import time

# ---------- CONFIG ----------
CAMERA_IDS = [0, "http://192.168.1.143:4747/video", "http://192.168.1.174:4747/video"]  # Your cameras
CHECKERBOARD = (6, 9)  # Number of internal corners in checkerboard (height, width)
SQUARE_SIZE_CM = 2.5   # Size of one checkerboard square in cm (adjust to your printout)
CALIBRATION_SAMPLES = 15

# ---------- INIT ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# ---------- FUNCTIONS ----------

def capture_frames():
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to grab frame from one of the cameras")
        frames.append(frame)
    return frames

def find_chessboard_corners(frames):
    # Prepare object points: e.g. (0,0,0), (1,0,0), ..., scaled by square size
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM

    objpoints = []
    imgpoints = [[] for _ in range(len(frames))]

    sample_count = 0
    print(f"Starting automatic calibration capture ({CALIBRATION_SAMPLES} samples required)...")

    while sample_count < CALIBRATION_SAMPLES:
        frames = capture_frames()
        all_corners_found = True
        current_imgpoints = []

        # For display and feedback
        display_frames = []

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                # Refine corner locations
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                current_imgpoints.append(corners2)
                # Draw corners for feedback
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                cv2.putText(frame, f"Cam {i}: Checkerboard found", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                all_corners_found = False
                current_imgpoints.append(None)
                cv2.putText(frame, f"Cam {i}: Checkerboard NOT found", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            display_frames.append(frame)

        # Show all cameras side by side
        combined_display = cv2.hconcat([cv2.resize(f, (640,480)) for f in display_frames])
        cv2.imshow("Calibration - Checkerboard Detection", combined_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit early
            print("Calibration cancelled by user.")
            cv2.destroyAllWindows()
            exit()

        # Capture sample if all cameras detect checkerboard
        if all_corners_found:
            objpoints.append(objp)
            for i, pts in enumerate(current_imgpoints):
                imgpoints[i].append(pts)
            sample_count += 1
            print(f"Captured sample {sample_count}/{CALIBRATION_SAMPLES}")
            time.sleep(0.5)  # Pause to avoid duplicate captures

    cv2.destroyAllWindows()
    return objpoints, imgpoints

def calibrate_cameras(objpoints, imgpoints, frames):
    calibrations = []
    reproj_errors = []

    for i in range(len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints[i], gray.shape[::-1], None, None)
        if not ret:
            raise RuntimeError(f"Calibration failed for camera {i}")
        error = compute_reprojection_error(objpoints, imgpoints[i], rvecs, tvecs, mtx, dist)
        calibrations.append((mtx, dist, rvecs, tvecs))
        reproj_errors.append(error)
        print(f"Camera {i} calibrated. Reprojection error: {error:.4f}")
    return calibrations, reproj_errors

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        
        # Reshape for comparison: both to Nx2
        imgpoints_actual = imgpoints[i].reshape(-1, 2)
        imgpoints_proj = imgpoints2.reshape(-1, 2)

        error = cv2.norm(imgpoints_actual, imgpoints_proj, cv2.NORM_L2)
        total_error += error**2
        total_points += len(objpoints[i])
    return np.sqrt(total_error / total_points)


def stereo_calibrate_pair(objpoints, imgpoints1, imgpoints2, calib1, calib2, image_size):
    mtx1, dist1, _, _ = calib1
    mtx2, dist2, _, _ = calib2

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        mtx1,
        dist1,
        mtx2,
        dist2,
        image_size,
        criteria=criteria,
        flags=flags
    )
    print(f"Stereo calibration done. RMS error: {retval}")
    return R, T

def get_shoulder_points(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        x1, y1 = int(l_sh.x * w), int(l_sh.y * h)
        x2, y2 = int(r_sh.x * w), int(r_sh.y * h)

        return (x1, y1), (x2, y2)
    return None, None

def triangulate_points(p1, p2, calib1, calib2, R, T):
    mtx1, dist1, _, _ = calib1
    mtx2, dist2, _, _ = calib2

    # Undistort points
    p1_ud = cv2.undistortPoints(np.array([[p1]], dtype=np.float64), mtx1, dist1)
    p2_ud = cv2.undistortPoints(np.array([[p2]], dtype=np.float64), mtx2, dist2)

    # Projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((R, T))

    P1 = mtx1 @ P1
    P2 = mtx2 @ P2

    pts_4d = cv2.triangulatePoints(P1, P2, p1_ud.reshape(2,1), p2_ud.reshape(2,1))
    pts_3d = pts_4d[:3] / pts_4d[3]
    return pts_3d.flatten()

# ---------- MAIN ----------

print("Opening cameras...")
caps = [cv2.VideoCapture(cid) for cid in CAMERA_IDS]
time.sleep(2)  # Allow cameras to warm up

# 1. Capture calibration samples automatically
objpoints, imgpoints = find_chessboard_corners(capture_frames())

# 2. Calibrate each camera
initial_frames = capture_frames()
calibrations, reproj_errors = calibrate_cameras(objpoints, imgpoints, initial_frames)

# 3. Stereo calibration for camera pair 0 and 1 (adjust if needed)
image_size = initial_frames[0].shape[1], initial_frames[0].shape[0]
R, T = stereo_calibrate_pair(objpoints, imgpoints[0], imgpoints[1], calibrations[0], calibrations[1], image_size)

print("Calibration complete. Starting live posture tracking with depth triangulation...")

while True:
    frames = capture_frames()
    display_frames = []

    # Get shoulder points from cams 0 and 1 for triangulation
    p1_l, p1_r = get_shoulder_points(frames[0])
    p2_l, p2_r = get_shoulder_points(frames[1])

    if p1_l and p1_r and p2_l and p2_r:
        # Triangulate left shoulders
        left_3d = triangulate_points(p1_l, p2_l, calibrations[0], calibrations[1], R, T)
        # Triangulate right shoulders
        right_3d = triangulate_points(p1_r, p2_r, calibrations[0], calibrations[1], R, T)

        # Compute real-world distance between shoulders (cm)
        shoulder_distance = np.linalg.norm(left_3d - right_3d)

    else:
        shoulder_distance = None

    for i, frame in enumerate(frames):
        p1, p2 = get_shoulder_points(frame)
        if p1 and p2:
            cv2.circle(frame, p1, 6, (255, 0, 0), -1)
            cv2.circle(frame, p2, 6, (0, 255, 0), -1)
            cv2.line(frame, p1, p2, (0, 255, 255), 2)

            if i == 0 and shoulder_distance is not None:
                cv2.putText(frame, f"Shoulder Dist: {shoulder_distance:.2f} cm", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display_frames.append(frame)

    combined = cv2.hconcat([cv2.resize(f, (640, 480)) for f in display_frames])
    cv2.imshow("Posture with Depth Triangulation", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
[]