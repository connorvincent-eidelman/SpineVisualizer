import cv2
import numpy as np
from config import CHECKERBOARD, CALIBRATION_SAMPLES, SQUARE_SIZE_CM

def capture_frames(caps):
    return [cap.read()[1] for cap in caps]

def find_chessboard_corners(caps):
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM

    objpoints = []
    imgpoints = [[] for _ in range(len(caps))]
    samples = 0

    while samples < CALIBRATION_SAMPLES:
        frames = capture_frames(caps)
        found = [False] * len(caps)
        corners_per_cam = [None] * len(caps)

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                found[i] = True
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_per_cam[i] = corners2

        if all(found):
            objpoints.append(objp)
            for i in range(len(caps)):
                imgpoints[i].append(corners_per_cam[i])
            samples += 1
            print(f"Captured sample {samples}/{CALIBRATION_SAMPLES}")
        else:
            print("Checkerboard not visible on all cameras.")

        # Display
        for i, frame in enumerate(frames):
            disp = frame.copy()
            if found[i] and corners_per_cam[i] is not None:
                disp = cv2.drawChessboardCorners(disp, CHECKERBOARD, corners_per_cam[i], True)
            cv2.imshow(f"Camera {i}", cv2.resize(disp, (640, 480)))
        if cv2.waitKey(500) & 0xFF == 27:  # Esc
            break

    return objpoints, imgpoints

def calibrate_individual_cameras(objpoints, imgpoints, caps):
    calibrations = []
    for i in range(len(caps)):
        ret, frame = caps[i].read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Repeat objpoints to match length of imgpoints[i]
        objp_cam = objpoints[:len(imgpoints[i])]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp_cam, imgpoints[i], gray.shape[::-1], None, None)

        error = compute_reprojection_error(objp_cam, imgpoints[i], rvecs, tvecs, mtx, dist)
        print(f"Camera {i} calibrated. Reprojection error: {error:.4f}")
        calibrations.append((mtx, dist, rvecs, tvecs))
    return calibrations


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    return total_error
