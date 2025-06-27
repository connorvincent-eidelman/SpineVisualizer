import cv2
import numpy as np
from itertools import combinations
from config import CHECKERBOARD, CALIBRATION_SAMPLES, SQUARE_SIZE_CM
import time

def capture_frames(caps):
    return [cap.read()[1] for cap in caps]

def wait_with_timer(caps, seconds=5):
    start_time = time.time()
    while time.time() - start_time < seconds:
        frames = [cap.read()[1] for cap in caps]
        remaining = int(seconds - (time.time() - start_time)) + 1

        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.putText(frame, f"Next capture in {remaining}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.imshow(f"Camera {i}", cv2.resize(frame, (640, 480)))

        if cv2.waitKey(1) & 0xFF == 27:
            break
def flush_camera_buffers(caps, flush_count=5):
    for _ in range(flush_count):
        for cap in caps:
            cap.read()

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
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                corners_per_cam[i] = corners2

        if all(found):
            objpoints.append(objp)
            for i in range(len(caps)):
                imgpoints[i].append(corners_per_cam[i])
            samples += 1
            print(f"Captured sample {samples}/{CALIBRATION_SAMPLES}")
            wait_with_timer(caps, seconds=10.0)
            flush_camera_buffers(caps)
        else:
            print("Checkerboard not visible on all cameras.")

        # Display corners
        for i, frame in enumerate(frames):
            disp = frame.copy()
            if found[i] and corners_per_cam[i] is not None:
                disp = cv2.drawChessboardCorners(disp, CHECKERBOARD, corners_per_cam[i], True)
            cv2.imshow(f"Camera {i}", cv2.resize(disp, (640, 480)))
        if cv2.waitKey(500) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return objpoints, imgpoints

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    return total_error / len(objpoints)

def calibrate_individual_cameras(objpoints, imgpoints, caps):
    calibrations = []
    for i in range(len(caps)):
        ret, frame = caps[i].read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objp_cam = objpoints[:len(imgpoints[i])]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp_cam, imgpoints[i], gray.shape[::-1], None, None)

        error = compute_reprojection_error(objp_cam, imgpoints[i], rvecs, tvecs, mtx, dist)
        print(f"Camera {i} calibrated. Reprojection error: {error:.4f}")
        calibrations.append((mtx, dist))
    return calibrations

def stereo_calibrate_all(objpoints, imgpoints, intrinsics, image_shape):
    extrinsics = {}

    for i, j in combinations(range(len(intrinsics)), 2):
        mtx1, dist1 = intrinsics[i]
        mtx2, dist2 = intrinsics[j]

        retval, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objpoints, imgpoints[i], imgpoints[j],
            mtx1, dist1, mtx2, dist2, image_shape,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        print(f"Stereo calibration between camera {i} and {j} RMS error: {retval:.4f}")

        # Transform to reference (camera 0)
        extrinsics[(i, j)] = (R, T)

    return extrinsics

def build_projection_matrices(intrinsics, extrinsics, reference_cam=0):
    proj_mats = []
    proj_mats.append(intrinsics[reference_cam][0] @ np.hstack((np.eye(3), np.zeros((3, 1)))))

    for i in range(1, len(intrinsics)):
        if (reference_cam, i) in extrinsics:
            R, T = extrinsics[(reference_cam, i)]
        elif (i, reference_cam) in extrinsics:
            R_raw, T_raw = extrinsics[(i, reference_cam)]
            R = R_raw.T
            T = -R @ T_raw
        else:
            raise ValueError(f"No extrinsic calibration between camera {reference_cam} and camera {i}")

        P = intrinsics[i][0] @ np.hstack((R, T))
        proj_mats.append(P)

    return proj_mats
