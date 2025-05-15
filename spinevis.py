import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load stereo calibration data (replace with your actual calibration results)
# These should be generated using cv2.calibrateCamera() and cv2.stereoCalibrate()
cameraMatrix1 = np.load("cameraMatrix1.npy")
distCoeffs1 = np.load("distCoeffs1.npy")
cameraMatrix2 = np.load("cameraMatrix2.npy")
distCoeffs2 = np.load("distCoeffs2.npy")
R = np.load("R.npy")
T = np.load("T.npy")

# Projection matrices from calibration
P1 = cameraMatrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = cameraMatrix2 @ np.hstack((R, T))

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Indices for spine-related landmarks
SPINE_LANDMARKS = [0, 11, 23]  # Nose, Left Shoulder, Left Hip

def get_landmarks(image):
    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None
    h, w = image.shape[:2]
    landmarks = result.pose_landmarks.landmark
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

def triangulate_3d(points1, points2):
    if points1.shape != points2.shape:
        return np.zeros((0, 3))
    pts1 = points1.T
    pts2 = points2.T
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = pts4d_hom[:3] / pts4d_hom[3]
    return pts3d.T

def main():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    plt.ion()  # Enable interactive plotting
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            break

        landmarks1 = get_landmarks(frame1)
        landmarks2 = get_landmarks(frame2)

        if landmarks1 is not None and landmarks2 is not None:
            spine1 = landmarks1[SPINE_LANDMARKS]
            spine2 = landmarks2[SPINE_LANDMARKS]
            spine3d = triangulate_3d(spine1, spine2)

            # 3D Plot
            ax.clear()
            xs, ys, zs = spine3d[:, 0], spine3d[:, 1], spine3d[:, 2]
            ax.plot(xs, ys, zs, marker='o', color='blue')
            ax.set_xlim([-1000, 1000])
            ax.set_ylim([-1000, 1000])
            ax.set_zlim([0, 2000])
            ax.set_title("3D Spine Points")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.draw()
            plt.pause(0.001)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
