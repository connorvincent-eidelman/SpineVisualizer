import cv2
import math
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms as T

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MiDaS model (Large)
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Camera setup
mac_cam = cv2.VideoCapture(0)
iphone_cam = cv2.VideoCapture("http://192.168.1.58:4747/video")  # Update to your IP stream

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(img)
    
    input_tensor = input_tensor.to(device)
    

    with torch.no_grad():
        prediction = midas(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    return depth_colored

def annotate_pose(frame):
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        x1, y1 = int(left.x * w), int(left.y * h)
        x2, y2 = int(right.x * w), int(right.y * h)

        cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        distance = math.hypot(x2 - x1, y2 - y1)
        cv2.putText(frame, f"Shoulder Width: {distance:.1f}px", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return frame

while True:
    ret1, frame1 = mac_cam.read()
    ret2, frame2 = iphone_cam.read()

    if not ret1 or not ret2:
        print("Could not read from one or both cameras.")
        break

    # Resize
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # Annotate pose
    processed1 = annotate_pose(frame1.copy())
    processed2 = annotate_pose(frame2.copy())

    # Estimate depth
    depth1 = estimate_depth(frame1)
    depth2 = estimate_depth(frame2)

    # Stack views
    top_row = cv2.hconcat([processed1, depth1])
    bottom_row = cv2.hconcat([processed2, depth2])
    combined = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("Mac + iPhone Pose + Depth", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

mac_cam.release()
iphone_cam.release()
cv2.destroyAllWindows()
