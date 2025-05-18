import cv2
import torch
import mediapipe as mp
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# Ask user for their real height in cm (used to scale 2D to real-world)
USER_HEIGHT_CM = 170  # ← Replace with user input or GUI later

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Step 1: Get Depth Map ---
    input_tensor = transform(img_rgb)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # --- Step 2: Get 2D Body Landmarks ---
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        h, w = frame.shape[:2]

        # Get two keypoints for body height scale (neck to ankle)
        idx_top = mp_pose.PoseLandmark.NOSE.value
        idx_bottom = mp_pose.PoseLandmark.RIGHT_ANKLE.value

        pt_top = results.pose_landmarks.landmark[idx_top]
        pt_bottom = results.pose_landmarks.landmark[idx_bottom]

        pixel_height = abs((pt_bottom.y - pt_top.y) * h)
        if pixel_height > 0:
            scale = USER_HEIGHT_CM / pixel_height
        else:
            scale = 1.0  # fallback

        # Draw and print 3D coordinates of some keypoints
        for lm in [mp_pose.PoseLandmark.LEFT_SHOULDER,
                   mp_pose.PoseLandmark.RIGHT_SHOULDER,
                   mp_pose.PoseLandmark.LEFT_HIP,
                   mp_pose.PoseLandmark.RIGHT_HIP]:
            lmk = results.pose_landmarks.landmark[lm]
            cx, cy = int(lmk.x * w), int(lmk.y * h)

            # Get depth value at that pixel
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                z = depth_map[cy, cx] * scale
                x = cx * scale
                y = cy * scale

                # Display 3D coordinates
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{lm.name}: ({x:.1f}, {y:.1f}, {z:.1f})",
                            (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("3D Pose Estimation", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
