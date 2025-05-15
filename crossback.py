import cv2
import mediapipe 
import math
import sys
print(sys.executable)

# Initialize MediaPipe pose
mp_pose = mediapipe.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mediapipe.solutions.drawing_utils

# Start the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Get left and right shoulders
        left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        x1, y1 = int(left.x * w), int(left.y * h)
        x2, y2 = int(right.x * w), int(right.y * h)

        # Draw points and line
        cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Calculate pixel distance
        distance_px = math.hypot(x2 - x1, y2 - y1)
        cv2.putText(frame, f"Shoulder Width: {distance_px:.1f}px", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Live Shoulder Distance", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
