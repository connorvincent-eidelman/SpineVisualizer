import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam (Mac camera)
mac_cam = cv2.VideoCapture(0)

# Open iPhone IP stream (change this IP to match your iPhone stream)
iphone_cam = cv2.VideoCapture("http://192.168.1.143:4747/video")  # Replace with your stream URL

def process_frame(frame):
    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    h, w, _ = frame.shape
    shoulder_distance = None

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        x1, y1 = int(left.x * w), int(left.y * h)
        x2, y2 = int(right.x * w), int(right.y * h)

        cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        shoulder_distance = math.hypot(x2 - x1, y2 - y1)
        cv2.putText(frame, f"Shoulder Width: {shoulder_distance:.1f}px", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return frame

while True:
    ret1, frame1 = mac_cam.read()
    ret2, frame2 = iphone_cam.read()

    if not ret1 or not ret2:
        print("Could not read from one or both cameras.")
        break

    # Resize for consistency
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # Process both frames for pose detection and annotation
    processed1 = process_frame(frame1)
    processed2 = process_frame(frame2)

    # Stack vertically
    combined = cv2.vconcat([processed1, processed2])

    # Display the result
    cv2.imshow("Shoulder Width - Mac (Top) & iPhone (Bottom)", combined)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

mac_cam.release()
iphone_cam.release()
cv2.destroyAllWindows()
