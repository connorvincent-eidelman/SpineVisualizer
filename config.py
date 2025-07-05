CHECKERBOARD = (6, 9)
CALIBRATION_SAMPLES = 15
SQUARE_SIZE_CM = 2.5  # real-world size of checker square (in cm)

CAMERA_IDS = [
    0, 
    "http://192.168.1.180:4747/video", 
    "http://192.168.1.143:4747/video"
]

# Selected landmarks for spine posture modeling
from mediapipe.python.solutions.pose import PoseLandmark
SPINE_LANDMARKS = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
]