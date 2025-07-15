import cv2
import os
import time

# Create a folder to store images
folder_name = "captured_images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

countdown_seconds = 10
print(f"Starting in {countdown_seconds} seconds...")
for i in range(countdown_seconds, 0, -1):
    print(f"{i}...", end='\r')
    time.sleep(1)

print("Starting image capture now!")

# Start capturing from the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Generate file path
        filename = os.path.join(folder_name, f"photo_{count:04d}.jpg")

        # Save the frame as an image
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        count += 1
        time.sleep(1)  # Wait for 1 second

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cap.release()
    print("Camera released.")
