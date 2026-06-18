import cv2
from config import RTSP_URL

# Force TCP transport — more reliable than UDP for RTSP over WiFi
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: Could not open RTSP stream. Check the URL/network.")
    exit()

print("Stream opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to grab frame. Retrying...")
        continue

    cv2.imshow("Xiaomi C200 - RTSP Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()