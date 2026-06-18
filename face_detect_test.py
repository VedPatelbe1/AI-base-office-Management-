import cv2
import os
import time
import threading
from insightface.app import FaceAnalysis
from config import RTSP_URL

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPStream:
    """Threaded RTSP reader — always keeps ONLY the latest frame."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded.")

stream = RTSPStream(RTSP_URL)
if not stream.cap.isOpened():
    print("ERROR: Could not open RTSP stream.")
    exit()

# Shared detection results, written by the detection thread, read by the display loop
faces = []
faces_lock = threading.Lock()
running = True


def detection_worker():
    """Continuously detects faces on the latest frame, decoupled from display."""
    global faces
    while running:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        detected = app.get(frame)
        with faces_lock:
            faces = detected


det_thread = threading.Thread(target=detection_worker, daemon=True)
det_thread.start()

print("Running. Press 'q' to quit.")
while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        time.sleep(0.01)
        continue

    with faces_lock:
        current_faces = faces

    for face in current_faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{face.det_score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
stream.release()
cv2.destroyAllWindows()