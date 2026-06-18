import cv2
import os
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, date
from insightface.app import FaceAnalysis
from config import RTSP_URL
from database import SessionLocal, Employee, Attendance, bytes_to_embedding

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

THRESHOLD = 0.30
WINDOW = "Attendance - Live Recognition"


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


def load_known_faces():
    session = SessionLocal()
    employees = session.query(Employee).all()
    ids, names, embs = [], [], []
    for e in employees:
        ids.append(e.id)
        names.append(e.name)
        embs.append(bytes_to_embedding(e.embedding))
    session.close()
    embs = np.vstack(embs).astype(np.float32) if embs else np.zeros((0, 512), np.float32)
    return ids, names, embs


def export_today(session, target):
    records = (
        session.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .filter(Attendance.date == target)
        .order_by(Attendance.check_in)
        .all()
    )
    rows = [
        {
            "Employee Code": emp.emp_code,
            "Name": emp.name,
            "Date": target.strftime("%d-%m-%Y"),
            "Check-in Time": att.check_in.strftime("%H:%M:%S") if att.check_in else "",
        }
        for att, emp in records
    ]
    df = pd.DataFrame(rows, columns=["Employee Code", "Name", "Date", "Check-in Time"])
    filename = target.strftime("%d-%m-%Y") + ".xlsx"
    df.to_excel(filename, index=False, sheet_name="Attendance")
    print(f"Updated {filename} ({len(df)} check-in(s))")


def mark_attendance(emp_id, name):
    session = SessionLocal()
    today = date.today()
    exists = session.query(Attendance).filter_by(employee_id=emp_id, date=today).first()
    if not exists:
        session.add(Attendance(employee_id=emp_id, date=today, check_in=datetime.now()))
        session.commit()
        print(f"[CHECK-IN] {name} at {datetime.now():%H:%M:%S}")
        export_today(session, today)
    session.close()


print("Loading enrolled faces...")
known_ids, known_names, known_embs = load_known_faces()
print(f"Loaded {len(known_ids)} enrolled people.")

print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(416, 416))
print("Model loaded.")

stream = RTSPStream(RTSP_URL)
if not stream.cap.isOpened():
    print("ERROR: Could not open RTSP stream.")
    exit()

results = []
results_lock = threading.Lock()
checked_today = set()
running = True


def recognition_worker():
    global results
    while running:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        faces = app.get(frame)
        new_results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            if known_embs.shape[0] > 0:
                sims = known_embs @ face.normed_embedding
                best = int(np.argmax(sims))
                score = float(sims[best])
            else:
                best, score = -1, 0.0

            if best >= 0 and score >= THRESHOLD:
                name, emp_id = known_names[best], known_ids[best]
                label = f"{name} ({score:.2f})"
                color = (0, 255, 0)
                if emp_id not in checked_today:
                    mark_attendance(emp_id, name)
                    checked_today.add(emp_id)
            else:
                label = f"Unknown person ({score:.2f})"
                color = (0, 0, 255)

            new_results.append((bbox, label, color))

        with results_lock:
            results = new_results


rec_thread = threading.Thread(target=recognition_worker, daemon=True)
rec_thread.start()

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)   # resizable window
cv2.resizeWindow(WINDOW, 1280, 720)
fullscreen = False

print("Running attendance.  q = quit   f = toggle fullscreen")
while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        time.sleep(0.01)
        continue

    with results_lock:
        current = results

    for bbox, label, color in current:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )

running = False
stream.release()
cv2.destroyAllWindows()