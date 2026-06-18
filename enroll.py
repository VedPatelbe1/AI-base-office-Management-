import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from database import SessionLocal, Employee, embedding_to_bytes

MIN_SAMPLES = 5
CAM_INDEX = 0


def main():
    if len(sys.argv) >= 3:
        name, emp_code = sys.argv[1].strip(), sys.argv[2].strip()
    else:
        name = input("Enter person's name: ").strip()
        emp_code = input("Enter employee code (e.g. EMP001): ").strip()

    if not name or not emp_code:
        print("Name and employee code are required.")
        return

    session = SessionLocal()
    if session.query(Employee).filter_by(emp_code=emp_code).first():
        print(f"Code '{emp_code}' already exists. Aborting.")
        session.close()
        return

    print("Loading InsightFace model...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open laptop camera. Try CAM_INDEX = 1.")
        session.close()
        return

    embeddings = []
    win = f"Register: {name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 600)
    print("SPACE = capture   q = finish & save")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = app.get(frame)
        ready = len(faces) == 1 and faces[0].det_score > 0.6
        if ready:
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hint = "Ready - press SPACE"
        elif len(faces) == 0:
            hint = "No face detected"
        else:
            hint = "One face only"

        cv2.putText(frame, f"{name}   captured: {len(embeddings)}   {hint}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and ready:
            embeddings.append(faces[0].normed_embedding)
            print(f"Captured {len(embeddings)}")
        elif key == ord("q"):
            if len(embeddings) >= MIN_SAMPLES:
                break
            print(f"Need at least {MIN_SAMPLES} (have {len(embeddings)}).")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < MIN_SAMPLES:
        print("Not enough samples; nothing saved.")
        session.close()
        return

    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)
    session.add(Employee(name=name, emp_code=emp_code, embedding=embedding_to_bytes(avg)))
    session.commit()
    session.close()
    print(f"Enrolled {name} ({emp_code}) with {len(embeddings)} photos.")


if __name__ == "__main__":
    main()

# import sys
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from database import SessionLocal, Employee, embedding_to_bytes

# MIN_SAMPLES = 5
# CAM_INDEX = 0


# def main():
#     # Take name/code from the dashboard if passed, else ask in the terminal
#     if len(sys.argv) >= 3:
#         name, emp_code = sys.argv[1].strip(), sys.argv[2].strip()
#     else:
#         name = input("Enter person's name: ").strip()
#         emp_code = input("Enter employee code (e.g. EMP001): ").strip()

#     if not name or not emp_code:
#         print("Name and employee code are required.")
#         return

#     session = SessionLocal()
#     if session.query(Employee).filter_by(emp_code=emp_code).first():
#         print(f"Code '{emp_code}' already exists. Aborting.")
#         session.close()
#         return

#     print("Loading model...")
#     app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
#     app.prepare(ctx_id=0, det_size=(640, 640))

#     cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         print("Could not open laptop camera. Try CAM_INDEX = 1.")
#         session.close()
#         return

#     embeddings = []
#     win = f"Register: {name}"
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(win, 900, 600)
#     print("SPACE = capture   q = finish & save")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         faces = app.get(frame)
#         ready = len(faces) == 1 and faces[0].det_score > 0.6
#         if ready:
#             x1, y1, x2, y2 = faces[0].bbox.astype(int)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             hint = "Ready - press SPACE"
#         elif len(faces) == 0:
#             hint = "No face detected"
#         else:
#             hint = "One face only"

#         cv2.putText(frame, f"{name}   captured: {len(embeddings)}   {hint}",
#                     (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         cv2.imshow(win, frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord(" ") and ready:
#             embeddings.append(faces[0].normed_embedding)
#             print(f"Captured {len(embeddings)}")
#         elif key == ord("q"):
#             if len(embeddings) >= MIN_SAMPLES:
#                 break
#             print(f"Need at least {MIN_SAMPLES} (have {len(embeddings)}).")

#     cap.release()
#     cv2.destroyAllWindows()

#     avg = np.mean(embeddings, axis=0)
#     avg = avg / np.linalg.norm(avg)
#     session.add(Employee(name=name, emp_code=emp_code, embedding=embedding_to_bytes(avg)))
#     session.commit()
#     session.close()
#     print(f"Enrolled {name} ({emp_code}) with {len(embeddings)} photos.")


# if __name__ == "__main__":
#     main()