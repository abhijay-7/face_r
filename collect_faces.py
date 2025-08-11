import cv2
import os
import time
from mtcnn import MTCNN
import numpy as np


detector = MTCNN()

def create_dir(person_id, name):
    folder_name = f"{person_id}_{name}"
    path = os.path.join("data", folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def extract_face(frame):
    results = detector.detect_faces(frame)
    if results:
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)
        face = frame[y:y+h, x:x+w]
        return face
    return None

def collect_faces(person_id, name, num_samples=20):
    save_path = create_dir(person_id, name)
    cap = cv2.VideoCapture(0)
    count = 0
    last_capture_time = 0

    print(f"Collecting faces for '{name}' (ID: {person_id}) every 0.5s... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        face = extract_face(frame)

        if face is not None and (current_time - last_capture_time) >= 0.5:
            face = cv2.resize(face, (160, 160))
            file_path = os.path.join(save_path, f"{person_id}_{name}_{count+1}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            last_capture_time = current_time

            cv2.rectangle(frame, (0, 0), (250, 30), (0, 255, 0), -1)
            cv2.putText(frame, f"Saved {count}/{num_samples}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Collecting Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} face images for '{name}' (ID: {person_id})")

if __name__ == "__main__":
    person_id = input("Enter unique ID: ").strip()
    person_name = input("Enter name of the person: ").strip()
    collect_faces(person_id, person_name)
