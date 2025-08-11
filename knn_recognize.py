import cv2
import pickle
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading
import os
import json
from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

with open('trained_model/knn_classifier.pkl', 'rb') as f:
    knn = pickle.load(f)

threshold = 0.9
stop_event = threading.Event()


def get_face_embedding(img):
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            return resnet(face.unsqueeze(0).to(device)).cpu().numpy()
    return None

def recognize_embedding(embedding):
    distance, _ = knn.kneighbors(embedding, n_neighbors=1)
    if distance[0][0] > threshold:
        return "Unknown"
    return knn.predict(embedding)[0]

def recognize_from_image(image_path):
    os.makedirs("inference", exist_ok=True)
    img = Image.open(image_path).convert('RGB')
    np_img = np.array(img)
    boxes, _ = mtcnn.detect(np_img)

    results = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = np_img[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)
            embedding = get_face_embedding(face_pil)
            if embedding is not None:
                label = recognize_embedding(embedding)
            else:
                label = "Unknown"

            cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(np_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            results.append({
                "name": label,
                "bbox": [x1, y1, x2, y2]
            })

    else:
        print("No faces detected.")
        return None

 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(image_path).split('.')[0]
    output_img_path = f"inference/{base}_{timestamp}.jpg"
    output_json_path = f"inference/{base}_{timestamp}.json"

    cv2.imwrite(output_img_path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved annotated image to {output_img_path}")
    print(f"Saved detections to {output_json_path}")
    return results

def recognize_from_webcam():
    cap = cv2.VideoCapture(0)
    print("Webcam started. Type 'q' + Enter to stop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = rgb[y1:y2, x1:x2]
                try:
                    face_pil = Image.fromarray(face)
                    embedding = get_face_embedding(face_pil)
                    if embedding is not None:
                        label = recognize_embedding(embedding)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception:
                    continue

        cv2.imshow("KNN Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('`'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = input("Enter mode (img / webcam): ").strip().lower()
    if mode == 'img':
        path = input("Enter image path: ").strip()
        recognize_from_image(path)

    elif mode == 'webcam':
        def listen_for_q():
            while True:
                if input().strip().lower() == 'q':
                    stop_event.set()
                    break

        input_thread = threading.Thread(target=listen_for_q)
        input_thread.start()
        recognize_from_webcam()
        input_thread.join()
        print("Webcam stopped.")

    else:
        print("Invalid mode.")
