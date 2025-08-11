import cv2
import pickle
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import json
from datetime import datetime
import threading
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

with open('trained_model/svm_classifier.pkl', 'rb') as f:
    svm, label_encoder = pickle.load(f)

threshold = 0.75 

def get_face_embedding(img):
    """Returns embedding for a single face image"""
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            return resnet(face.unsqueeze(0).to(device)).cpu().numpy()
    return None

def recognize_embedding(embedding):
    """Predict label from embedding"""
    probs = svm.predict_proba(embedding)[0]
    max_prob = np.max(probs)
    if max_prob < threshold:
        return "Unknown"
    pred = np.argmax(probs)
    return label_encoder.inverse_transform([pred])[0]


def recognize_from_image(image_path):
    """Recognize face(s) from an image file, save annotated image + JSON metadata."""
    
    img = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

   
    boxes, _ = mtcnn.detect(img)

    results = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_pil = img.crop((x1, y1, x2, y2))
            embedding = get_face_embedding(face_pil)
            if embedding is not None:
                full_label = recognize_embedding(embedding)
                if '_' in full_label:
                    id_str, name_str = full_label.split('_', 1)
                else:
                    id_str, name_str = "unknown", full_label

                results.append({
                    "id": id_str,
                    "name": name_str,
                    "box": [x1, y1, x2, y2]
                })

                display_text = f"{name_str} ({id_str})"
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_cv, display_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = "inference"
    os.makedirs(out_dir, exist_ok=True)
    img_out_path = os.path.join(out_dir, f"{base_name}_{timestamp}.jpg")
    json_out_path = os.path.join(out_dir, f"{base_name}_{timestamp}.json")

    cv2.imwrite(img_out_path, img_cv)
    with open(json_out_path, 'w') as jf:
        json.dump(results, jf, indent=2)

    print(f"Saved annotated image to {img_out_path}")
    print(f"Saved metadata to {json_out_path}")

    return results if results else None




stop_event = threading.Event()

def recognize_from_webcam():
    """Run webcam recognition in a loop until stop_event is set."""
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' to stop.")
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except:
                    continue

        cv2.imshow("SVM Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
        print("Starting webcam. Type 'q' + Enter anytime to stop.")

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
