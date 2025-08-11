import os
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device))
        return embedding.squeeze().cpu().numpy()
    return None

def process_dataset(data_path='data'):
    embeddings = []
    names = []

    for person_folder in os.listdir(data_path):
        person_path = os.path.join(data_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        person_id, person_name = person_folder.split("_", 1)
        label = f"{person_id}_{person_name}"

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            embedding = get_face_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                names.append(label)

    embeddings = np.array(embeddings)
    names = np.array(names)

    os.makedirs('trained_model', exist_ok=True)
    with open('trained_model/face_embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings, names), f)

    print(f"Saved {len(embeddings)} embeddings for {len(set(names))} identities.")

if __name__ == "__main__":
    process_dataset()
