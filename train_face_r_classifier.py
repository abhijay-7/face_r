import pickle
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def load_embeddings(embedding_path):
    with open(embedding_path, 'rb') as f:
        embeddings, names = pickle.load(f)
    return np.array(embeddings), np.array(names)

def train_knn(embeddings, names, output_path):
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(embeddings, names)
    with open(output_path, 'wb') as f:
        pickle.dump(knn, f)
    print(f"KNN classifier trained on {len(set(names))} people and saved to {output_path}")

def train_svm(embeddings, names, output_path):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(names)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(embeddings, labels)
    with open(output_path, 'wb') as f:
        pickle.dump((svm, label_encoder), f)
    print(f"SVM classifier trained on {len(label_encoder.classes_)} people and saved to {output_path}")

def train_classifier(method='svm',
                     embedding_path='trained_model/face_embeddings.pkl',
                     output_path=None):
    embeddings, names = load_embeddings(embedding_path)
    
    if method == 'svm':
        output_path = output_path or 'trained_model/svm_classifier.pkl'
        train_svm(embeddings, names, output_path)
    elif method == 'knn':
        output_path = output_path or 'trained_model/knn_classifier.pkl'
        train_knn(embeddings, names, output_path)
    else:
        raise ValueError("Invalid method. Choose 'svm' or 'knn'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train face recognition classifier (SVM or KNN).")
    parser.add_argument('--method', choices=['svm', 'knn'], default='svm', help="Classifier method to use")
    parser.add_argument('--embedding_path', default='trained_model/face_embeddings.pkl', help="Path to embeddings pickle file")
    parser.add_argument('--output_path', default=None, help="Path to save trained model")

    args = parser.parse_args()
    train_classifier(
        method=args.method,
        embedding_path=args.embedding_path,
        output_path=args.output_path
    )
