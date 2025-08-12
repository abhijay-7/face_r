from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import pickle
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import json
import time
import threading
from datetime import datetime
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
import ssl

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


collection_data = {"count": 0, "target": 20, "person_id": "", "person_name": ""}

knn_model = None
svm_model = None
label_encoder = None

def load_models():
    global knn_model, svm_model, label_encoder
    try:
        if os.path.exists('trained_model/knn_classifier.pkl'):
            with open('trained_model/knn_classifier.pkl', 'rb') as f:
                knn_model = pickle.load(f)
        
        if os.path.exists('trained_model/svm_classifier.pkl'):
            with open('trained_model/svm_classifier.pkl', 'rb') as f:
                svm_model, label_encoder = pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {e}")

def create_self_signed_cert():
    """save a self-signed certificate for HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from datetime import datetime, timedelta
        import ipaddress
        import socket
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Face Recognition System"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("*.localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address(local_ip)),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        os.makedirs('ssl', exist_ok=True)
        
        with open('ssl/cert.pem', 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open('ssl/key.pem', 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        print(f"Self-signed certificate created for localhost and {local_ip}")
        return True
        
    except ImportError:
        print("cryptography library not found")
        return False
    except Exception as e:
        print(f"Error creating certificate: {e}")
        return False

def get_face_embedding(img):
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            return resnet(face.unsqueeze(0).to(device)).cpu().numpy()
    return None

def rec_embedding(embedding, model_type='knn', threshold=0.9):

    if model_type == 'knn' and knn_model is not None:
        distance, _ = knn_model.kneighbors(embedding, n_neighbors=1)
        if distance[0][0] > threshold:
            return "Unknown", 0.0
        prediction = knn_model.predict(embedding)[0]
        confidence = max(0, (threshold - distance[0][0]) / threshold * 100)
        return prediction, confidence
    
    elif model_type == 'svm' and svm_model is not None:
        prediction = svm_model.predict(embedding)[0]
        probabilities = svm_model.predict_proba(embedding)[0]
        confidence = max(probabilities) * 100
        
        if confidence < 70: 
            return "Unknown", confidence
        
        return label_encoder.inverse_transform([prediction])[0], confidence
    
    return "Unknown", 0.0

def get_dataset_info():
  
    if not os.path.exists('data'):
        return {"identities": 0, "total_images": 0, "folders": []}
    
    folders = []
    total_images = 0
    
    for folder in os.listdir('data'):
        folder_path = os.path.join('data', folder)
        if os.path.isdir(folder_path):
            image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            folders.append({"name": folder, "images": image_count})
            total_images += image_count
    
    return {
        "identities": len(folders),
        "total_images": total_images,
        "folders": folders
    }

@app.route('/')
def index():
    return render_template('mod_index.html')

@app.route('/api/system-status')
def system_status():
    dataset_info = get_dataset_info()
    active_models = []
    
    if knn_model is not None:
        active_models.append("KNN")
    if svm_model is not None:
        active_models.append("SVM")
    
    return jsonify({
        "device": device.upper(),
        "identities": dataset_info["identities"],
        "total_images": dataset_info["total_images"],
        "active_models": active_models
    })

@app.route('/api/dataset-info')
def dataset_info():
    return jsonify(get_dataset_info())


@app.route('/api/start-collection', methods=['POST'])
def start_collection():
    global collection_data
    
    data = request.json
    person_id = data.get('person_id', '').strip()
    person_name = data.get('person_name', '').strip()
    
    if not person_id or not person_name:
        return jsonify({"success": False, "message": "Person ID and Name are required"})
    
    folder_name = f"{person_id}_{person_name}"
    folder_path = os.path.join("data", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    collection_data = {
        "count": 0,
        "target": 20,
        "person_id": person_id,
        "person_name": person_name,
        "folder_path": folder_path
    }
    
    return jsonify({"success": True, "message": "Collection started"})

@app.route('/api/collection-status')
def collection_status():
    return jsonify({
        "count": collection_data["count"],
        "target": collection_data["target"],
        "person_name": collection_data.get("person_name", "")
    })

@app.route('/api/save-collected-image', methods=['POST'])
def save_collected_image():
    global collection_data
    
    if collection_data["count"] >= collection_data["target"]:
        return jsonify({"success": False, "message": "Collection target reached"})
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"success": False, "message": "No image data received"})
    
    try:
        image_data = image_data.split(',')[1] 
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        np_img = np.array(img)
        
        boxes, _ = mtcnn.detect(np_img)
        
        if boxes is not None and len(boxes) > 0:
            # Take the first detected face
            x1, y1, x2, y2 = map(int, boxes[0])
            face = np_img[y1:y2, x1:x2]
            
            if face.size > 0:
                face_resized = cv2.resize(face, (160, 160))
                face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                
                filename = f"{collection_data['person_id']}_{collection_data['person_name']}_{collection_data['count'] + 1}.jpg"
                filepath = os.path.join(collection_data['folder_path'], filename)
                cv2.imwrite(filepath, face_bgr)
                
                collection_data["count"] += 1
                
                return jsonify({
                    "success": True, 
                    "count": collection_data["count"],
                    "face_detected": True,
                    "bbox": [x1, y1, x2, y2]
                })
        
        return jsonify({"success": True, "face_detected": False})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/api/recognize-webcam-frame', methods=['POST'])
def recognize_webcam_frame():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"success": False, "message": "No image data received"})
    
    try:
        image_data = image_data.split(',')[1] 
        image_bytes = base64.b64decode(image_data)
        
        
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
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
                    name, confidence = rec_embedding(embedding)
                    results.append({
                        "name": name,
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2]
                    })
        
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/recognize-image', methods=['POST'])
def recognize_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "message": "No image selected"})
    
    try:
      
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        img = Image.open(filepath).convert('RGB')
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
                    name, confidence = rec_embedding(embedding)
                    results.append({
                        "name": name,
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(np_img, f"{name} ({confidence:.1f}%)", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        os.makedirs("inference", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"inference/result_{timestamp}.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
        
        os.remove(filepath)
        
        return jsonify({
            "success": True,
            "results": results,
            "result_image": result_path
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/generate-embeddings', methods=['POST'])
def generate_embeddings():
    try:
        from generate_embeddings import process_dataset
        process_dataset()
        return jsonify({"success": True, "message": "Embeddings generated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    

@app.route('/api/train-classifier', methods=['POST'])
def train_classifier():
    data = request.json
    method = data.get('method', 'knn')
    
    try:
        from train_face_r_classifier import train_classifier as train_func
        train_func(method=method)
        
        load_models()
        
        return jsonify({
            "success": True, 
            "message": f"{method.upper()} classifier trained successfully"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/clear-data', methods=['POST'])
def clear_data():
    try:
        import shutil
        if os.path.exists('data'):
            shutil.rmtree('data')
        if os.path.exists('trained_model'):
            shutil.rmtree('trained_model')
        if os.path.exists('inference'):
            shutil.rmtree('inference')
        
        global knn_model, svm_model, label_encoder
        knn_model = None
        svm_model = None
        label_encoder = None
        
        return jsonify({"success": True, "message": "All data cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/export-results', methods=['POST'])
def export_results():
    try:
        import zipfile
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"face_recognition_results_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            if os.path.exists('inference'):
                for root, dirs, files in os.walk('inference'):
                    for file in files:
                        zipf.write(os.path.join(root, file))
        
        return send_file(zip_filename, as_attachment=True)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

def get_local_ip():
    """Get local IP address"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def run_http_server():
    print("Starting HTTP server...")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

def run_https_server():
    if not os.path.exists('ssl/cert.pem') or not os.path.exists('ssl/key.pem'):
        print("Creating self-signed certificate...")
        if not create_self_signed_cert():
            print("Failed to create certificate. HTTPS server will not start.")
            return
    
    print("Starting HTTPS server...")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('ssl/cert.pem', 'ssl/key.pem')
    
    app.run(debug=False, host='0.0.0.0', port=5443, ssl_context=context, threaded=True)

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('inference', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    load_models()
    
    local_ip = get_local_ip()
    
    print("Face Recognition System Starting...")
    print("=" * 50)
    print(f"   Local access:")
    print(f"   HTTP:  http://localhost:5000")
    print(f"   HTTPS: https://localhost:5443")
    print(f"   Network access:")
    print(f"   HTTP:  http://{local_ip}:5000")
    print(f"   HTTPS: https://{local_ip}:5443")
    print("=" * 50)
    print("=" * 50)
    
    import threading

    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    time.sleep(1)
    
    try:
        run_https_server()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    except Exception as e:
        print(f"Error starting HTTPS server: {e}")
        print("HTTP server is still running on port 5000")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down servers...")