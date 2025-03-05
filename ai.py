# from PIL import Image
import os
import cv2
import joblib
import requests
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from sklearn.svm import SVC
from db import EmbeddingData
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
directories = ['images', 'models', 'face']
model_dir = os.path.join(os.path.dirname(__file__), 'models')
rec_path = os.path.join(model_dir, 'face_recognition.onnx')
cls_path = os.path.join(model_dir, 'face_classifier.pkl')
url = 'https://thanglongedu-my.sharepoint.com/:u:/g/personal/a44212_thanglong_edu_vn/Ef3sjhgaRKNFqOrzTAi7ZgcBmef8hzm37GGOTTAZsuFTlw?download=1'

engine = create_engine('sqlite:///embeddings.db')
Session = sessionmaker(bind=engine)
session = Session()

def load_model():
    #Kiểm tra sự tồn tại của các thư mục cần thiết
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Thư mục '{directory}' đã được tạo.")
        else:
            print(f"Thư mục '{directory}' đã tồn tại.")
    #Kiểm tra File ONNX
    if not os.path.exists(rec_path):
        print("File ONNX chưa tồn tại, đang tải về...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(rec_path, 'wb') as f:
                f.write(response.content)
            print("Tải file thành công.")
        else:
            print(f"Không thể tải file ONNX, mã trạng thái: {response.status_code}")
    rec_model = ort.InferenceSession(rec_path)
    print("Model ONNX sẵn sàng.")

    #Kiểm tra File SVC
    if not os.path.exists(cls_path):
        print("File SVC chưa tồn tại, đang tái tạo...")
        cls_model = SVC(kernel='linear', probability=True)
        joblib.dump(cls_model, cls_path)
    cls_model = joblib.load(cls_path)
    print("Model SVC sẵn sàng.")

    return rec_model, cls_model

rec_model, cls_model = load_model()

def detect_face(image_path):
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)
    if results.detections:
        best_detection = max(results.detections, key=lambda d: d.score[0])
        bboxC = best_detection.location_data.relative_bounding_box
        img_h, img_w, _ = image.shape
        x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
        x1, y1 = int(x * img_w), int(y * img_h)
        x2, y2 = int((x + w) * img_w), int((y + h) * img_h)
        face = image[y1:y2, x1:x2]
        face_filename = 'face/face_image.jpg'
        cv2.imwrite(face_filename, face)
        return face_filename
    return None

def process_image(image_path):
    global rec_model
    face_image_path = detect_face(image_path)
    if face_image_path is None: return None
    img = cv2.imread(face_image_path)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)

def train(images_path, label):
    global cls_model, cls_path, rec_model
    session.query(EmbeddingData).filter(EmbeddingData.label == label).delete()
    for image_path in images_path:
        face_input = process_image(image_path)
        output = rec_model.run(None, {rec_model.get_inputs()[0].name: face_input})
        embedding = output[0][0].tolist()
        instance = EmbeddingData(embedding=embedding, label=label)
        session.add(instance)
        session.commit()
        os.remove(image_path)

    data = session.query(EmbeddingData).all()
    embeddings = [item.embedding for item in data]
    labels = [item.label for item in data]
    if len(set(labels)) == 1:
        print("Chỉ có một nhãn duy nhất, thêm embedding giả...")
        dummy_embedding = np.random.rand(len(embeddings[0])).tolist()
        embeddings.append(dummy_embedding)
        labels.append("unknown")

    cls_model = SVC(kernel='linear', probability=True)
    cls_model.fit(embeddings, labels)
    joblib.dump(cls_model, cls_path)
    print("Huấn luyện lại thành công")
    print(cls_model.classes_)
    return {'status':'Thành công', 'classes':cls_model.classes_}

def predict(image_path):
    global cls_model, rec_model, cls_path
    face_input = process_image(image_path)
    outputs = rec_model.run(None, {rec_model.get_inputs()[0].name: face_input})
    predicted_label = cls_model.predict(outputs[0])
    return predicted_label[0]

