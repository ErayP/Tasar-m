import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Kaydedilmiş özellik vektörlerini yükle
feature_vectors = np.load("feature_vectors.npy", allow_pickle=True).item()

# Kamera ve ResNet modeli başlatma
cap = cv2.VideoCapture(0)
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to("cuda" if torch.cuda.is_available() else "cpu")
resnet.eval()

# Dönüşüm işlemi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Yüz algılama
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (224, 224))
        face_tensor = transform(Image.fromarray(face_resized)).unsqueeze(0).to("cuda")

        # Özellik vektörü çıkarma
        with torch.no_grad():
            live_feature = resnet(face_tensor).cpu().numpy().flatten()
            live_feature = live_feature / np.linalg.norm(live_feature)

        # Kayıtlı özellik vektörleriyle karşılaştırma
        max_similarity = 0
        best_match = "Bilinmiyor"
        for person_name, feature_vector in feature_vectors.items():
            feature_vector = feature_vector / np.linalg.norm(feature_vector)
            similarity = cosine_similarity([live_feature], [feature_vector])[0][0]

            print(f"Kişi: {person_name}, Benzerlik: {similarity*100}")  # Benzerlikleri ekrana yazdır
            if similarity > max_similarity and similarity > 0.001:  # Optimize edilmiş eşik değeri
                max_similarity = similarity
                best_match = person_name

        # Sonucu ekranda göster
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
