import torch
import torch.nn as nn
import cv2
import json
from torchvision import models, transforms
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1) Model ve Sınıf Yükleme
# =========================
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# Kaç sınıf varsa ayarlayın, örneğin 4:
model.classifier[6] = nn.Linear(4096, 3)
model = model.to(device)

model.load_state_dict(torch.load("vgg_unknown.pth", map_location=device))
model.eval()

with open("class_names.json", "r") as f:
    class_names = json.load(f)  # ['Ali','Ayse','Veli','unknown'] gibi

# =========================
# 2) Dönüşümler
# =========================
transform_ops = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 3) Yüz Algılama
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        face_tensor = transform_ops(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            class_label = class_names[class_idx]

        # "unknown" ise "Bilinmiyor" yaz
        text_label = "Bilinmiyor" if class_label == "unknown" else class_label

        # Yüz etrafına dikdörtgen ve metin
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text_label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
