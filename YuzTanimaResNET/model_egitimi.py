import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.image_paths = []
        self.labels = []

        for label, person_name in enumerate(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person_name)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.augment:
            image = self._augment_image(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _augment_image(self, image):
        angle = random.uniform(-15, 15)
        image = image.rotate(angle)
        brightness_factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

# Veri seti ve transform işlemleri
data_dir = "Yuz_Verileri"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = FaceDataset(data_dir, transform=transform, augment=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Modelin hazırlanması
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()  # Özellik çıkarıcı olarak ayarlanır

# İlk birkaç katmanı dondurun
for name, param in resnet.named_parameters():
    if "layer4" not in name:  # Sadece layer4 ve sonrası eğitilecek
        param.requires_grad = False

resnet = resnet.to("cuda" if torch.cuda.is_available() else "cpu")

# Optimizasyon ve kayıp fonksiyonu
optimizer = optim.AdamW(resnet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# Performans analizini tutmak için
train_losses = []
val_losses = []
accuracies = []
f1_scores = []

# Eğitim döngüsü
best_val_loss = float('inf')
num_epochs = 25
for epoch in range(num_epochs):
    resnet.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Tahminler ve gerçek etiketler
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}")

    # Doğrulama döngüsü
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        resnet.eval()
        for images, labels in dataloader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = resnet(images)
            val_loss += criterion(outputs, labels).item()

            # Tahminler ve gerçek etiketler
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(dataloader)
    val_losses.append(val_loss)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="weighted")
    accuracies.append(val_accuracy)
    f1_scores.append(val_f1)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(resnet.state_dict(), "best_model.pth")
        print("Model kaydedildi.")
    else:
        print("Erken durdurma uygulanıyor.")
        break

# Performans analiz grafikleri
plt.figure(figsize=(12, 6))

# Kayıp grafiği
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Eğitim Kaybı")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Doğrulama Kaybı")
plt.xlabel("Epoch")
plt.ylabel("Kayıp (Loss)")
plt.title("Eğitim ve Doğrulama Kaybı")
plt.legend()

# Doğruluk ve F1 Score grafiği
plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracies) + 1), accuracies, label="Doğruluk (Accuracy)")
plt.plot(range(1, len(f1_scores) + 1), f1_scores, label="F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Skor")
plt.title("Doğruluk ve F1 Score")
plt.legend()

plt.tight_layout()
plt.show()
