import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # Grafik için
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split

# F1 Skoru için:
from sklearn.metrics import f1_score

# =======================
# 1) Veri hazırlığı
# =======================
data_dir = "Yuz_Verileri"  # Tek klasör ise random_split kullanacağız

transform_all = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_all)
class_names = full_dataset.classes
num_classes = len(class_names)

dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)  # %80 train, %20 val
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

# =======================
# 2) Model Seçimi - VGG16
# =======================
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, num_classes)  # son katmanı değiştir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =======================
# 3) Optimizasyon
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =======================
# 4) Erken Durdurma İçin Parametreler
# =======================
num_epochs = 10
best_val_loss = float('inf')
patience = 2  # en fazla kaç epoch üst üste kötüleşmeye izin vereceğiz
counter = 0   # erken durdurmayı tetiklemek için sayaç

# =======================
# 5) Performans Takibi İçin Listeler
# =======================
train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []
train_f1s    = []
val_f1s      = []

for epoch in range(num_epochs):
    # --------------------------
    # A) EĞİTİM AŞAMASI
    # --------------------------
    model.train()
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0

    # F1 skorunu hesaplamak için gerçek (labels) ve tahmin (predictions) listeleri:
    train_preds_list = []
    train_labels_list = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Kayıp hesapları
        running_loss_train += loss.item() * images.size(0)

        # Tahminler ve doğru etiketler
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # CPU'ya alıp listeye ekleyelim:
        train_preds_list.extend(predicted.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())

    epoch_loss_train = running_loss_train / len(train_loader.dataset)
    epoch_acc_train  = 100.0 * correct_train / total_train
    # F1 Skoru (weighted):
    train_f1 = f1_score(train_labels_list, train_preds_list, average="weighted") * 100.0

    # --------------------------
    # B) DOĞRULAMA AŞAMASI
    # --------------------------
    model.eval()
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0

    # F1 skorunu hesaplamak için listeler:
    val_preds_list = []
    val_labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss_val += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            val_preds_list.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    epoch_loss_val = running_loss_val / len(val_loader.dataset)
    epoch_acc_val  = 100.0 * correct_val / total_val
    val_f1 = f1_score(val_labels_list, val_preds_list, average="weighted") * 100.0

    # --------------------------
    # C) Ekrana Yazdırma
    # --------------------------
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  * Training Loss: {epoch_loss_train:.4f}, Acc: {epoch_acc_train:.2f}%, F1: {train_f1:.2f}")
    print(f"  * Validation Loss: {epoch_loss_val:.4f}, Acc: {epoch_acc_val:.2f}%, F1: {val_f1:.2f}")

    # Performans listelerine ekle
    train_losses.append(epoch_loss_train)
    val_losses.append(epoch_loss_val)
    train_accs.append(epoch_acc_train)
    val_accs.append(epoch_acc_val)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    # --------------------------
    # D) Erken Durdurma Kontrolü
    # --------------------------
    if epoch_loss_val < best_val_loss:
        best_val_loss = epoch_loss_val
        torch.save(model.state_dict(), "best_vgg_model.pth")
        print("  -> Model kaydedildi (best_vgg_model.pth).")
        counter = 0
    else:
        counter += 1
        print(f"  -> Validation loss kötüleşti. (Counter: {counter}/{patience})")
        if counter >= patience:
            print("  -> Erken durdurma tetiklendi!")
            break

# =======================
# 6) Eğitim Sonrası Grafikler
# =======================
epochs_trained = len(train_losses)  # Erken durdurma olduysa 10'dan az olabilir

plt.figure(figsize=(18, 5))

# 1) Loss Grafiği
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs_trained+1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, epochs_trained+1), val_losses, label="Val Loss", marker='s')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 2) Accuracy Grafiği
plt.subplot(1, 3, 2)
plt.plot(range(1, epochs_trained+1), train_accs, label="Train Acc", marker='o')
plt.plot(range(1, epochs_trained+1), val_accs, label="Val Acc", marker='s')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

# 3) F1 Score Grafiği
plt.subplot(1, 3, 3)
plt.plot(range(1, epochs_trained+1), train_f1s, label="Train F1", marker='o')
plt.plot(range(1, epochs_trained+1), val_f1s, label="Val F1", marker='s')
plt.title("F1 Score over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score (%)")
plt.legend()

plt.tight_layout()
plt.show()
