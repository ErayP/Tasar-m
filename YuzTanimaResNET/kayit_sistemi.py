import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import random

# Yüz algılama için Haar Cascade modeli yükleniyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def augment_image(image):
    # Veri artırma işlemleri
    angle = random.uniform(-15, 15)  # Döndürme
    image = image.rotate(angle)
    brightness_factor = random.uniform(0.7, 1.3)  # Parlaklık
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    if random.random() > 0.5:  # %50 ihtimalle yatay döndürme
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def capture_face_and_save():
    person_name = input("Lütfen kaydedilecek kişinin adını girin: ").strip()
    if not person_name:
        print("Geçersiz isim! Lütfen tekrar deneyin.")
        return

    try:
        max_images = int(input("Kaç yüz resmi kaydedilsin (örneğin: 250): "))
        if max_images <= 0:
            print("Geçerli bir sayı giriniz.")
            return
    except ValueError:
        print("Lütfen bir sayı giriniz!")
        return

    save_dir = os.path.join("Yuz_Verileri", person_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"'{save_dir}' dizinine veri kaydedilecek.")

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while frame_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamıyor!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            if frame_count >= max_images:
                break

            face_roi = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (224, 224))

            # Veri artırma ve kaydetme
            for i in range(3):  # Her yüz için 3 artırılmış versiyon
                augmented_face = augment_image(Image.fromarray(face_resized))
                augmented_face.save(os.path.join(save_dir, f"face_{frame_count}_{i}.jpg"))

            frame_count += 1

            # Yüzün etrafına mavi dikdörtgen çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Alınan kare sayısını sol üst köşeye yaz
        cv2.putText(frame, f"Kare Sayisi: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Yüz Algılama", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("İşlem kullanıcı tarafından durduruldu.")
            break

    print(f"{frame_count} yüz görüntüsü '{save_dir}' dizinine kaydedildi.")
    cap.release()
    cv2.destroyAllWindows()

capture_face_and_save()
