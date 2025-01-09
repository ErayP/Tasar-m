import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def read_and_increment():
    try:
        with open("person.txt", "r") as file:
            content = file.read().strip()
            if content:
                difFace = int(content)
            else:
                difFace = 0
    except FileNotFoundError: pass

    return difFace

imgs = glob.glob("./Faces/*.jpg")
difFace = read_and_increment()
width = 223
height = 223

X=[]
Y=[]

for img in imgs:
    filename = os.path.basename(img)
    label = filename.split("_")[0][-1]
    im= np.array(Image.open(img).convert("L").resize((width,height)))
    im = im /255
    X.append(im)
    Y.append(label)

X=np.array(X)
X=X.reshape(X.shape[0],width,height,1)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoder = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse_output=False)  
    integer_encoder = integer_encoder.reshape(len(integer_encoder), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoder)
    return onehot_encoded


train_X,test_X,train_y,test_y = train_test_split(X,Y,test_size=0.25,random_state=2)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(width, height, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(difFace, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.fit(train_X, train_y, epochs=35, batch_size=64)

score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %", score_train[1] * 100)

score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %", score_test[1] * 100)

open("model.json","w").write(model.to_json())

model.save_weights("faces.weights.h5")