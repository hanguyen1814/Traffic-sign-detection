import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam

# 1. Chuẩn bị dữ liệu
print("Đang chuẩn bị dữ liệu...")
data = []
labels = []
classes = 43  # 43 loại biển báo

cur_path = os.getcwd()

# Tải ảnh từ thư mục train
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((32, 32))  # Thay đổi kích thước thành 32x32
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Lỗi khi tải ảnh {a}: {e}")

# Chuyển dữ liệu thành numpy array
data = np.array(data)
labels = np.array(labels)

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành one-hot encoding
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

# 2. Hàm xây dựng các mô hình
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mobilenetv2_model():
    base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Đóng băng các layer của MobileNetV2

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet50_model():
    base_model = ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Đóng băng các layer của ResNet50

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate(model, model_name):
    print(f"Đang huấn luyện mô hình: {model_name}...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    model.save(f"{model_name}.h5")

    # Vẽ biểu đồ accuracy và loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.show()

# 4. Kiểm tra trên tập test
def test_model(model_name):
    print(f"Đang kiểm tra mô hình: {model_name}...")
    model = load_model(f"{model_name}.h5")

    # Đọc dữ liệu từ Test.csv
    test_data = pd.read_csv('Test.csv')
    true_labels = test_data["ClassId"].values
    img_paths = test_data["Path"].values

    # Tải và xử lý ảnh trong tập test
    data = []
    for img_path in img_paths:
        try:
            image = Image.open(img_path)
            image = image.resize((32, 32))  # Resize ảnh thành 32x32
            data.append(np.array(image))
        except Exception as e:
            print(f"Lỗi khi tải ảnh {img_path}: {e}")

    X_test = np.array(data)

    # Dự đoán và tính độ chính xác
    predictions = model.predict(X_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(true_labels, predicted_classes)
    print(f"{model_name} Test Accuracy: {accuracy}")

# 5. Huấn luyện và kiểm tra các mô hình
models = {
    "CNN": build_cnn_model(),
    "MobileNetV2": build_mobilenetv2_model(),
    "ResNet50": build_resnet50_model()
}

for model_name, model in models.items():
    train_and_evaluate(model, model_name)
    test_model(model_name)
