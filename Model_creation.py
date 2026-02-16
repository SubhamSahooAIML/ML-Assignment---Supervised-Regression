# ===================== IMPORTS =====================
import os
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ===================== CONFIG =====================
IMAGE_SIZE = 50
NUM_SAMPLES = 10000
IMAGE_FOLDER = "images"
LABEL_FILE = "labels.csv"
MODEL_PATH = "pixel_regression_model"
EPOCHS = 20
BATCH_SIZE = 32

# 🔁 CHANGE THIS FLAG
TRAIN_MODEL = True   # True = train & save | False = load & predict

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ===================== TRAINING PIPELINE =====================
if TRAIN_MODEL:

    print("Generating dataset...")

    with open(LABEL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "row", "col"])

        for i in range(NUM_SAMPLES):
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

            row = np.random.randint(0, IMAGE_SIZE)
            col = np.random.randint(0, IMAGE_SIZE)

            img[row, col] = 255

            filename = f"img_{i}.png"
            cv2.imwrite(os.path.join(IMAGE_FOLDER, filename), img)

            writer.writerow([
                filename,
                row / IMAGE_SIZE,
                col / IMAGE_SIZE
            ])

    print("Dataset generated.")

    # ===================== LOAD DATA =====================
    labels = pd.read_csv(LABEL_FILE)

    X, Y = [], []

    for _, r in labels.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, r["filename"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img / 255.0
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

        X.append(img)
        Y.append([r["row"], r["col"]])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # ===================== MODEL =====================
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="linear")
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    model.summary()

    # ===================== TRAIN =====================
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # ===================== LOSS PLOT =====================
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # ===================== SAVE MODEL =====================
    model.save(r"C:\Users\Acer\OneDrive\Desktop\New folder\pixel_regression_model.keras")

    print("Model saved!")


