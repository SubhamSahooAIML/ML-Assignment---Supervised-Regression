# ===================== IMPORTS =====================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ===================== CONFIG =====================
IMAGE_SIZE = 50
IMAGE_FOLDER = "images"
LABEL_FILE = "labels.csv"

# 👉 CHANGE THIS TO YOUR MODEL PATH
MODEL_PATH = r"C:\Users\Acer\OneDrive\Desktop\New folder\pixel_regression_model.keras"

# ===================== LOAD MODEL =====================
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ===================== LOAD DATA =====================
print("Loading dataset...")

labels = pd.read_csv(LABEL_FILE)

X = []
Y = []

for _, row in labels.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row["filename"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = img / 255.0
    img = img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

    X.append(img)
    Y.append([row["row"], row["col"]])

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print("Dataset loaded:", X.shape, Y.shape)

# ===================== PREDICTION =====================
print("Running inference on dataset...")
preds = model.predict(X)

# Convert normalized → pixel coordinates
pred_pixels = preds * IMAGE_SIZE
true_pixels = Y * IMAGE_SIZE

# ===================== METRICS =====================
pixel_errors = np.sqrt(
    (pred_pixels[:, 0] - true_pixels[:, 0])**2 +
    (pred_pixels[:, 1] - true_pixels[:, 1])**2
)

print("\n📊 VALIDATION METRICS")
print("Mean pixel error :", np.mean(pixel_errors))
print("Std pixel error  :", np.std(pixel_errors))
print("Max pixel error  :", np.max(pixel_errors))

# ===================== PREDICTED vs ACTUAL PLOT =====================
plt.figure(figsize=(6, 6))

plt.scatter(true_pixels[:, 0], pred_pixels[:, 0],
            alpha=0.4, label="X-coordinate")

plt.scatter(true_pixels[:, 1], pred_pixels[:, 1],
            alpha=0.4, label="Y-coordinate")

plt.plot([0, IMAGE_SIZE], [0, IMAGE_SIZE], "k--", label="Ideal")

plt.xlabel("Ground Truth Pixel")
plt.ylabel("Predicted Pixel")
plt.title("Predicted vs Actual Pixel Coordinates")
plt.legend()
plt.grid(True)
plt.show()

# ===================== ERROR DISTRIBUTION =====================
plt.figure()
plt.hist(pixel_errors, bins=30)
plt.xlabel("Pixel Error (Euclidean Distance)")
plt.ylabel("Frequency")
plt.title("Pixel Localization Error Distribution")
plt.show()

# ===================== VISUAL SAMPLE CHECK =====================
for i in range(5):
    idx = np.random.randint(0, len(X))

    img = X[idx].squeeze()
    true = true_pixels[idx]
    pred = pred_pixels[idx]

    plt.imshow(img, cmap="gray")
    plt.scatter(true[1], true[0], c="green", label="Ground Truth")
    plt.scatter(pred[1], pred[0], c="red", label="Prediction")
    plt.legend()
    plt.title(f"Sample {idx}")
    plt.show()
# ===================== REGRESSION ACCURACY =====================

def regression_accuracy(pixel_errors, threshold):
    """
    Accuracy = percentage of predictions within 'threshold' pixels
    """
    return np.mean(pixel_errors <= threshold) * 100


acc_1px = regression_accuracy(pixel_errors, 1)
acc_2px = regression_accuracy(pixel_errors, 2)
acc_3px = regression_accuracy(pixel_errors, 3)

print("\n🎯 REGRESSION ACCURACY")
print(f"Accuracy within 1 pixel : {acc_1px:.2f}%")
print(f"Accuracy within 2 pixels: {acc_2px:.2f}%")
print(f"Accuracy within 3 pixels: {acc_3px:.2f}%")

#OPTIONAL (but impressive): Accuracy vs Tolerance plot
thresholds = range(0, 6)
accuracies = [regression_accuracy(pixel_errors, t) for t in thresholds]

plt.figure()
plt.plot(thresholds, accuracies, marker="o")
plt.xlabel("Pixel Tolerance")
plt.ylabel("Accuracy (%)")
plt.title("Regression Accuracy vs Pixel Tolerance")
plt.grid(True)
plt.show()
