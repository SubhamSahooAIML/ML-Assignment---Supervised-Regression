# ML-Assignment---Supervised-Regression
Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of 255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0. You may generate a dataset as required for solving the problem. Please explain your rationale behind dataset choices.
# Supervised Regression: Single Pixel Localization

## 📌 Project Overview
[cite_start]This project solves the machine learning assignment of predicting the `(x, y)` coordinates of a single active pixel (value `255`) in a `50x50` grayscale image where all other pixels are `0`[cite: 3, 4]. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to perform supervised continuous regression.

## 📂 Project Structure

| File / Directory | Purpose |
| :--- | :--- |
| 🛠️ **[`check_dependencies.py`](https://github.com/SubhamSahooAIML/ML-Assignment---Supervised-Regression/blob/main/check_dependencies.py)** | An automated script to verify and install required libraries before running the project. |
| ⚙️ **[`dataset_generator.py`](https://github.com/SubhamSahooAIML/ML-Assignment---Supervised-Regression/blob/main/dataset_generator.py)** | Generates the synthetic dataset (images & labels), defines the CNN architecture, trains the model, and plots training loss. |
| 🔍 **[`loading_model.py`](https://github.com/SubhamSahooAIML/ML-Assignment---Supervised-Regression/blob/main/Loading_model.py)** | Loads the trained model, performs inference on unseen test images, and plots ground truth vs. predicted coordinates. |
| 📦 **`pixel_regression_model.keras`** | The saved weights and architecture of the successfully trained model. |
| 📁 **`images/`** & 📄 **`labels.csv`** | The generated dataset files (these are created dynamically when the generator script is run). |

## ⚙️ Installation & Setup
[cite_start]This project includes a dependency management script for easy setup. 

## for jupiter file
you can download and check the dependency and then run the ipynb file for jupiter 

1. Ensure you have Python 3.8+ installed.
2. Open your terminal in the project directory.
3. Run the automated dependency checker:
   ```bash
   python check_dependencies.py

# OUTPUTS
 Loading data into memory...
Training data shape: (8000, 50, 50, 1)
Validation data shape: (2000, 50, 50, 1)
Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 48, 48, 16)     │           160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 24, 24, 16)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 22, 22, 32)     │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 11, 11, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 3872)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │       247,872 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 252,802 (987.51 KB)

 Trainable params: 252,802 (987.51 KB)

 Non-trainable params: 0 (0.00 B)

Starting training...
Epoch 1/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 16s 59ms/step - loss: 0.0525 - mae: 0.1410 - val_loss: 0.0021 - val_mae: 0.0312
Epoch 2/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 19s 53ms/step - loss: 0.0013 - mae: 0.0239 - val_loss: 6.4307e-04 - val_mae: 0.0173
Epoch 3/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 14s 54ms/step - loss: 5.8372e-04 - mae: 0.0175 - val_loss: 5.9667e-04 - val_mae: 0.0185
Epoch 4/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 14s 54ms/step - loss: 3.4539e-04 - mae: 0.0139 - val_loss: 2.7213e-04 - val_mae: 0.0122
Epoch 5/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 21s 55ms/step - loss: 2.3105e-04 - mae: 0.0114 - val_loss: 2.8962e-04 - val_mae: 0.0136
Epoch 6/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 20s 55ms/step - loss: 1.7705e-04 - mae: 0.0102 - val_loss: 1.9117e-04 - val_mae: 0.0102
Epoch 7/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 54ms/step - loss: 1.7878e-04 - mae: 0.0100 - val_loss: 8.7137e-05 - val_mae: 0.0071
Epoch 8/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 14s 55ms/step - loss: 9.1788e-05 - mae: 0.0073 - val_loss: 1.1167e-04 - val_mae: 0.0079
Epoch 9/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 54ms/step - loss: 1.2686e-04 - mae: 0.0086 - val_loss: 1.0934e-04 - val_mae: 0.0080
Epoch 10/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 53ms/step - loss: 1.1623e-04 - mae: 0.0082 - val_loss: 1.6768e-04 - val_mae: 0.0100
Epoch 11/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 14s 54ms/step - loss: 1.1972e-04 - mae: 0.0083 - val_loss: 1.1555e-04 - val_mae: 0.0081
Epoch 12/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 20s 53ms/step - loss: 1.4255e-04 - mae: 0.0091 - val_loss: 1.0106e-04 - val_mae: 0.0077
Epoch 13/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 20s 53ms/step - loss: 9.7217e-05 - mae: 0.0076 - val_loss: 4.7745e-05 - val_mae: 0.0053
Epoch 14/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 53ms/step - loss: 1.1347e-04 - mae: 0.0082 - val_loss: 2.2800e-04 - val_mae: 0.0112
Epoch 15/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 52ms/step - loss: 1.3847e-04 - mae: 0.0090 - val_loss: 1.2843e-04 - val_mae: 0.0089
Epoch 16/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 53ms/step - loss: 1.2260e-04 - mae: 0.0085 - val_loss: 1.5953e-04 - val_mae: 0.0100
Epoch 17/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 53ms/step - loss: 1.2264e-04 - mae: 0.0084 - val_loss: 1.9024e-04 - val_mae: 0.0117
Epoch 18/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 53ms/step - loss: 9.3150e-05 - mae: 0.0074 - val_loss: 1.4922e-04 - val_mae: 0.0095
Epoch 19/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 14s 54ms/step - loss: 7.0478e-05 - mae: 0.0063 - val_loss: 4.5420e-05 - val_mae: 0.0052
Epoch 20/20
250/250 ━━━━━━━━━━━━━━━━━━━━ 13s 53ms/step - loss: 4.5703e-05 - mae: 0.0052 - val_loss: 8.5502e-05 - val_mae: 0.0070
Model successfully saved to pixel_regression_model.keras!

#OUTPUT OF MODEL AND METRICS

 Loading trained model...
Model loaded successfully.
Loading dataset...
Dataset loaded: (10000, 50, 50, 1) (10000, 2)
Running inference on dataset...
313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step

📊 VALIDATION METRICS
Mean pixel error : 0.5421653
Std pixel error  : 0.35053822
Max pixel error  : 2.6223183

🎯 REGRESSION ACCURACY
Accuracy within 1 pixel : 89.86%
Accuracy within 2 pixels: 99.68%
Accuracy within 3 pixels: 100.00%

