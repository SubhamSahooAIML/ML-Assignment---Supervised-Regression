# ML-Assignment---Supervised-Regression
Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of 255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0. You may generate a dataset as required for solving the problem. Please explain your rationale behind dataset choices.
# Supervised Regression: Single Pixel Localization

## 📌 Project Overview
[cite_start]This project solves the machine learning assignment of predicting the `(x, y)` coordinates of a single active pixel (value `255`) in a `50x50` grayscale image where all other pixels are `0`[cite: 3, 4]. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to perform supervised continuous regression.

## 📂 Project Structure
* `[check_dependencies.py](https://github.com/SubhamSahooAIML/ML-Assignment---Supervised-Regression/blob/main/check_dependencies)`: An automated script to verify and install required libraries.
* `[dataset_generator.py](https://github.com/SubhamSahooAIML/ML-Assignment---Supervised-Regression/blob/main/Loading_model.py)` *(or your filename)*: Generates the synthetic dataset of images and coordinate labels. Defines the CNN architecture, trains the model, and plots training loss.
* `[loading_model.py](https://github.com/SubhamSahooAIML/ML-Assignment---Supervised-Regression/blob/main/Loading_model.py)`: Loads the trained model, performs inference on unseen images, and plots ground truth vs. predicted coordinates.
* `labels.csv` & `images/`: The generated dataset (created dynamically).
* `pixel_regression_model.keras`: The saved weights of the trained model.

## ⚙️ Installation & Setup
[cite_start]This project includes a dependency management script for easy setup. 

1. Ensure you have Python 3.8+ installed.
2. Open your terminal in the project directory.
3. Run the automated dependency checker:
   ```bash
   python check_dependencies.py
