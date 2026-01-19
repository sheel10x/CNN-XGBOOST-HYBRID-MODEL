# CNN-XGBOOST-HYBRID-MODEL
# 🧠 Brain Tumor Classification: Hybrid CNN-XGBoost Architecture

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.26%25-brightgreen)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)

## 📌 Overview
This project implements a state-of-the-art **Hybrid Medical Image Classification System** to detect brain tumors from MRI scans. 

Unlike standard deep learning approaches, this solution uses a **Fine-Tuned MobileNet (CNN)** as a powerful feature extractor and feeds those features into an **XGBoost Classifier**. This hybrid approach combines the spatial feature learning of CNNs with the gradient boosting power of XGBoost, achieving a remarkable **99.26% test accuracy**.

The model classifies MRI scans into four classes:
1.  **Glioma**
2.  **Meningioma**
3.  **Pituitary Tumor**
4.  **No Tumor**

---

## 🚀 Key Features
* **Hybrid Architecture:** MobileNetV2 backbone (with custom Swish activation) + XGBoost Classifier.
* **High Performance:** Outperforms standard CNN implementations with **99.26% Accuracy**.
* **Web Interface:** Includes a user-friendly **Streamlit** web app for instant predictions.
* **Advanced Preprocessing:** Uses Histogram Equalization and automated resizing.
* **Robust Metrics:** High Precision (99.27%) and Recall (99.26%) across all classes.

---

## 📊 Model Performance

The model was evaluated on a comprehensive test set of **1,345 MRI images**.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.26%** |
| **Precision** | 99.27% |
| **Recall** | 99.26% |
| **F1-Score** | 99.26% |
| **MCC (Matthews Corr. Coeff.)** | 0.9901 |

### Class-Wise Breakdown
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Glioma** | 99.69% | 98.46% | 99.07% |
| **Meningioma** | 97.62% | 99.70% | 98.65% |
| **No Tumor** | 100.00% | 99.50% | 99.75% |
| **Pituitary** | 99.66% | 99.32% | 99.49% |

---

## 🛠️ Project Structure

```text
brain-tumor-classification/
├── app.py                   # The Streamlit Web Application
├── train.py                 # Hybrid Training Pipeline (CNN + XGBoost)
├── requirements.txt         # List of dependencies
├── README.md                # Project Documentation
└── models/                  # Saved Model Artifacts
    ├── best_cnn_model.h5    # The Feature Extractor (MobileNet)
    ├── xgboost_model.json   # The Classifier
    └── scaler.pkl           # Feature Scaler
