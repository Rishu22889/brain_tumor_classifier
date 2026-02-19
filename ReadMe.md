# 🧠 AI-Based Brain Tumor Detection System

A Deep Learning web application that detects brain tumors from MRI scans using a ResNet18-based Convolutional Neural Network (CNN) deployed with Flask.

This project demonstrates end-to-end AI system development — from model training and evaluation to web deployment.

---

## 🚀 Project Overview

Brain tumors require early and accurate detection. This project builds an AI-based classification system that analyzes MRI scans and predicts one of the following categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The trained PyTorch model is integrated into a modern glass-themed web interface where users can upload MRI images and receive predictions along with confidence scores.

---

## 🏗 Tech Stack

### 🔹 Frontend

* HTML5
* Tailwind CSS
* JavaScript
* Glassmorphism UI Design

### 🔹 Backend

* Flask (Python)
* REST API (`/api/analyze`)

### 🔹 Machine Learning

* PyTorch
* Transfer Learning (ResNet18 pretrained on ImageNet)
* torchvision for image preprocessing

---

## 🧠 Model Architecture

Instead of training from scratch, this system uses **ResNet18 pretrained on ImageNet** and fine-tunes the final layers for MRI classification.

### Architecture Details

* Backbone: ResNet18 (Pretrained)
* Final Layer: Modified Fully Connected layer (4 output classes)
* Input Size: 224 × 224 RGB
* Activation: ReLU
* Output: Softmax (via CrossEntropyLoss)

### Training Configuration

* Framework: PyTorch
* Loss Function: CrossEntropyLoss
* Optimizer: Adam (lr = 0.0003)
* Weight Decay: 1e-4
* Epochs: 10
* Batch Size: 32
* Dataset Size: ~4000 training images

---

## 📊 Model Performance

### 🔹 Overall Accuracy

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 99.84% |
| Validation Accuracy | 98.97% |

> Note: Performance may vary depending on dataset split and training configuration.

---

## 📂 Project Structure

```

brain-tumor-detection/
│
├── app.py                # Flask backend
├── model.pth             # Trained PyTorch model
├── utils.py              # Image preprocessing & prediction logic
├── templates/
│   └── index.html        # Frontend UI
├── static/
│   ├── images/
│   └── css/
├── requirements.txt
└── README.md

---

## 📊 Model Evaluation Visualizations

### 🔹 Confusion Matrix (Validation Set)

![Confusion Matrix](static/images/output.png)

The confusion matrix demonstrates strong class separation with minimal misclassification across all four categories:

* Glioma
* Meningioma
* No Tumor
* Pituitary Tumor

---

### 🔹 Performance Summary

| Class      | Precision | Recall | F1 Score |
| ---------- | --------- | ------ | -------- |
| Glioma     | 1.0000    | 1.0000 | 1.0000   |
| Meningioma | 0.9959    | 0.9796 | 0.9877   |
| No Tumor   | 0.9965    | 1.0000 | 0.9982   |
| Pituitary  | 0.9864    | 0.9966 | 0.9915   |

These results indicate high model reliability with very low false positives and false negatives on the validation dataset.

---

### Key Observations

* High precision across all classes indicates low false positives.
* High recall ensures tumor cases are rarely missed.
* Only 1 tumor case was misclassified as no-tumor.
* Minimal inter-class confusion between tumor subtypes.

---

## 🔬 How It Works

1. User uploads MRI scan.
2. Image is resized to 224×224 and normalized.
3. Model performs forward pass.
4. Softmax probabilities are computed.
5. Highest probability class is selected.
6. Prediction and confidence score are displayed.

---

## ⚠️ Model Limitations

* Trained on a single curated dataset.
* External validation on multi-hospital data not performed.
* Real-world performance may vary under domain shift.
* Not optimized for clinical deployment.

---

## 🌐 API Endpoint

### POST `/api/analyze`

**Request:**
Form-data containing MRI image file.

**Response:**

```json
{
  "prediction": "Glioma",
  "confidence": 0.9997
}
```

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

It is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for clinical decisions.

---

## 👨‍💻 Author

Developed as a deep learning deployment project demonstrating transfer learning, evaluation metrics, and production-ready Flask integration.

---
If you found this project useful, consider giving it a star ⭐

---
