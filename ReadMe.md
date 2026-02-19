# 🧠 AI-Based Brain Tumor Detection System

A Deep Learning web application that detects brain tumors from MRI scans using a Convolutional Neural Network (CNN) model deployed with Flask.

This project demonstrates end-to-end AI system development — from model training to web deployment.

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
- HTML5
- Tailwind CSS
- JavaScript
- Glassmorphism UI Design

### 🔹 Backend
- Flask (Python)
- REST API (`/api/analyze`)

### 🔹 Machine Learning
- PyTorch
- Custom Convolutional Neural Network (CNN)
- torchvision for image preprocessing

---

## 🧠 Model Architecture

The model is a custom Convolutional Neural Network consisting of:

- Convolutional layers for feature extraction
- ReLU activation
- MaxPooling layers
- Fully connected layers
- Softmax output for multi-class classification

### ⚙️ Training Details

- Framework: PyTorch
- Input Size: 224 × 224 RGB
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Output Classes: 4

---

## 📊 Model Performance

| Metric              | Value |
|---------------------|--------|
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
````
---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🔬 How It Works

1. User uploads MRI scan.
2. Image is resized and normalized.
3. Model performs forward pass.
4. Softmax probabilities are calculated.
5. Highest probability class is selected.
6. Prediction and confidence score displayed in UI.

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

## 🔮 Future Improvements

* Add class probability bar chart visualization
* Add Grad-CAM for model explainability
* Improve dataset size for better generalization
* Deploy on cloud (AWS / Render / Azure)
* Add user authentication and prediction history

---

## 👨‍💻 Author

Developed as a deep learning deployment project to demonstrate practical AI model integration into a web application.

---
If you found this project useful, consider giving it a star ⭐

```
