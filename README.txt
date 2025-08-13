# 🩺 Skin Cancer Detection System

## 📖 Overview
Skin cancer is one of the most common forms of cancer worldwide, with early detection being critical for effective treatment and survival rates. This project presents a **Machine Learning-based Skin Cancer Detection System** capable of classifying skin images into:

- **Benign** (non-cancerous lesion)  
- **Malignant** (cancerous lesion)  
- **Healthy Skin**  
- **Not a Skin Image**  

The system integrates **image preprocessing**, **feature extraction**, and **predictive modeling** into a streamlined workflow, delivered via a **Streamlit web application**.

---

## 🎯 Objectives
- Develop a **fast, lightweight, and accurate** skin cancer detection model.
- Provide a **user-friendly** interface for uploading and classifying images.
- Support **real-time inference** using pre-trained ML models.
- Assist **medical professionals and researchers** with an AI-based decision-support tool.

---

## 🧠 Technical Approach
### 1. Image Preprocessing
- **Grayscale Conversion**: Reduces computational complexity while preserving texture details.
- **Resizing**: Standardized to `100x100` pixels for model compatibility.
- **Flattening**: Converts image matrix into a single feature vector.

### 2. Feature Extraction
- Uses raw pixel intensity values as features for classification.
- Maintains shape consistency with `reshape(1, -1)` to fit model input format.

### 3. Model Training
- Conducted in `Prediction.ipynb`.
- Multiple algorithms tested; the best-performing model is saved as `best_model.pkl`.
- Trained using **scikit-learn** with optimal hyperparameters.

### 4. Classification Labels
| Label Index | Class Name        |
|-------------|-------------------|
| 0           | Benign            |
| 1           | Malignant         |
| 2           | Healthy Skin      |
| 3           | Not a Skin Image  |

### 5. Deployment
- **Streamlit** provides an intuitive and interactive frontend.
- **Joblib** is used for model serialization and loading.
- Predictions are generated instantly upon image upload.

---

## 📂 Repository Structure
```
.
├── gui.py              # Streamlit application
├── Prediction.ipynb    # Model training and evaluation notebook
├── best_model.pkl      # Serialized trained ML model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 Installation & Usage
### Prerequisites
- **Python 3.8+**
- Pip package manager

### Steps
```bash
# Clone the repository
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run gui.py
```

---

## 📊 Example Output
**Input Image:**  
<img src="example.jpg" width="200">

**Model Prediction:**  
```
Prediction: Malignant
```

---

## 💡 Potential Use Cases
- **Clinical Decision Support**: Assists dermatologists in initial screening.
- **Educational Tool**: Demonstrates the application of ML in medical imaging.
- **Research**: Serves as a baseline model for academic projects.

---
## 📦 requirements
```
streamlit
numpy
opencv-python
Pillow
joblib
scikit-learn
```
