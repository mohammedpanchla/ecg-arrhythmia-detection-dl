# 🫀 ECG Heartbeat Classification System  
### Deep Learning–Based Cardiac Abnormality Detection using 1D CNN and CNN+LSTM (PyTorch)

---

## 📌 Project Overview

Cardiovascular diseases are the leading cause of death globally, responsible for approximately 17.9 million deaths per year. Early detection of abnormal heart rhythms is critical for preventing severe cardiac events.

This project builds a deep learning–based ECG heartbeat classification system that analyzes raw electrocardiogram (ECG) signals and automatically classifies each heartbeat as normal or abnormal.

Input: ECG heartbeat signal (187 time steps)  
Output: Normal or Abnormal heartbeat classification  

The system uses advanced deep learning architectures including:

• 1D Convolutional Neural Network (CNN)  
• CNN + LSTM Hybrid Model  

This enables accurate and real-time cardiac abnormality detection suitable for clinical decision support and wearable health monitoring.

#### 🧠MODEL :- https://muhammedpanchla-ecg-heartbeat-classifier.hf.space/#

---

## 🎯 Project Objective

The primary goal is to build a deep learning model capable of accurately detecting abnormal heartbeats.

Classification output:

| Class | Meaning |
|---|---|
| 0 | Normal heartbeat |
| 1 | Abnormal heartbeat |

This allows automated detection of cardiac irregularities without manual ECG review.

---

## 🔬 Deep Learning for Time-Series Medical Signals

This project applies deep learning to ECG time-series data to detect cardiac abnormalities.

| Aspect | Value |
|---|---|
| Data type | ECG time-series |
| Task | Binary classification |
| Input size | 187 time steps |
| Output | Normal / Abnormal |
| Domain | Healthcare AI |

---

## 🧠 Architecture Overview

Two architectures were developed and evaluated.

---

### Model 1: 1D CNN Architecture

Designed to extract local waveform patterns such as peaks, slopes, and distortions.

Architecture flow:

Input Signal (187 × 1)  
↓  
Conv1D Layer  
↓  
Batch Normalization  
↓  
ReLU Activation  
↓  
Max Pooling  
↓  
Conv1D Layer  
↓  
Batch Normalization  
↓  
ReLU Activation  
↓  
Global Average Pooling  
↓  
Fully Connected Layer  
↓  
Output Classification  

Purpose: Detect waveform abnormalities.

---

### Model 2: CNN + LSTM Hybrid Architecture (Best Model)

Combines spatial feature extraction with temporal sequence modeling.

Architecture flow:

Input Signal  
↓  
CNN Feature Extraction  
↓  
LSTM Layer (Temporal modeling)  
↓  
Fully Connected Layer  
↓  
Output Classification  

Purpose: Capture both waveform shape and temporal dependencies.

---

## 📊 Dataset

Dataset: PTB Diagnostic ECG Database  
Source: Kaggle  

Dataset characteristics:

| Property | Value |
|---|---|
| Signal length | 187 |
| Classes | 2 |
| Data type | Time-series |
| Format | CSV |

Each sample represents one heartbeat waveform.

---

## ⚠️ Handling Class Imbalance

Medical datasets often suffer from imbalance between normal and abnormal samples.

Solutions applied:

• Balanced dataset creation  
• Equal class representation  
• Prevent model bias  

This improves reliability and real-world performance.

---

## 🔧 Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Epochs | 60 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross Entropy |
| Hardware | GPU / CPU |

---

## 📈 Evaluation Metrics

The following metrics were used for medical-grade evaluation:

| Metric | Purpose |
|---|---|
| Accuracy | Overall performance |
| Precision | False positive control |
| Recall | Abnormal detection strength |
| F1 Score | Balanced performance |
| ROC-AUC | Overall classification quality |

ROC-AUC is the most reliable metric for classification models in healthcare.

---

## 🚀 Machine Learning Pipeline

Complete workflow:

Dataset Loading  
↓  
Data Preprocessing  
↓  
Class Balancing  
↓  
Train/Test Split  
↓  
PyTorch Dataset Creation  
↓  
Model Training (CNN and CNN+LSTM)  
↓  
Model Evaluation  
↓  
Model Comparison  
↓  
Final Model Selection  
↓  
Inference Simulation  

---

## 🔍 Real-World Inference Simulation

The trained model can predict heartbeat condition from a single ECG signal.

Input: ECG signal  
Output: Normal or Abnormal prediction with confidence score  

This simulates real-world clinical deployment.

---

## 📁 Repository Structure

ecg-heartbeat-classification/
│
├── notebooks/
│ └── ECG_Heartbeat_Classification_Final.ipynb
│
├── app/
│ |── app.py/
| └── templates/
|   └── index.html/
|
├── model/
│ └── ecg_model_best.pth
│
├── data/
│ ├── normal.csv
│ └── abnormal.csv
│
├── requirements.txt
│
└── README.md


---

## ⚙️ Technologies Used

Deep Learning  
• PyTorch  

Data Processing  
• NumPy  
• Pandas  

Visualization  
• Matplotlib  
• Seaborn  

Machine Learning  
• Scikit-learn  

---

## 🔬 Technical Highlights

• End-to-end medical AI pipeline  
• Deep learning for ECG signal classification  
• CNN and CNN+LSTM hybrid architecture  
• Time-series modeling  
• Clinical-grade evaluation metrics  
• GPU-supported training  

---

## 🏥 Clinical and Business Impact

| Stakeholder | Benefit |
|---|---|
| Doctors | Faster ECG interpretation |
| Hospitals | Automated screening |
| Patients | Early detection |
| Wearable devices | Real-time monitoring |
| Healthcare AI | Scalable diagnosis |

---

## 🎯 Future Improvements

• Transformer-based models  
• Larger datasets  
• Real-time deployment  
• Mobile integration  
• Multi-class arrhythmia detection  

---

## 👨‍💻 Author

Mohammed Panchla  

Machine Learning Engineer focused on Healthcare AI and Deep Learning Systems.

---

## ⭐ If you found this project useful, consider giving it a star!


