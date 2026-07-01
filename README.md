# Anti-Spoof Face Recognition Model

A deep learning-based face anti-spoofing system that distinguishes genuine users from presentation attacks (printed photos, replay attacks, etc.) using transfer learning and liveness detection.

---

## Overview

This project implements a robust Face Anti-Spoofing pipeline built using **DenseNet121 (Transfer Learning)**. Before authentication, the application performs **blink-based liveness detection** to ensure a real user is present. Once liveness is verified, video frames are extracted and passed to the trained model for spoof detection.

The project is designed for real-world deployment with an interactive **Streamlit** frontend.

---

## Features

- Transfer Learning with DenseNet121
- Blink-based liveness detection
- Video capture through Streamlit
- Automatic frame extraction
- Real vs Spoof classification
- Face preprocessing pipeline
- Group-aware train/validation/test splitting
- Weighted sampling for class imbalance
- Hard example mining
- Early stopping
- Cosine Annealing Warm Restarts scheduler
- Exponential Moving Average (EMA) model
- CUDA GPU support

---

## Model Performance

### Dataset

- **14,540 images**
  - Real: **7,146**
  - Fake: **7,394**

Split using **group-aware splitting** to prevent identity leakage across train, validation, and test sets.

| Split | Real | Fake |
|--------|-----:|-----:|
| Train | 5,347 | 5,480 |
| Validation | 1,018 | 1,399 |
| Test | 781 | 515 |

---

### Best Validation Performance

| Metric | Score |
|---------|-------|
| Accuracy | **95.08%** |
| APCER | **4.50%** |
| BPCER | **5.50%** |
| ACER | **5.00%** |
| AUC | **99.02%** |

(Best checkpoint obtained at Epoch 18)

---

### Final Test Performance

| Metric | Score |
|---------|-------|
| Accuracy | **88.97%** |
| APCER | **19.81%** |
| BPCER | **5.25%** |
| ACER | **12.53%** |
| AUC | **97.29%** |

These results were obtained on a completely unseen holdout test set.

---

## Project Workflow

```
User
   │
   ▼
Streamlit Frontend
   │
   ▼
Record Video
   │
   ▼
Blink Detection
   │
   ├── No Blink
   │      ❌ Reject
   │
   └── Blink Detected
           │
           ▼
Extract Frames
           │
           ▼
Face Detection & Preprocessing
           │
           ▼
DenseNet121 Model
           │
           ▼
Real / Spoof Prediction
```

---

## Project Structure

```text
.
├── app.py
├── trainer.py
├── preprocessing.py
├── inference.py
├── models/
├── dataset/
│   ├── real/
│   └── fake/
├── training/
├── logs/
└── streamlit_app.py
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Kr1v/Anti-Spoof-Face-Recognition-Model.git
cd Anti-Spoof-Face-Recognition-Model
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python trainer.py
```

Custom training

```bash
python trainer.py --epochs 60 --batch_size 32 --lr 0.0005
```

---

## Running the Application

Launch the Streamlit interface

```bash
streamlit run app.py
```

### Authentication Pipeline

1. User opens the web application.
2. Webcam records a short video.
3. Blink detection verifies liveness.
4. Frames are extracted from the recorded video.
5. Frames are processed and passed through the trained DenseNet121 model.
6. The application predicts whether the face is **Real** or **Spoof**.

---

## Evaluation Metrics

The model reports:

- APCER (Attack Presentation Classification Error Rate)
- BPCER (Bona Fide Presentation Classification Error Rate)
- ACER (Average Classification Error Rate)
- AUC (Area Under the ROC Curve)
- Accuracy

Lower ACER and higher AUC indicate better anti-spoofing performance.

---

## Tech Stack

- Python
- PyTorch
- DenseNet121 (Transfer Learning)
- OpenCV
- Streamlit
- NumPy
- Scikit-learn
- CUDA

---

## Future Improvements

- Vision Transformers for Anti-Spoofing
- Mobile deployment (TensorFlow Lite)
- Multi-modal liveness detection
- Cross-dataset benchmarking
- ONNX/TensorRT optimization

---



## License

This project is intended for academic and research purposes.
