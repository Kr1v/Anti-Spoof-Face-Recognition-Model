# Anti-Spoof-Face-Recognition-Model
# CDCN Face Anti-Spoofing

A face anti-spoofing system based on Central Difference Convolutional Networks (CDCN) for distinguishing between genuine and spoofed face images.

## Overview

This project implements a CDCN-based deep learning model for face anti-spoofing using PyTorch. The model learns depth-based representations to differentiate between real face images and presentation attacks such as printed photos or screen replays.

## Features

* CDCN architecture implementation
* Face image preprocessing pipeline
* Train/validation split with stratification
* Data augmentation
* Weighted sampling for class imbalance
* Early stopping
* Cosine learning rate scheduling
* APCER, BPCER, ACER, and AUC evaluation metrics
* GPU acceleration with CUDA support

## Project Structure

```text
.
├── preprocessing_faces.py
├── trainer.py
├── dataset/
│   ├── real/
│   └── fake/
├── models/
└── logs/
```

## Dataset Structure

```text
dataset/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision numpy opencv-python scikit-learn tqdm
```

## Preprocessing

Prepare the dataset before training:

```bash
python preprocessing.py
```

This step extracts and processes face images into the required dataset structure.

## Training

Run training using:

```bash
python trainer.py
```

Custom paths:

```bash
python trainer.py --real_dir dataset/real --fake_dir dataset/fake
```

Custom hyperparameters:

```bash
python trainer.py --epochs 80 --batch_size 16 --lr 0.0005
```

## Evaluation Metrics

The model reports:

* APCER (Attack Presentation Classification Error Rate)
* BPCER (Bona Fide Presentation Classification Error Rate)
* ACER (Average Classification Error Rate)
* AUC (Area Under ROC Curve)

Lower ACER indicates better anti-spoofing performance.

## Output

Training generates:

* Best model checkpoint
* Training logs
* Training history CSV

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Scikit-learn
* CUDA (optional)

## Future Improvements

* Original CDCN++ architecture
* Multi-scale feature fusion
* Cross-dataset evaluation
* ONNX/TensorRT deployment
* Real-time webcam inference

## License

This project is intended for academic and research purposes.
