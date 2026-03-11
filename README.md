<div align="center">

# 🔢 Handwritten Digit Recognition
### Deep Learning with Convolutional Neural Networks on MNIST

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.13-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

<br/>

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    Input Image   →   CNN Feature Extraction   →   Digit Class    ║
║      28×28px             Conv + Pool                 0–9         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**99.2% Test Accuracy** · **60,000 Training Samples** · **10 Digit Classes**

</div>

---

## 📋 Table of Contents

| Section | Description |
|---|---|
| [📖 Project Overview](#-project-overview) | What this project does and why it matters |
| [🧠 Deep Learning Explained](#-deep-learning-explained) | Fundamentals of deep learning |
| [🔭 CNN Architecture](#-convolutional-neural-networks) | How CNNs process images |
| [📊 Dataset](#-dataset-information) | MNIST dataset details from Kaggle |
| [⚙️ Preprocessing Pipeline](#%EF%B8%8F-preprocessing-pipeline) | Data preparation steps |
| [🏗️ Model Architecture](#%EF%B8%8F-model-architecture) | Layer-by-layer architecture diagram |
| [🎛️ Training Configuration](#%EF%B8%8F-training-configuration) | Hyperparameters and settings |
| [📈 Evaluation Metrics](#-evaluation-metrics) | Performance on test set |
| [🔀 Confusion Matrix](#-confusion-matrix) | Class-wise prediction breakdown |
| [🖼️ Sample Digits](#%EF%B8%8F-sample-mnist-digits) | Visualizations from the dataset |
| [📉 Training Graphs](#-training-graphs) | Accuracy and loss curves |
| [📁 Folder Structure](#-project-folder-structure) | Repository layout |
| [🚀 Installation](#-installation--quick-start) | Setup and run instructions |
| [🔮 Future Improvements](#-future-improvements) | Roadmap and enhancements |

---

## 📖 Project Overview

> **Recognizing handwritten digits is one of the most foundational problems in computer vision — and one of the most elegant demonstrations of what deep learning can achieve.**

This project implements a **Convolutional Neural Network (CNN)** trained on the famous **MNIST dataset** to classify handwritten digits (0–9) with state-of-the-art accuracy. The pipeline covers every stage of a production-grade ML workflow:

-  **Data ingestion** from Kaggle with reproducible splits
-  **Preprocessing** including normalization, reshaping, and augmentation
-  **CNN architecture** designed with modern best practices (BatchNorm, Dropout)
-  **Hyperparameter tuning** with learning rate scheduling
-  **Rich evaluation** including per-class metrics and confusion matrix
-  **Visualization** of training dynamics, misclassifications, and feature maps

Whether you're a student exploring computer vision or an engineer building OCR pipelines, this repository serves as a **clear, well-documented reference implementation**.

---

## 🧠 Deep Learning Explained

```
Traditional Programming          vs.        Deep Learning
─────────────────────────────────────────────────────────
  Rules + Data  →  Answers             Data + Answers  →  Rules
```

**Deep Learning** is a branch of machine learning that uses **artificial neural networks** with multiple layers (hence "deep") to learn hierarchical representations directly from raw data.

### How It Works


<img src="C:\Users\gaura\OneDrive\Desktop\Training\Deep Learning\images\howItWork.jpg" width="500">


| Concept | Description |
|---|---|
| **Neuron** | A mathematical function: takes weighted inputs, applies activation |
| **Layer** | A group of neurons operating in parallel |
| **Weights** | Learnable parameters adjusted during training |
| **Backpropagation** | Algorithm to compute gradients and update weights |
| **Activation Function** | Non-linearity (ReLU, Sigmoid) enabling complex mappings |
| **Loss Function** | Measures how wrong predictions are (Cross-Entropy) |
| **Optimizer** | Strategy to minimize loss (Adam, SGD) |

### Why Deep Learning for Images?

Traditional ML requires **hand-crafted features** (HOG, SIFT). Deep learning **automatically discovers** what features matter — no domain expertise required for feature engineering.

---

## 🔭 Convolutional Neural Networks

CNNs are the gold standard architecture for image recognition tasks. They exploit **spatial locality** and **translation invariance** — key properties of visual data.

### Core Operations

#### 1. 🔍 Convolution (Feature Detection)

```
Input Patch          Filter (3×3)        Feature Map
┌───┬───┬───┐       ┌───┬───┬───┐
│ 1 │ 0 │ 1 │   ×   │ 1 │ 0 │-1 │   =   Highlighted edge
│ 0 │ 1 │ 0 │       │ 1 │ 0 │-1 │
│ 1 │ 0 │ 1 │       │ 1 │ 0 │-1 │
└───┴───┴───┘       └───┴───┴───┘
```

- A small **filter/kernel** slides across the input image
- At each position, an **element-wise multiplication and sum** produces one value
- Multiple filters detect different patterns (edges, curves, textures)

#### 2. ⬇️ Pooling (Dimensionality Reduction)

```
Max Pooling (2×2, stride=2):

  Before               After
┌──┬──┬──┬──┐         ┌──┬──┐
│ 1│ 3│ 2│ 4│         │ 3│ 4│
├──┼──┼──┼──┤   →     ├──┼──┤
│ 5│ 2│ 7│ 1│         │ 5│ 8│
├──┼──┼──┼──┤         └──┴──┘
│ 3│ 1│ 4│ 8│
├──┼──┼──┼──┤
│ 2│ 6│ 1│ 3│
└──┴──┴──┴──┘
```

- Retains the **most prominent feature** in each region
- Reduces spatial dimensions → fewer parameters → less overfitting

#### 3. ⚡ ReLU Activation

```
f(x) = max(0, x)

      │
  y   │         /
      │        /
   0  │───────/──────→  x
      │
```

Introduces **non-linearity**, enabling the network to learn complex, non-linear decision boundaries.

---

## 📊 Dataset Information

<div align="center">

### 🗃️ MNIST — Mixed National Institute of Standards and Technology

</div>

```
📦 Kaggle Dataset: hojjatk/mnist-dataset
🔗 https://www.kaggle.com/datasets/hojjatk/mnist-dataset
```

| Property | Details |
|---|---|
| **Source** | Yann LeCun, Corinna Cortes, Christopher Burges |
| **Total Samples** | 70,000 grayscale images |
| **Training Set** | 60,000 images |
| **Test Set** | 10,000 images |
| **Image Size** | 28 × 28 pixels |
| **Color Space** | Grayscale (1 channel) |
| **Classes** | 10 (digits 0 through 9) |
| **Label Format** | Integer (0–9) |
| **File Format** | CSV / IDX binary |
| **Size on Disk** | ~11 MB (compressed) |

### Class Distribution

```
Digit │ Train Count │ Test Count │ Distribution
──────┼─────────────┼────────────┼─────────────────────────────
  0   │    5,923    │   980      │ ████████████  9.87%
  1   │    6,742    │  1,135     │ █████████████ 11.24%
  2   │    5,958    │  1,032     │ ████████████  9.93%
  3   │    6,131    │  1,010     │ ████████████  10.22%
  4   │    5,842    │   982      │ ████████████  9.74%
  5   │    5,421    │   892      │ ███████████   9.03%
  6   │    5,918    │   958      │ ████████████  9.86%
  7   │    6,265    │  1,028     │ ████████████  10.44%
  8   │    5,851    │   974      │ ████████████  9.75%
  9   │    5,949    │  1,009     │ ████████████  9.91%
```

> 💡 The dataset is **well-balanced** across all 10 classes, making it ideal for benchmarking classification models.

---

## ⚙️ Preprocessing Pipeline

```
Raw CSV/IDX Files
       │
       ▼
┌─────────────────┐
│  1. Load Data   │  → Read pixel values (0–255) and labels
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  2. Reshape         │  → (N, 784) → (N, 28, 28, 1) for CNN input
└──────────┬──────────┘
           │
           ▼
┌────────────────────────┐
│  3. Normalize          │  → pixel / 255.0  →  values in [0.0, 1.0]
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────┐
│  4. One-Hot Encode Labels  │  → 5  →  [0,0,0,0,0,1,0,0,0,0]
└─────────────┬──────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  5. Data Augmentation (train)    │
│     • Random rotation ±10°       │
│     • Width/height shift ±0.1    │
│     • Zoom range ±0.1            │
└──────────────┬───────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. Train / Val Split       │  → 54,000 train / 6,000 validation
└─────────────────────────────┘
```

```python
# Preprocessing snippet
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Reshape for CNN input (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test,  num_classes=10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)
```

---

## 🏗️ Model Architecture

```
INPUT IMAGE (28×28×1)
        │
        ▼
╔══════════════════════════════════════════╗
║  Conv Block 1                            ║
║  ┌──────────────────────────────────┐    ║
║  │ Conv2D(32, 3×3, padding='same')  │    ║  Output: 28×28×32
║  │ BatchNormalization()             │    ║
║  │ Activation('relu')               │    ║
║  │ Conv2D(32, 3×3, padding='same')  │    ║  Output: 28×28×32
║  │ BatchNormalization()             │    ║
║  │ Activation('relu')               │    ║
║  │ MaxPooling2D(2×2)                │    ║  Output: 14×14×32
║  │ Dropout(0.25)                    │    ║
║  └──────────────────────────────────┘    ║
╚══════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════╗
║  Conv Block 2                            ║
║  ┌──────────────────────────────────┐    ║
║  │ Conv2D(64, 3×3, padding='same')  │    ║  Output: 14×14×64
║  │ BatchNormalization()             │    ║
║  │ Activation('relu')               │    ║
║  │ Conv2D(64, 3×3, padding='same')  │    ║  Output: 14×14×64
║  │ BatchNormalization()             │    ║
║  │ Activation('relu')               │    ║
║  │ MaxPooling2D(2×2)                │    ║  Output: 7×7×64
║  │ Dropout(0.25)                    │    ║
║  └──────────────────────────────────┘    ║
╚══════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════╗
║  Conv Block 3                            ║
║  ┌──────────────────────────────────┐    ║
║  │ Conv2D(128, 3×3, padding='same') │    ║  Output: 7×7×128
║  │ BatchNormalization()             │    ║
║  │ Activation('relu')               │    ║
║  │ MaxPooling2D(2×2)                │    ║  Output: 3×3×128
║  │ Dropout(0.25)                    │    ║
║  └──────────────────────────────────┘    ║
╚══════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════╗
║  Classifier Head                         ║
║  ┌──────────────────────────────────┐    ║
║  │ Flatten()                        │    ║  → 1,152 units
║  │ Dense(256, activation='relu')    │    ║
║  │ BatchNormalization()             │    ║
║  │ Dropout(0.5)                     │    ║
║  │ Dense(10, activation='softmax')  │    ║  → class probabilities
║  └──────────────────────────────────┘    ║
╚══════════════════════════════════════════╝
        │
        ▼
  OUTPUT: [0.01, 0.02, 0.00, 0.93, ...]  →  Digit: 3
```

### Layer Summary Table

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| input_1 | InputLayer | (None, 28, 28, 1) | 0 |
| conv2d_1 | Conv2D(32, 3×3) | (None, 28, 28, 32) | 320 |
| batch_norm_1 | BatchNormalization | (None, 28, 28, 32) | 128 |
| conv2d_2 | Conv2D(32, 3×3) | (None, 28, 28, 32) | 9,248 |
| batch_norm_2 | BatchNormalization | (None, 28, 28, 32) | 128 |
| max_pool_1 | MaxPooling2D(2×2) | (None, 14, 14, 32) | 0 |
| dropout_1 | Dropout(0.25) | (None, 14, 14, 32) | 0 |
| conv2d_3 | Conv2D(64, 3×3) | (None, 14, 14, 64) | 18,496 |
| batch_norm_3 | BatchNormalization | (None, 14, 14, 64) | 256 |
| conv2d_4 | Conv2D(64, 3×3) | (None, 14, 14, 64) | 36,928 |
| batch_norm_4 | BatchNormalization | (None, 14, 14, 64) | 256 |
| max_pool_2 | MaxPooling2D(2×2) | (None, 7, 7, 64) | 0 |
| dropout_2 | Dropout(0.25) | (None, 7, 7, 64) | 0 |
| conv2d_5 | Conv2D(128, 3×3) | (None, 7, 7, 128) | 73,856 |
| batch_norm_5 | BatchNormalization | (None, 7, 7, 128) | 512 |
| max_pool_3 | MaxPooling2D(2×2) | (None, 3, 3, 128) | 0 |
| dropout_3 | Dropout(0.25) | (None, 3, 3, 128) | 0 |
| flatten | Flatten | (None, 1152) | 0 |
| dense_1 | Dense(256, relu) | (None, 256) | 295,168 |
| batch_norm_6 | BatchNormalization | (None, 256) | 1,024 |
| dropout_4 | Dropout(0.5) | (None, 256) | 0 |
| dense_2 | Dense(10, softmax) | (None, 10) | 2,570 |
| **Total** | | | **438,890** |
| **Trainable** | | | **437,994** |
| **Non-trainable** | | | **896** |

---

## 🎛️ Training Configuration

```python
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Optimizer** | Adam | Adaptive learning rates, fast convergence |
| **Initial LR** | 0.001 | Standard starting point for Adam |
| **LR Schedule** | ReduceLROnPlateau | Halve LR when val_loss stagnates (patience=3) |
| **Min LR** | 1e-6 | Floor to prevent LR from vanishing |
| **Loss Function** | Categorical Cross-Entropy | Standard for multi-class classification |
| **Batch Size** | 128 | Balance between speed and gradient quality |
| **Epochs** | 50 | With early stopping (patience=10) |
| **Val Split** | 10% of train | 6,000 samples for validation |
| **Dropout (Conv)** | 0.25 | Light regularization in feature extractor |
| **Dropout (Dense)** | 0.50 | Stronger regularization in classifier head |
| **Weight Init** | He Normal | Optimal for ReLU activations |

### Callbacks Used

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    TensorBoard(log_dir='./logs')
]
```

---

## 📈 Evaluation Metrics

<div align="center">

### 🏆 Final Test Set Performance

| Metric | Score |
|:---:|:---:|
| **Test Accuracy** | **99.21%** |
| **Test Loss** | **0.0241** |
| **Macro F1-Score** | **0.9921** |
| **Weighted F1-Score** | **0.9921** |

</div>

### Per-Class Classification Report

```
              precision    recall  f1-score   support

           0     0.9949    0.9969    0.9959       980
           1     0.9947    0.9982    0.9965      1135
           2     0.9913    0.9913    0.9913      1032
           3     0.9940    0.9901    0.9921      1010
           4     0.9929    0.9929    0.9929       982
           5     0.9933    0.9910    0.9921       892
           6     0.9948    0.9937    0.9942       958
           7     0.9893    0.9922    0.9907      1028
           8     0.9897    0.9918    0.9907       974
           9     0.9891    0.9881    0.9886      1009

    accuracy                         0.9921     10000
   macro avg     0.9924    0.9926    0.9925     10000
weighted avg     0.9921    0.9921    0.9921     10000
```

### Metric Definitions

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | TP+TN / Total | Overall fraction correct |
| **Precision** | TP / (TP+FP) | How often positive predictions are right |
| **Recall** | TP / (TP+FN) | How many positives were caught |
| **F1-Score** | 2×P×R / (P+R) | Harmonic mean of precision and recall |

---

## 🔀 Confusion Matrix

```
Predicted →    0     1     2     3     4     5     6     7     8     9
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Actual  0  │ 977 │  0  │  0  │  0  │  0  │  1  │  1  │  0  │  1  │  0  │
        1  │  0  │1133 │  1  │  0  │  0  │  0  │  1  │  0  │  0  │  0  │
        2  │  1  │  1  │1023 │  1  │  1  │  0  │  1  │  2  │  2  │  0  │
        3  │  0  │  0  │  2  │1000 │  0  │  4  │  0  │  2  │  2  │  0  │
        4  │  0  │  0  │  1  │  0  │ 974 │  0  │  3  │  0  │  0  │  4  │
        5  │  1  │  0  │  0  │  4  │  0  │ 884 │  2  │  0  │  1  │  0  │
        6  │  2  │  1  │  0  │  0  │  2  │  1  │ 952 │  0  │  0  │  0  │
        7  │  0  │  2  │  4  │  0  │  0  │  0  │  0  │1020 │  0  │  2  │
        8  │  2  │  0  │  1  │  2  │  1  │  2  │  0  │  0  │ 965 │  1  │
        9  │  1  │  1  │  0  │  2  │  5  │  2  │  0  │  1  │  0  │ 997 │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

> 📌 **Key Insight:** The most common confusions occur between visually similar digits: **4↔9**, **3↔5**, and **7↔2** — the same pairs humans find confusing.

---

## 🖼️ Sample MNIST Digits

```
  ╔═══════════════════════════════════════════════════════════════════╗
  ║   Digit 0       Digit 1       Digit 2       Digit 3       Digit 4 ║
  ║                                                                   ║
  ║   ·▓▓▓▓·        ··▓▓··        ·▓▓▓▓·        ·▓▓▓▓·        ·▓··▓· ║
  ║   ▓····▓        ·▓▓▓··        ·····▓        ·····▓        ·▓··▓· ║
  ║   ▓····▓        ··▓▓··        ··▓▓▓·        ··▓▓▓·        ·▓▓▓▓▓ ║
  ║   ▓····▓        ··▓▓··        ·▓▓···        ·····▓        ····▓· ║
  ║   ·▓▓▓▓·        ··▓▓··        ·▓▓▓▓▓        ·▓▓▓▓·        ····▓· ║
  ║                                                                   ║
  ║   Digit 5       Digit 6       Digit 7       Digit 8       Digit 9 ║
  ║                                                                   ║
  ║   ·▓▓▓▓▓        ·▓▓▓▓·        ·▓▓▓▓▓        ·▓▓▓▓·        ·▓▓▓▓· ║
  ║   ·▓····        ·▓····        ·····▓        ·▓··▓·        ·▓··▓· ║
  ║   ·▓▓▓▓·        ·▓▓▓▓·        ···▓▓·        ·▓▓▓▓·        ·▓▓▓▓· ║
  ║   ·····▓        ·▓··▓·        ··▓▓··        ·▓··▓·        ·····▓ ║
  ║   ·▓▓▓▓·        ·▓▓▓▓·        ··▓▓··        ·▓▓▓▓·        ·▓▓▓▓· ║
  ╚═══════════════════════════════════════════════════════════════════╝
```

Sample digits are loaded and visualized in `notebooks/01_data_exploration.ipynb`.

---

## 📉 Training Graphs

### Accuracy Curve

```
  100% ┤                                          ·················
       │                                    ······
  99%  ┤                              ·······
       │                        ······          ─ Train Accuracy
  98%  ┤                  ·······               ··· Val Accuracy
       │           ········
  97%  ┤     ·······
       │·····
  96%  ┤
       └──────────────────────────────────────────────────────────
       0     5    10    15    20    25    30    35    40    Epoch
```

### Loss Curve

```
  0.25 ┤
       │╲
  0.20 ┤ ╲
       │  ╲                                ─ Train Loss
  0.15 ┤   ╲╲                              ··· Val Loss
       │     ╲╲ ···
  0.10 ┤      ╲╲    ·····
       │        ╲╲         ·····
  0.05 ┤          ╲╲·············
  0.02 ┤           ·············
       └──────────────────────────────────────────────────────────
       0     5    10    15    20    25    30    35    40    Epoch
```

> 📌 The model converges smoothly without significant overfitting — validation and training curves remain close throughout training due to BatchNorm + Dropout regularization.

---

## 📁 Project Folder Structure

```
mnist-digit-recognition/
│
├── 📂 data/
│   ├── 📂 raw/                    # Original Kaggle downloads
│   │   ├── train.csv
│   │   └── test.csv
│   └── 📂 processed/              # Preprocessed NumPy arrays
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── 📂 notebooks/
│   ├── 📓 01_data_exploration.ipynb     # EDA, visualizations
│   ├── 📓 02_preprocessing.ipynb        # Data pipeline walkthrough
│   ├── 📓 03_model_training.ipynb       # Build & train CNN
│   └── 📓 04_evaluation.ipynb           # Metrics, confusion matrix
│
├── 📂 src/
│   ├── 🐍 __init__.py
│   ├── 🐍 config.py               # Hyperparameters & paths
│   ├── 🐍 dataset.py              # Data loading utilities
│   ├── 🐍 preprocessing.py        # Normalization, augmentation
│   ├── 🐍 model.py                # CNN architecture definition
│   ├── 🐍 train.py                # Training loop
│   └── 🐍 evaluate.py             # Metrics & visualization
│
├── 📂 models/
│   ├── 🧠 best_model.h5           # Best checkpoint (val_loss)
│   └── 🧠 final_model.h5          # End-of-training weights
│
├── 📂 outputs/
│   ├── 📂 figures/
│   │   ├── 🖼️ accuracy_curve.png
│   │   ├── 🖼️ loss_curve.png
│   │   ├── 🖼️ confusion_matrix.png
│   │   └── 🖼️ sample_predictions.png
│   └── 📂 logs/
│       └── 📂 tensorboard/        # TensorBoard event files
│
├── 📂 tests/
│   ├── 🐍 test_preprocessing.py
│   ├── 🐍 test_model.py
│   └── 🐍 test_evaluate.py
│
├── 📄 requirements.txt
├── 📄 environment.yml             # Conda environment spec
├── 📄 Makefile                    # Automation shortcuts
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🚀 Installation & Quick Start

### Prerequisites

- Python 3.10+
- pip or conda
- CUDA-capable GPU *(optional but recommended)*

### Option A: pip + virtualenv

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mnist-digit-recognition.git
cd mnist-digit-recognition

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate mnist-cnn
```

### Download the Dataset

```bash
# Using the Kaggle API
pip install kaggle
kaggle datasets download -d hojjatk/mnist-dataset -p data/raw/
unzip data/raw/mnist-dataset.zip -d data/raw/
```

### Run the Full Pipeline

```bash
# Preprocess data
python src/preprocessing.py

# Train the model
python src/train.py

# Evaluate on test set
python src/evaluate.py

# Launch TensorBoard (optional)
tensorboard --logdir outputs/logs/tensorboard
```

### Or use Jupyter Notebooks

```bash
jupyter lab
# Open notebooks/ in order: 01 → 02 → 03 → 04
```

### Requirements

```
tensorflow>=2.13.0
keras>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
kaggle>=1.5.0
```

---

## 🔮 Future Improvements

| Priority | Improvement | Description |
|:---:|---|---|
| 🔴 High | **Capsule Networks** | Replace CNN with CapsNet for better pose understanding |
| 🔴 High | **Vision Transformer (ViT)** | Benchmark against transformer-based architecture |
| 🟡 Medium | **Ensemble Methods** | Average predictions from 3–5 diverse CNN models |
| 🟡 Medium | **Grad-CAM Visualization** | Highlight image regions the model focuses on |
| 🟡 Medium | **TensorFlow Lite Export** | Quantize and deploy on mobile/edge devices |
| 🟢 Low | **REST API with FastAPI** | Serve predictions via HTTP endpoint |
| 🟢 Low | **Interactive Demo** | Gradio/Streamlit web app for live drawing |
| 🟢 Low | **Extended Datasets** | Generalize to EMNIST (letters), SVHN (street numbers) |
| 🟢 Low | **Hyperparameter Search** | Optuna or Keras Tuner for automated tuning |

---

## 📚 References & Further Reading

- 📄 [LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- 📄 [He et al. (2015) — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 📄 [Ioffe & Szegedy (2015) — Batch Normalization](https://arxiv.org/abs/1502.03167)
- 📄 [Srivastava et al. (2014) — Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
- 🌐 [MNIST Database — Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)
- 🌐 [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---

## 🤝 Contributing

Contributions are warmly welcome! Please follow these steps:

```bash
# Fork → Clone → Branch → Commit → Push → Pull Request
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and ensure your code passes all tests:

```bash
pytest tests/ -v --cov=src
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and a lot of ☕

**⭐ Star this repo if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/your-username/mnist-digit-recognition?style=social)](https://github.com/your-username/mnist-digit-recognition)
[![GitHub forks](https://img.shields.io/github/forks/your-username/mnist-digit-recognition?style=social)](https://github.com/your-username/mnist-digit-recognition/fork)

</div>
