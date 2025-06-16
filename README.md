# 🔬 Leukemia Classification with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

<p align="center">
  <strong>An advanced deep learning solution for automated leukemia detection using state-of-the-art computer vision techniques</strong>
</p>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Structure](#-dataset-structure)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a robust deep learning pipeline for binary classification of leukemia cells from microscopic blood images. Using advanced computer vision techniques and ensemble learning, the system achieves high accuracy in distinguishing between:

- **ALL (Acute Lymphoblastic Leukemia)** cells
- **HEM (Healthy)** cells

The implementation focuses on stability, performance, and real-world applicability with extensive protections against common training issues.

## ✨ Key Features

### 🚀 Advanced Architecture
- **EfficientNet-B3** backbone with custom classification head
- Ensemble learning with k-fold cross-validation
- Mixed precision training with gradient scaling
- Comprehensive gradient explosion prevention

### 🛡️ Stability Mechanisms
- Gradient clipping and normalization
- NaN/Inf detection and handling
- Conservative learning rate scheduling
- Focal loss with label smoothing

### 📊 Data Handling
- Memory-efficient dataset implementation
- Robust augmentation pipeline
- Stratified k-fold validation
- Automatic class weight balancing

### 🔧 Training Features
- OneCycleLR scheduler with warmup
- Early stopping with patience
- Comprehensive logging and monitoring
- CUDA optimization with expandable segments

## 🏗️ Architecture

```
StableLeukemiaClassifier
├── Backbone: EfficientNet-B3 (pretrained)
├── Feature Extraction: Global Average Pooling
└── Classification Head:
    ├── LayerNorm → Linear(1536, 256) → LayerNorm → GELU → Dropout(0.2)
    ├── Linear(256, 128) → LayerNorm → GELU → Dropout(0.1)
    └── Linear(128, 2) [Output]
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/leukemia-classifier.git
cd leukemia-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
tqdm>=4.65.0
numpy>=1.24.0
```

## 📁 Dataset Structure

Organize your dataset as follows:

```
dataset/
└── processado/
    ├── train/
    │   ├── all/       # ALL (leukemia) images
    │   │   ├── img_001.jpg
    │   │   ├── img_002.jpg
    │   │   └── ...
    │   └── hem/       # Healthy cell images
    │       ├── img_001.jpg
    │       ├── img_002.jpg
    │       └── ...
    └── test/
        ├── all/
        └── hem/
```

## 💻 Usage

### Training

Run the training script:

```python
python train.py
```

Or use the Jupyter notebook:

```bash
jupyter notebook cnn_vit7.ipynb
```

### Configuration

Modify the configuration in the main function:

```python
config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 4,
    'num_workers': 2,
    'epochs': 25,
    'patience': 10,
    'n_folds': 3,
    'train_dir': 'dataset/processado/train/',
    'test_dir': 'dataset/processado/test/',
}
```

## 📈 Model Performance

### Training Results (Example)
| Fold | Best Val F1 | Final Val Loss |
|------|-------------|----------------|
| 1    | 0.9550      | 0.0878         |
| 2    | 0.9480      | 0.0974         |
| 3    | 0.9533      | 0.0887         |

**Average Performance**: ~95% F1-Score

### Key Metrics
- **Weighted F1-Score**: Primary evaluation metric
- **Focal Loss**: Handles class imbalance
- **Per-class Precision/Recall**: Available through classification reports

## 🔬 Technical Details

### Data Augmentation
- **Geometric**: RandomRotate90, Flips, ShiftScaleRotate
- **Color**: ColorJitter, RandomBrightnessContrast
- **Conservative parameters** for medical imaging

### Loss Function
```python
StableFocalLoss(
    alpha=class_weights,
    gamma=1.5,
    smoothing=0.05
)
```

### Optimization
- **Optimizer**: AdamW with weight decay
- **Initial LR**: 1e-5 (ultra-conservative)
- **Scheduler**: OneCycleLR with cosine annealing
- **Gradient Clipping**: Max norm = 0.5

### Stability Features
1. **Gradient Management**
   - Aggressive clipping
   - Norm monitoring
   - NaN/Inf detection

2. **Numerical Stability**
   - Input/output clamping
   - Conservative weight initialization
   - Mixed precision with careful scaling

3. **Training Robustness**
   - Batch validation
   - Fallback mechanisms
   - Periodic cache clearing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Follow PEP 8 style guide
2. Add unit tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **timm library** for pretrained models
- **Albumentations** for augmentation pipeline
- Medical imaging community for dataset guidelines
- PyTorch team for the excellent framework

---

<div align="center">
  <p>Made with ❤️ for medical AI research</p>
  <p>
    <a href="#-leukemia-classification-with-deep-learning">Back to top</a>
  </p>
</div>
