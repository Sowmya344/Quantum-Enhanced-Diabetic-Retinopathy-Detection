# README: Enhanced VQE + ResNet Hybrid Model for Diabetic Retinopathy Detection

## Overview

This project implements a **Hybrid Quantum-Classical model** using transfer learning (ResNet50 backbone), quantum variational eigensolver (VQE) circuits, and advanced image preprocessing techniques to detect the severity of Diabetic Retinopathy (DR) from retinal fundus images.

- **Preprocessing**: Enhances images using CLAHE for contrast, random augmentation, and normalization.
- **Deep Learning**: Features are extracted using a fully fine-tuned ResNet50 model.
- **Quantum Layer**: Outputs from ResNet50 are fed into a custom, vector-based VQE quantum circuit (via Pennylane), which applies optimized entanglement and rotation gates.
- **Classification**: Quantum circuit outputs are mapped to DR severity classes.
- **Class Imbalance**: Training uses Focal Loss to address data imbalance.
- **Visualization**: Grad-CAM is included to visualize which regions of the retina the model focuses on.

## Folder Structure

- `/data`: Contains the diabetic retinopathy image dataset (must be downloaded from Kaggle).
- `/notebooks`: Colab/Jupyter notebooks containing model code, experiments, and results.
- `/models`: Saved model state dictionaries and weights.
- `/outputs`: Example Grad-CAM visualizations and confusion matrices.

## Main Features

- **Advanced Preprocessing**: CLAHE, cropping, flipping, rotation, color jitter, and normalization.
- **Transfer Learning**: Full fine-tuning of ResNet50 to adapt to retinal images.
- **Quantum Circuit**: VQE circuit with AngleEmbedding, StronglyEntanglingLayers, and custom RX/RZ/CNOT/CZ logic for higher expressivity.
- **Hybrid Model**: Combines classical deep features with quantum vector encoding for robust decision boundaries.
- **Mixed Precision Training**: Uses CUDA amp for faster and memory-efficient training.
- **Performance Analysis**: Training and validation curves, confusion matrix, classification reports.

## Dependencies

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/)
- [Pennylane](https://pennylane.ai/)
- [opencv-python-headless](https://pypi.org/project/opencv-python-headless/)
- [scikit-learn](https://scikit-learn.org/)
- [kagglehub](https://github.com/kagglehub)
- [seaborn](https://seaborn.pydata.org/) (for plots)
- [matplotlib](https://matplotlib.org/) (for visualization)

Install via:
```bash
pip install kagglehub pennylane opencv-python-headless torch torchvision torchaudio scikit-learn seaborn matplotlib
```

## Usage

1. **Download Data**:
   - Authenticate with Kaggle and download the `diabetic-retinopathy-dataset`.
2. **Run Notebook**:
   - Open `VQE and resnet 3.ipynb` in Colab.
   - Run all cells to preprocess data, train the model, and evaluate results.
3. **Model Training**:
   - Training uses mixed precision and Focal Loss.
   - Monitors accuracy, F1-score, and confusion matrix.
   - The best model checkpoint is saved automatically.
4. **Grad-CAM Visualization**:
   - Generates heatmaps to interpret model decisions on fundus images.

## Model Architecture

- **Image** → **CLAHE/Transforms** → **ResNet50 Backbone** → **Vector Extraction** → **Quantum VQE Circuit** → **Classifier** → **DR Class Label**
- Hybrid model output combines quantum circuit expectation values and variances for robust classification across five classes: `healthy`, `mild`, `moderate`, `severe`, `proliferative`.

## Key Classes/Functions

- `CLAHETransform`: Image contrast enhancement.
- `EnhancedVQEHybridModel`: Core hybrid model class.
- `enhanced_vqe_circuit`: Custom quantum module.
- `FocalLoss`: Handles imbalanced class issue.
- `GradCAM`: Helps create visualization overlays.
- `train_model`, `validate`: Training pipeline utilities.

## Results & Evaluation

- Training and validation curves assess loss/accuracy.
- Confusion matrices show prediction breakdown.
- Grad-CAM helps explain model decisions.

## Citation & Authors

**Author**: Sowmya Abirami  
Adapted for medical image analysis with hybrid quantum machine learning.

***

*This README is auto-generated for the Colab notebook "VQE and resnet 3.ipynb" (Hybrid quantum-classical model for diabetic retinopathy classification with explainable AI visualization).*

[1](https://colab.research.google.com/drive/1AYtSu0GsVYu1Z0MUnJ2uWErCxJ_2gUb3)
