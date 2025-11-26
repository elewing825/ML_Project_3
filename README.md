# ML_Project_3: Razorback Logo Image Classification

## Project Overview

This project trains convolutional neural networks (CNNs) to classify images as containing the official University of Arkansas Razorback logo or not. Two models were implemented: a **Baseline CNN** designed to demonstrate overfitting, and an **Improved CNN** designed to generalize effectively on a very small dataset. The improved version integrates stronger regularization, deeper feature extraction, adaptive pooling, and early stopping to achieve significantly better performance.

The dataset includes images collected from Etsy listings, Facebook Marketplace, and official Razorback branding.

---

## Dataset

Images are stored using the `ImageFolder` directory layout:

* `images/pigs` — Razorback logo images (label = 1)
* `images/not pigs` — Non-Razorback images (label = 0)

Total images: **126**

A reproducible 80/20 split is created using a seeded random permutation:

* **100 training images**
* **26 validation images**

---

## Configuration

All hyperparameters and training settings are defined in a centralized configuration block:

* `IMAGE_SIZE = (500, 500)`
* `BATCH_SIZE = 8`
* `NUM_EPOCHS = 20`
* `LEARNING_RATE = 1e-3`
* `WEIGHT_DECAY = 1e-4`
* `DROPOUT_RATE = 0.5`
* `SEED = 666`

A custom `set_seed()` function ensures reproducible results across Python, NumPy, and PyTorch.

---

## Image Preprocessing

Training and validation transforms are defined separately:

### **Training Transformations**

```python
transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
```

### **Validation Transformations**

```python
transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])
```

This ensures that data augmentation is applied only to training images, while validation remains deterministic.

---

## Model Architectures

### **1. BaselineCNN (Overfitting Demonstration)**

A minimal CNN intentionally designed to overfit:

* 2 convolutional layers (3→16→32)
* 2 max pooling layers
* **Massive fully connected layer:** 500,000 → 64 → 1
* **32,005,217 total parameters**
* No regularization

---

### **2. ImprovedCNN (Generalizing Model)**

A deeper, regularized architecture:

* 4 convolutional blocks with:
  Conv → BatchNorm → ReLU → MaxPool
  Channels: **32, 64, 128, 256**
* AdaptiveAvgPool2d((4,4))
* Classifier: **4096 → 256 → 1**
* Dropout (p=0.5)
* Weight decay in optimizer
* **1,438,465 total parameters** (95% fewer than baseline)

---

## Training Methodology

Both models share the following training setup:

* **Optimizer:** Adam (LR = 1e-3, weight decay = 1e-4)
* **Loss Function:** BCEWithLogitsLoss
* **Batch Size:** 8
* **Device:** CPU or CUDA
* **Transforms:** Augmented train set, clean validation set
* **Metrics:** Accuracy, loss curves, confusion matrix, ROC curve
* **Model Saving:** Only best validation-performing model is checkpointed

The improved model additionally uses:

* **Early stopping** (patience = 10)
* **Adaptive threshold selection by F1-score**

---

## Results

### **Model 1: Baseline CNN**

**Final Metrics:**

* **Train Accuracy:** 0.8200
* **Validation Accuracy:** 0.5000
* **Generalization Gap:** +0.3200 (severe overfitting)

**Behavior:**

* Validation accuracy fluctuated from **38% → 80%** depending on epoch
* Model memorizes training data due to enormous FC layer and lack of regularization
* Ultimately failed to generalize

---

### **Model 2: Improved CNN**

Training with early stopping produced stable convergence and strong generalization:

**Final Metrics:**

* **Train Accuracy:** 0.8300
* **Validation Accuracy:** 0.8462
* **Generalization Gap:** –0.0162
* **Best Validation Accuracy:** 0.8462 (Epoch 10)

**Behavior:**

* Validation accuracy consistently in the **73–85%** range
* Slightly higher validation vs training accuracy due to:

  * Strong data augmentation (harder train samples)
  * Small validation set (26 images)
* Overall, model generalizes extremely well for the dataset size

---

## Final Model Comparison

### **BaselineCNN Summary**

* 32 million parameters
* No regularization
* FC layer dominates capacity
* **Overfits heavily**
* Poor generalization (50% final validation accuracy)

### **ImprovedCNN Summary**

* 1.4 million parameters
* BatchNorm + Dropout + Weight Decay
* Adaptive pooling reduces FC size
* **Generalizes correctly**
* Achieves **84.62% validation accuracy**

---

## Key Improvements

| Aspect                 | Baseline      | Improved               | Impact                                      |
| ---------------------- | ------------- | ---------------------- | ------------------------------------------- |
| **Total Parameters**   | 32M           | 1.4M                   | Huge reduction prevents memorization        |
| **FC Layer Size**      | 500k → 64     | 4096 → 256             | Controlled capacity improves generalization |
| **Depth**              | 2 conv layers | 4 conv layers          | Better feature extraction                   |
| **Batch Norm**         | No            | Yes                    | Stabilizes training                         |
| **Regularization**     | None          | Dropout + Weight Decay | Prevents overfitting                        |
| **Early Stopping**     | No            | Yes                    | Stops training before overfit               |
| **Augmentation**       | Minimal       | Rotation + Flip        | More robust training                        |
| **Generalization Gap** | +0.32         | –0.016                 | Improved model generalizes correctly        |

---

## Conclusion

Even with only ~100 training images, the improved CNN demonstrates that:

* **Proper architecture design matters**
* **Overly large models destroy generalization**
* **Regularization dramatically improves performance**
* **Adaptive pooling and batch normalization stabilize small-dataset training**

The ImprovedCNN achieves **84.62% validation accuracy**, outperforming the baseline model by more than **34 percentage points**, and does so with **95% fewer parameters**.

### **Takeaway:**

For small datasets, prioritize:

1. Controlled model capacity
2. Strong regularization
3. Sufficient feature extraction depth
4. Early stopping
5. Data augmentation
6. Adaptive pooling to reduce FC layers

---

## Dependencies

* torch
* torchvision
* numpy
* matplotlib
* pillow
* scikit-learn

---

## Usage Instructions

1. Place images into `images/pigs` and `images/not pigs`.
2. Open **ML_Project3.4 copy.ipynb**.
3. Run all cells to preprocess data, train both models, evaluate performance, and save the improved model checkpoint.

---

## Authors

Group 11:
**William Donnell-Lonon**
**Emma Ewing**
