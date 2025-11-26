# ML_Project_3: Razorback Logo Image Classification

## Project Overview

This project trains convolutional neural networks (CNNs) to classify images as either containing the official University of Arkansas Razorback logo or not. The updated version introduces a clearer project structure, two explicit model architectures (a baseline and an improved CNN), centralized configuration, enhanced preprocessing, stronger regularization, adaptive pooling, and improved evaluation tools such as ROC curves, F1-score optimization, and threshold tuning.

The dataset includes images collected from Etsy listings, Facebook Marketplace, and official Razorback branding.

---

## Dataset

Images are organized using the `ImageFolder` structure:

* `images/pigs` — contains valid Razorback logo images (label = 1)
* `images/not pigs` — contains non-Razorback images (label = 0)

Total images: **126**

A reproducible split is created using a seeded random permutation:

* **80% training**
* **20% validation**

---

## Configuration

All key hyperparameters and settings are defined in a dedicated config block:

* `IMAGE_SIZE = (500, 500)`
* `BATCH_SIZE = 8`
* `NUM_EPOCHS = 20`
* `LEARNING_RATE = 1e-3`
* `WEIGHT_DECAY = 1e-4`
* `DROPOUT_RATE = 0.5`
* `SEED = 42`

A `set_seed()` function ensures reproducibility across Python, NumPy, and PyTorch.

---

## Image Preprocessing

Training and validation pipelines are defined separately:

### **Training Transformations**

* Resize to 500×500
* Random horizontal flip
* Random rotation (±10°)
* Convert to tensor

```python
train_transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
```

### **Validation Transformations**

* Resize to 500×500
* Convert to tensor

```python
val_transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])
```

This ensures augmentation applies only to training data while validation remains deterministic.

---

## Model Architectures

### **1. BaselineCNN (Overfitting Model)**

A small CNN designed to demonstrate overfitting on a small dataset:

* Two convolution + pooling blocks
* No batch normalization
* Very large fully connected layer (`32 × 125 × 125 → 64`)
* No regularization

As expected, this model achieves nearly **100% training accuracy** but much worse generalization.

---

### **2. ImprovedCNN (Final Model)**

A deeper, regularized architecture with far better generalization:

**Feature extractor (4 blocks):**

* Conv → BatchNorm → ReLU → MaxPool
* Channels: 3 → 32 → 64 → 128 → 256

**Adaptive pooling & classifier:**

* `AdaptiveAvgPool2d((4, 4))` to reduce FC size
* Flatten
* Linear → ReLU → Dropout(0.5)
* Linear → 1 output logit

This design dramatically reduces overfitting and increases validation stability.

---

## Training Methodology

Both models are trained using the same pipeline:

* **Optimizer:** Adam (LR = 1e-3, weight decay = 1e-4)
* **Loss:** `BCEWithLogitsLoss`
* **Batch size:** 8
* **Device:** CPU or CUDA
* **Early stopping:** stops training if validation accuracy does not improve for 10 epochs
* **Best model tracking:** best-performing weights are restored at the end

During training, the notebook records:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

---

## Evaluation Tools

The notebook includes additional evaluation functionality:

### **Threshold Tuning**

* Tests thresholds from 0.1 to 0.9
* Selects the threshold with the **best F1-score**
* Reports the corresponding accuracy

### **Metrics Generated**

* Final train accuracy
* Final validation accuracy
* Best F1-score
* ROC curve + AUC
* Confusion matrix
* Visualizations of predictions on validation samples

---

## Results

### **BaselineCNN**

* Training Accuracy: **≈ 1.00**
* Validation Accuracy: **≈ 0.70**
* Severe overfitting due to massive FC layer and minimal regularization

### **ImprovedCNN**

* Training Accuracy: **≈ 0.81**
* Validation Accuracy: **≈ 0.73**
* Significantly improved stability
* Much lower overfitting
* Better F1-score after optimal threshold selection

Even with only ~100 training images, the improved CNN achieves consistent performance and demonstrates meaningful generalization.

---

## Saved Model

The improved model is saved as:

```
Group_11_CNN_FullModel.ph
```

Load it with:

```python
model = ImprovedCNN()
model.load_state_dict(torch.load("Group_11_CNN_FullModel.ph", map_location=device))
model.eval()
```

---

## Suggestions for Improvement

* Collect more labeled Razorback and non-Razorback images
* Try transfer-learning architectures (ResNet18, MobileNetV3, EfficientNet-B0)
* Add additional augmentations (color jitter, blur, cutout)
* Experiment with Mixup or CutMix
* Tune dropout rate, learning rate, and convolution depth
* Add Precision-Recall curves and calibration plots

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

1. Place images into the `images/pigs` and `images/not pigs` folders.
2. Open **ML_Project3.4 copy.ipynb**.
3. Run all cells to:

   * preprocess the data
   * train both CNN models
   * evaluate validation performance
   * save the improved model

---

## Authors

Group 11:
**William Donnell-Lonon**
**Emma Ewing**
