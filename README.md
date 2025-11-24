# ML_Project_3: Razorback Logo Image Classification (Convolutional Neural Network)

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images as either containing the official University of Arkansas Razorback logo or not. The dataset includes images scraped from Etsy listings, Facebook Marketplace, and official Razorback branding. This project performs preprocessing, model training, early stopping, evaluation, and saving the best model for future use.

## Dataset

The dataset is divided into two folders:

- images/pigs : images containing the official Razorback logo (label = 1)
- images/not pigs : images without the official Razorback logo (label = 0)

Total images: 126

Train/validation split: 80 percent training, 20 percent validation

## Image Preprocessing

Each image is transformed using a preprocessing pipeline that includes:

1. Resizing all images to 500x500 pixels  
2. Random horizontal flipping for simple data augmentation  
3. Converting images to PyTorch tensors and scaling pixel values to the 0–1 range

Transform pipeline used:

transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

## Model Architecture

A small custom CNN was built to balance model capacity and overfitting risk due to the small dataset.

Model structure:

- Convolutional Layer 1 (3 input channels to 16 filters, ReLU)
- MaxPooling Layer (reduces image size from 500x500 to 250x250)
- Convolutional Layer 2 (16 filters to 32 filters, ReLU)
- MaxPooling Layer (reduces size from 250x250 to 125x125)
- Flatten Layer (32 * 125 * 125 features)
- Fully Connected Layer (64 units, ReLU)
- Dropout Layer (30 percent dropout)
- Output Layer (1 logit for binary classification)

Model code:

class RazorbackCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = 32 * 125 * 125
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

## Training Methodology

- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam with learning rate 0.001
- Batch size: 8
- Train/validation split created using random_split
- Early stopping used by tracking best validation accuracy

Accuracy was calculated by applying a sigmoid to model outputs and comparing predictions to labels.

## Results

The final model achieved a best validation accuracy of approximately 76 percent.

Example training output:

Epoch 02 | Loss: 0.6976 | Train Acc: 0.64 | Val Acc: 0.68  
Epoch 03 | Loss: 0.6843 | Train Acc: 0.59 | Val Acc: 0.76  
Epoch 05 | Loss: 0.6696 | Train Acc: 0.71 | Val Acc: 0.76  
Best Val Acc: 0.76

This accuracy is strong given the small dataset size and real-world variation in images.

## Saved Model

The best model from training is saved as:

models/Group_#_CNN_FullModel.ph

It can be loaded with:

model = RazorbackCNN()
model.load_state_dict(torch.load("models/Group_#_CNN_FullModel.ph"))
model.eval()

## Suggestions for Improvement

- Collect more Razorback and non-Razorback images  
- Experiment with transfer learning (such as ResNet18)  
- Add more augmentation techniques  
- Tune hyperparameters such as dropout rate, learning rate, and number of filters  
- Add evaluation tools such as confusion matrices and ROC curves  

## Dependencies

torch  
torchvision  
numpy  
matplotlib  
pillow

## Usage Instructions

1. Place images into the images folder with pigs and not pigs subfolders  
2. Open the ML_Project3.ipynb notebook  
3. Run all cells to preprocess the data, train the CNN, and save the best model  

## Authors

Group 11:

William Donnell-Lonon  
Emma Ewing

## License

This project is part of an academic assignment for machine learning coursework.
