# 🔬 Diabetic Retinopathy Classification using EfficientNet-B4 + Vision Transformer (ViT)


<p align="center"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/> <img src="https://img.shields.io/badge/timm-FF6F00?style=for-the-badge"/> </p>

 Project Overview
Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes and can lead to blindness if not detected early. This project builds an automated DR severity classification system using a hybrid deep learning model that combines the power of Convolutional Neural Networks (CNN) and Vision Transformers (ViT).

The model takes a retinal fundus image as input and predicts one of 5 severity levels of Diabetic Retinopathy, helping ophthalmologists with faster and more accurate diagnosis.

 Table of Contents
Dataset
Problem Statement
Architecture
Architecture Flow
Data Pipeline
Training Strategy
Evaluation Metrics
Results
Project Structure
How to Run
Dependencies
Author
Dataset
Name: APTOS 2019 Blindness Detection
Source: Kaggle Competition
Type: Retinal fundus images (PNG format)
Task: Multi-class classification (5 classes)
Class Distribution
Label	Severity Level	Description
0	No DR	No signs of diabetic retinopathy
1	Mild	Microaneurysms only
2	Moderate	More than just microaneurysms but less than PDR
3	Severe	Extensive abnormalities, no signs of PDR
4	Proliferative DR	Most severe stage with neovascularization
Data Split
Split	Percentage	Purpose
Train	70%	Model training
Validation	15%	Hyperparameter tuning
Test	15%	Final evaluation
Stratified splitting was used to maintain class distribution across all splits.

  Problem Statement
Diabetic Retinopathy affects millions of diabetic patients worldwide. Manual screening by trained professionals is:

Time-consuming
Expensive
Prone to human error
Not scalable in rural/low-resource settings
This project aims to build an automated, accurate, and scalable DR grading system using state-of-the-art deep learning techniques.

  Model Architecture
The model EfficientNet_ViT is a dual-branch hybrid architecture that processes the same image through two parallel networks and combines their strengths:

Branch 1 — EfficientNet-B4 (CNN)
Pre-trained on ImageNet
Captures local features: edges, textures, microaneurysms, hemorrhages
Outputs a high-dimensional feature vector
Used with num_classes=0 to extract features only (no classification head)
Branch 2 — ViT-Tiny-Patch16-224 (Vision Transformer)
Pre-trained on ImageNet
Divides image into 16×16 patches and processes them with self-attention
Captures global features: spatial relationships, overall retinal structure
Used with num_classes=0 to extract features only
Fusion & Classification Head
Feature vectors from both branches are concatenated
Passed through a fully connected classifier:
  Linear(cnn_dim + vit_dim → 512) → ReLU → Dropout(0.4) → Linear(512 → 5)
🔄 Architecture Flow
Input Retinal Image (224×224×3)
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
 ┌──────────────────┐               ┌──────────────────────┐
 │  EfficientNet-B4 │               │   ViT-Tiny-Patch16   │
 │  (CNN Branch)    │               │   (Transformer Branch)│
 │                  │               │                       │
 │ • Compound       │               │ • Split image into    │
 │   scaling        │               │   16×16 patches       │
 │ • MBConv blocks  │               │ • Patch embeddings    │
 │ • SE attention   │               │ • Multi-head          │
 │ • Local feature  │               │   self-attention      │
 │   extraction     │               │ • Global feature      │
 │                  │               │   extraction          │
 └────────┬─────────┘               └──────────┬────────────┘
          │                                     │
          │  CNN Feature Vector                 │  ViT Feature Vector
          │  (cnn_dim)                          │  (vit_dim)
          │                                     │
          └──────────────┬──────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Feature Concat     │
              │  [cnn_feat, vit_feat]│
              │  (cnn_dim + vit_dim) │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Linear Layer       │
              │  → 512 units        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  ReLU Activation    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Dropout (p=0.4)    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Linear Layer       │
              │  → 5 classes        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Output Prediction  │
              │  (0, 1, 2, 3, 4)    │
              └─────────────────────┘
Why This Hybrid Approach?
Feature	CNN (EfficientNet)	ViT
Focus	Local patterns	Global context
Strength	Texture, edges, fine details	Long-range dependencies
Weakness	Limited global view	Needs large data
Together	✅ Best of both worlds	✅ Complementary features
🔧 Data Pipeline
Custom Dataset Class — APTOSDataset
Loads images from disk using PIL
Converts to RGB
Applies albumentations transforms
Returns (image_tensor, label) pairs
Data Augmentation
Training Augmentations (to prevent overfitting):

Augmentation	Parameters
RandomResizedCrop	size=224×224, scale=(0.8, 1.0)
HorizontalFlip	p=0.5
VerticalFlip	p=0.5
Rotate	limit=±30°, p=0.5
RandomBrightnessContrast	p=0.4
Normalize	mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
Validation/Test Augmentations (only preprocessing):

Step	Parameters
Resize	224×224
Normalize	ImageNet mean & std
🏋️ Training Strategy
Loss Function — Focal Loss
Standard cross-entropy struggles with class imbalance. Focal Loss addresses this by:

Down-weighting easy examples
Focusing learning on hard misclassified samples
Formula: FL = (1 - pt)^γ * CE, where γ=2
This is especially important for DR classification since class 0 (No DR) is much more frequent.

Optimizer — AdamW
Learning rate: 1e-4
AdamW adds weight decay regularization to prevent overfitting
Learning Rate Scheduler — StepLR
Reduces LR by factor of 0.5 every 3 epochs
Helps model converge to a better minimum in later epochs
Training Loop Summary
For each epoch (1 to 10):
  → Train on train_loader (forward + backward + optimize)
  → Evaluate on val_loader (no gradient)
  → Step the LR scheduler
  → Print: Loss, Train Acc, Val Acc
Hyperparameter	Value
Epochs	10
Batch Size	16
Optimizer	AdamW
Learning Rate	1e-4
Loss Function	Focal Loss (γ=2)
LR Scheduler	StepLR (step=3, γ=0.5)
Dropout	0.4
Device	GPU (CUDA) / CPU
📊 Evaluation Metrics
The model is evaluated on Train, Validation, and Test sets using:

Metric	Description
Accuracy	Overall correct predictions / total
Precision (Macro)	Average precision across all classes equally
Precision (Weighted)	Precision weighted by class support
Recall (Macro)	Average recall across all classes equally
Recall (Weighted)	Recall weighted by class support
F1-Score (Macro)	Harmonic mean of precision & recall (macro)
F1-Score (Weighted)	Harmonic mean of precision & recall (weighted)
QWK	Quadratic Weighted Kappa — measures agreement considering severity ordering
ROC-AUC (Macro OvR)	Area under ROC curve for multi-class (One vs Rest)
Confusion Matrix	Full breakdown of predictions per class
Classification Report	Per-class precision, recall, F1
QWK (Quadratic Weighted Kappa) is the most important metric for DR grading as it penalizes predictions that are far from the true severity level more heavily than close predictions.

📁 Project Structure
DR-Classification-ViT/
│
├── DR_classification_by_VIT.ipynb   # Main Jupyter Notebook
├── README.md                        # Project documentation
│
└── (Dataset - Download from Kaggle)
    └── aptos2019-blindness-detection/
        ├── train.csv
        └── train_images/
            ├── image1.png
            ├── image2.png
            └── ...
🚀 How to Run
Option 1: Run on Kaggle (Recommended)
Go to Kaggle and create an account
Navigate to the APTOS 2019 competition and join it
Upload DR_classification_by_VIT.ipynb as a new notebook
Enable GPU accelerator (Settings → Accelerator → GPU)
Run all cells
Option 2: Run Locally
Clone this repository:
bash
   git clone https://github.com/your-username/DR-Classification-ViT.git
   cd DR-Classification-ViT
Install dependencies:
bash
   pip install torch torchvision timm albumentations scikit-learn pandas numpy pillow
Download the dataset from Kaggle:
bash
   kaggle competitions download -c aptos2019-blindness-detection
   unzip aptos2019-blindness-detection.zip -d data/
Update the DATA_DIR path in the notebook to point to your local dataset
Open and run the notebook:
bash
   jupyter notebook DR_classification_by_VIT.ipynb
📦 Dependencies
torch
torchvision
timm
albumentations
scikit-learn
pandas
numpy
pillow
Install all at once:

bash
pip install torch torchvision timm albumentations scikit-learn pandas numpy pillow
🔑 Key Concepts Used
Transfer Learning — Both EfficientNet-B4 and ViT are initialized with ImageNet pretrained weights
Feature Fusion — Combining CNN and Transformer features for richer representation
Focal Loss — Handles class imbalance in medical imaging datasets
Stratified Split — Ensures balanced class distribution across train/val/test
Data Augmentation — Prevents overfitting on medical image data
AdamW + StepLR — Stable training with decaying learning rate


👤 Author

[ABHISHEK SINGH] — VIT

📧 [your-ar5621996@gmail.com]

📜 License
This project is for academic and educational purposes.
Dataset credit: APTOS 2019 Blindness Detection — Kaggle

⭐ If you found this project helpful, please give it a star on GitHub!







👤 Author

[ABHISHEK SINGH] — VIT

📧 [your-ar5621996@gmail.com]
