# DR-classification-by-vit

DR Classification using Vision Transformer (ViT) + EfficientNet

Overview

This project classifies Diabetic Retinopathy (DR) severity from retinal fundus images using a hybrid deep learning model combining EfficientNet-B4 and Vision Transformer (ViT-Tiny). The model is trained on the APTOS 2019 Blindness Detection dataset from Kaggle.


 Model Architecture

A custom hybrid model EfficientNet_ViT that:


Extracts local features using EfficientNet-B4 (CNN)
Extracts global features using ViT-Tiny-Patch16-224 (Transformer)
Concatenates both feature vectors and passes through a fully connected classifier



 Dataset


Source: APTOS 2019 Blindness Detection
Classes (5):

0 — No DR
1 — Mild
2 — Moderate
3 — Severe
4 — Proliferative DR






⚙️ Tech Stack


Python, PyTorch
timm (EfficientNet-B4, ViT-Tiny)
albumentations (data augmentation)
scikit-learn (evaluation metrics)



🏋️ Training Details

ParameterValueInput Size224×224Batch Size16Epochs10OptimizerAdamW (lr=1e-4)LossFocal Loss (γ=2)SchedulerStepLR (step=3, γ=0.5)Train/Val/Test Split70/15/15


📊 Evaluation Metrics


Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
Quadratic Weighted Kappa (QWK)
ROC-AUC (Macro, OvR)
Confusion Matrix & Classification Report



How to Run


Clone this repo
Install dependencies:


bash   pip install timm albumentations torch torchvision scikit-learn


Download the APTOS 2019 dataset from Kaggle and place it at:


   /kaggle/input/competitions/aptos2019-blindness-detection/


Run the notebook: DR_classification_by_VIT.ipynb



👤 Author

[ABHISHEK SINGH] — VIT

📧 [your-ar5621996@gmail.com]
