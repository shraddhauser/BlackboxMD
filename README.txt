# BLACKBOX MD — Comparative XAI Experiment

## Dataset
The supplied dataset is placed under:
Data/train/
Data/test/

Classes:
- adenocarcinoma
- large.cell.carcinoma
- normal
- squamous.cell.carcinoma

## Project structure

BLACKBOX_MD_XAI/
├── Data/
│   ├── train/
│   └── test/
├── models/
│   └── best_densenet121.pth
├── xai/
│   ├── xai_common.py
│   ├── gradcam.py
│   ├── integrated_gradients.py
│   └── shap_explainer.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.txt

## 1. Install dependencies

pip install -r requirements.txt

## 2. Train the new DenseNet-121

python train.py

This creates:
models/best_densenet121.pth

## 3. Evaluate the model

python evaluate.py

## 4. Generate Grad-CAM

python xai/gradcam.py "Data/test/adenocarcinoma/IMAGE.jpg"

## 5. Generate Integrated Gradients

python xai/integrated_gradients.py "Data/test/adenocarcinoma/IMAGE.jpg"

## 6. Generate SHAP

python xai/shap_explainer.py "Data/test/adenocarcinoma/IMAGE.jpg"

All three XAI methods use the exact same DenseNet-121 checkpoint.
