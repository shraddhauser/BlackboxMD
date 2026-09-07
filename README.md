# BLACKBOX MD

### Explainable AI Clinical Decision Support System

> **An Explainable AI prototype for lung cancer CT classification with visual and natural-language explanations.**

BLACKBOX MD is an **Explainable Artificial Intelligence (XAI)** based clinical decision-support project designed to make medical AI predictions more understandable to clinicians.

The current prototype focuses on **lung cancer CT image classification** using a pretrained **DenseNet-121** model. In addition to predicting the cancer category, BLACKBOX MD uses **Grad-CAM** to identify image regions that influenced the model's decision and an **NLP-based explanation layer** to convert this information into a human-readable explanation.

The goal is not to replace a doctor, but to provide an additional, interpretable AI perspective that can support clinical decision-making.

---

##  Problem

Deep learning models can achieve strong performance on medical image classification, but their predictions are often difficult to interpret.

A model may predict:

> **Adenocarcinoma — 97.10% confidence**

but this alone does not tell a clinician:

* Which region of the CT image influenced the prediction?
* How strongly did the model focus on that region?
* How can the AI result be communicated in understandable language?
* Can the prediction be presented as an interpretable second opinion rather than a black-box result?

BLACKBOX MD addresses this problem by combining:

**Medical Image → AI Prediction → Visual Explanation → Natural-Language Explanation**

---

#  What We Built

The current working prototype consists of three major layers:

### 1. Deep Learning Classification

A pretrained **DenseNet-121** model performs four-class CT image classification.

### 2.  Visual Explainability

**Grad-CAM** generates a heatmap showing the regions that contributed to the model's prediction.

### 3. 💬 NLP Explanation Layer

The prediction confidence and Grad-CAM attention information are converted into a clinician-readable explanation.

This creates an explainability pipeline rather than providing only a raw classification result.

---

#  Current Prototype Architecture

```text
                    ┌─────────────────────┐
                    │   CT Scan Image     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    │ 256 × 256           │
                    │ Normalization       │
                    │ Augmentation         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     DenseNet-121    │
                    │ ImageNet Pretrained │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │   Prediction     │   │    Grad-CAM      │
          │ Class + Confidence│   │ Attention Map   │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │ NLP Explanation     │
                    │ Layer               │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Human-Readable      │
                    │ AI Explanation      │
                    └─────────────────────┘
```

---

# 🧪 Dataset

The current prototype uses the:

**Kaggle Chest CT-Scan Images Dataset**

Dataset characteristics used in the current implementation:

| Property            | Value                         |
| ------------------- | ----------------------------- |
| Task                | Lung cancer CT classification |
| Number of classes   | 4                             |
| Test images         | 315                           |
| Input size          | 256 × 256                     |
| Dataset source      | Kaggle Chest CT-Scan Images   |
| Classification type | Multi-class                   |

### Classes

The current model predicts:

* Adenocarcinoma
* Large Cell Carcinoma
* Normal
* Squamous Cell Carcinoma

---

#  Image Preprocessing

The preprocessing pipeline includes:

* Image resizing to **256 × 256**
* ImageNet mean/std normalization
* Training-only augmentation
* Random affine transformations
* Rotation of approximately **±10°**
* Scale variation between **0.90–1.10**
* Visual verification of sample images before training

The augmentation is applied only during training to avoid modifying validation and test distributions.

---

#  Deep Learning Model

### DenseNet-121

The classification backbone is an **ImageNet-pretrained DenseNet-121**.

The final classification head is adapted for the four lung CT classes.

### Training configuration

| Parameter         | Configuration |
| ----------------- | ------------- |
| Backbone          | DenseNet-121  |
| Pretraining       | ImageNet      |
| Optimizer         | Adam          |
| Learning rate     | 0.0001        |
| Loss function     | Cross-Entropy |
| Epochs            | 20            |
| Number of classes | 4             |
| Input size        | 256 × 256     |

The best validation checkpoint was selected rather than automatically using the final epoch. The best validation accuracy was **95.83% at epoch 7**.

---

#  Current Results

The current prototype achieved:

### **96.19% Test Accuracy**

on the current test set of **315 images**.

### Class-wise performance

| Class                   | Precision | Recall | F1-Score |
| ----------------------- | --------: | -----: | -------: |
| Adenocarcinoma          |     0.991 |  0.917 |    0.952 |
| Large Cell Carcinoma    |     0.879 |  1.000 |    0.936 |
| Normal                  |     1.000 |  0.982 |    0.991 |
| Squamous Cell Carcinoma |     0.957 |  0.989 |    0.973 |

### Observed confusion

The main systematic confusion in the current evaluation occurred between:

**Adenocarcinoma ↔ Large Cell Carcinoma**

while the Normal class was classified with very high performance.

> **Important:** These results represent the current prototype's performance on its available test set and should not be interpreted as clinical diagnostic accuracy.

---

#  Explainability with Grad-CAM

A major component of BLACKBOX MD is **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Instead of returning only:

```text
Prediction: Adenocarcinoma
Confidence: 97.10%
```

the system also produces a visual heatmap showing the regions that contributed to the model's prediction.

### Grad-CAM provides:

* Prediction-specific attention visualization
* Localization of influential image regions
* Visual interpretation of model decisions
* A way to inspect whether the model is focusing on meaningful regions

The Grad-CAM implementation was also validated across test batches and rewritten as a reusable function after resolving gradient-hook handling issues.

---

# 💬 NLP Explanation Layer

### Our additional component

The major addition to the existing image-classification/XAI pipeline is the **NLP Explanation Layer**.

The layer takes structured information from the prediction and Grad-CAM analysis and converts it into a human-readable explanation.

Current inputs include:

```text
Predicted Class
        +
Confidence Score
        +
Grad-CAM Attention
        ↓
NLP Explanation
```

The system currently supports two explanation approaches.

---

## 1. Rule-Based Explanation

The rule-based generator:

* Categorizes confidence into predefined levels
* Categorizes attention into predefined levels
* Uses a deterministic explanation template
* Produces explanations with zero model-inference cost
* Adds an appropriate clinical caveat

Example:

> **The AI model predicts Adenocarcinoma with a 97.10% confidence level. The Grad-CAM analysis shows strong model attention, with approximately 22.32% of the image showing strong activation. These highlighted areas indicate regions that influenced the model's classification. The result is an AI prediction and should be interpreted together with appropriate clinical assessment.**

---

## 2. LLM-Prompt Explanation

The second approach structures the same model information into a prompt for a language model.

The prompt is designed to:

* Explain the model output
* Describe the significance of the highlighted regions
* Avoid unsupported medical claims
* Clearly distinguish AI prediction from diagnosis
* Maintain a clinical safety caveat

The system explicitly treats Grad-CAM regions as areas that **influenced the model**, rather than claiming that those regions independently prove a diagnosis.

---

# 🔄 End-to-End Workflow

```text
1. Upload CT Image
        ↓
2. Preprocess Image
        ↓
3. DenseNet-121 Classification
        ↓
4. Generate Prediction + Confidence
        ↓
5. Generate Grad-CAM Heatmap
        ↓
6. Calculate Attention Information
        ↓
7. NLP Explanation Generation
        ↓
8. Display Prediction + Explanation
```

---

# Planned Application Architecture

The next stage is to integrate the current ML/XAI pipeline into a full application.

```text
                ┌─────────────────────┐
                │   React Dashboard   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    FastAPI Backend  │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌─────────────────┐      ┌─────────────────┐
      │ DenseNet-121    │      │ Explanation     │
      │ Classifier      │      │ Engine          │
      └────────┬────────┘      └────────┬────────┘
               │                        │
               ▼                        ▼
      ┌─────────────────┐      ┌─────────────────┐
      │   Grad-CAM      │      │ NLP Explanation │
      └─────────────────┘      └─────────────────┘
```

---

# 🛣️ Roadmap

## Phase 1 — Current Prototype

* [x] CT image preprocessing
* [x] Four-class DenseNet-121 classifier
* [x] Model training and evaluation
* [x] Best-checkpoint selection
* [x] Grad-CAM implementation
* [x] Grad-CAM validation
* [x] Rule-based NLP explanation
* [x] LLM-prompt explanation design

## Phase 2 — Application Integration

* [ ] FastAPI backend
* [ ] Model inference API
* [ ] Grad-CAM generation API
* [ ] NLP explanation API
* [ ] React dashboard
* [ ] Upload-and-predict workflow
* [ ] Prediction + heatmap visualization
* [ ] Human-readable explanation display

## Phase 3 — Clinical Data Integration

* [ ] EHR integration
* [ ] FHIR-based data exchange
* [ ] Patient-context integration
* [ ] Clinical alert framework
* [ ] Audit logging

## Phase 4 — Research Expansion

* [ ] Larger and more diverse datasets
* [ ] Cross-validation
* [ ] Robustness testing
* [ ] Additional medical conditions
* [ ] Hindi/Hinglish NLP support
* [ ] Broader multimodal clinical information

---

#  Research Foundation

BLACKBOX MD is being developed around the principle that medical AI should provide not only predictions but also interpretable evidence about the model's decision process.

Grad-CAM is particularly useful for medical imaging because it can provide spatial information about regions influencing a prediction.

However, explainability maps should be treated as **model-attribution information**, not as independent medical evidence.

The current research direction also recognizes that generalization depends on dataset diversity, imaging variations, and evaluation across broader patient populations.

---

#  Limitations

The current prototype has several important limitations:

* The model is currently trained and evaluated on a specific Kaggle dataset.
* The current dataset is relatively limited compared with real-world clinical data.
* The four-class classification task does not represent the full complexity of lung cancer diagnosis.
* Grad-CAM highlights regions influencing the model but does not establish causality or diagnosis.
* NLP-generated explanations must not introduce information unsupported by the model output.
* The current system has not undergone clinical validation.
* The current performance should not be interpreted as real-world clinical performance.

---

# 🩺 Clinical Safety Principle

BLACKBOX MD is designed as a **clinical decision-support / second-opinion prototype**, not an autonomous diagnostic system.

```text
AI Prediction
     +
Visual Explanation
     +
Natural-Language Explanation
     ↓
Clinical Support
     ↓
NOT Autonomous Diagnosis
```

The final clinical decision must remain with a qualified healthcare professional.

---

#  Ethics & Responsible AI

The project follows these principles:

* AI output should be treated as advisory.
* The system should not replace qualified medical professionals.
* Explanations should not overstate model certainty.
* Grad-CAM visualizations should not be presented as definitive lesion localization.
* Patient privacy should be considered when integrating clinical data.
* Future clinical deployment requires appropriate validation and regulatory consideration.

---

#  Project Structure

The structure will evolve as the working prototype is integrated into the application.

```text
BLACKBOX-MD/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── model/
│   ├── train.py
│   ├── evaluate.py
│   └── classifier.py
│
├── explainability/
│   └── gradcam.py
│
├── nlp/
│   ├── rule_based.py
│   └── llm_prompt.py
│
├── backend/
│   └── FastAPI/
│
├── frontend/
│   └── React/
│
├── notebooks/
│
├── research/
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* PyTorch
* DenseNet-121
* Torchvision

### Explainable AI

* Grad-CAM
* Gradient-based visual attribution

### NLP

* Rule-based explanation generation
* LLM prompt-based explanation generation

### Backend

* FastAPI *(integration stage)*

### Frontend

* React *(integration stage)*

### Future Clinical Integration

* FHIR
* EHR systems
* Healthcare APIs

---

#  Getting Started

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd BLACKBOX-MD
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Prepare the dataset

Place the CT dataset according to the project data structure:

```text
data/
├── train/
├── validation/
└── test/
```

## 5. Train the model

```bash
python model/train.py
```

## 6. Evaluate the model

```bash
python model/evaluate.py
```

## 7. Generate Grad-CAM explanations

Run the Grad-CAM pipeline using a trained model checkpoint and input CT image.

## 8. Generate the NLP explanation

Pass the prediction and Grad-CAM information to the explanation layer.

---

#  Current Development Status

| Component                 | Status                         |
| ------------------------- | ------------------------------ |
| CT preprocessing          | ✅ Completed                    |
| DenseNet-121 classifier   | ✅ Completed                    |
| Model evaluation          | ✅ Completed                    |
| 96.19% test accuracy      | ✅ Achieved on current test set |
| Grad-CAM                  | ✅ Implemented                  |
| Grad-CAM validation       | ✅ Completed                    |
| Rule-based NLP            | ✅ Implemented                  |
| LLM-prompt explanation    | ✅ Implemented                  |
| FastAPI integration       | 🔄 Next                        |
| React dashboard           | 🔄 Next                        |
| Cross-validation          | 🔄 Planned                     |
| Larger dataset validation | 🔄 Planned                     |
| EHR/FHIR integration      | 🔮 Future                      |
| Hindi/Hinglish NLP        | 🔮 Future                      |

---

# 🎯 Vision

The long-term vision of BLACKBOX MD is to move from a conventional medical AI classifier toward an **interpretable multimodal clinical decision-support system**.

The intended progression is:

```text
Medical Images
      +
Clinical Data
      +
Medical Reports
      ↓
AI Prediction
      ↓
Explainability
      ↓
Natural-Language Reasoning
      ↓
Clinician-Friendly Decision Support
```

The current lung CT prototype serves as the foundation for this broader system.

---

# 👩‍💻 Team

**BLACKBOX MD**

Developed as a student research and engineering project focused on:

* Artificial Intelligence
* Deep Learning
* Explainable AI
* Medical Imaging
* NLP
* Clinical Decision Support

---

# Disclaimer

BLACKBOX MD is an experimental research prototype.

It is **not a medical device and is not intended to diagnose, treat, cure, or prevent any disease**.

Model predictions and explanations should not be used as a substitute for professional medical evaluation.

Any future clinical deployment would require extensive validation, appropriate datasets, expert review, privacy protections, and applicable regulatory approvals.
