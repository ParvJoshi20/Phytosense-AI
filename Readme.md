# PhytoSense AI 🌱

> Current ML workflow and development progress for PhytoSense AI.

---

## 📁 Current Project Structure

```text
backend/
└── ml_pipeline/
    │
    ├── audit_dataset.py
    ├── config.py
    ├── dataset.py
    ├── evaluate.py
    ├── prepare_plantvillage.py
    ├── requirements.txt
    ├── test_dataset.py
    │
    ├── manifests/
    │   └── plantvillage_tomato_split.csv
    │
    ├── models/
    │   └── efficientnetv2.py
    │
    └── preprocessing/
        └── transforms.py
```
---

## ✅ Current ML Pipeline Status

```md
[x] PlantVillage tomato dataset verified
[x] 10 tomato disease/healthy classes verified
[x] Reproducible 70/15/15 train/validation/test split
[x] Dataset integrity and leakage audit
[x] Dataset manifest generated
[x] Image preprocessing pipeline implemented
[x] Dataset and DataLoader pipeline implemented
[x] Class distribution verified
[x] Moderated class weights implemented
[x] Dataset/DataLoader smoke test passed
[ ] EfficientNetV2-B0 model implementation prepared
[ ] Final model configuration
[ ] Model training
[ ] Model evaluation
[ ] Grad-CAM
[ ] SHAP
```

---

## 🛣️ Development Roadmap

```text
Dataset Preparation
        ↓
Preprocessing
        ↓
Dataset / DataLoader Validation
        ↓
Model Configuration
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Grad-CAM + SHAP
        ↓
Final ML Pipeline
```

---

## 🔬 Research Direction

The current ML pipeline establishes a reproducible baseline using:

* PlantVillage tomato images
* Controlled 70/15/15 dataset splitting
* Modern image preprocessing
* Class-imbalance handling
* Pretrained deep learning
* Validation-based model selection
* Independent test evaluation
* Explainable AI analysis

This baseline will support further experimentation and the research paper.

---

## 📌 Current Status

**Current Phase: Model Configuration**

Dataset preparation, preprocessing, and Dataset/DataLoader validation have been completed successfully.

The next step is to finalize the model configuration and training strategy before beginning the first full training run.

---
