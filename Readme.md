# Phytosense AI 🌱

> **An HXAI-Powered Intelligent Decision Support System for Tomato Disease Diagnostics.**

Phytosense AI is an end-to-end, publication-grade diagnostic tool designed to bridge the gap between complex deep learning models and actionable agricultural interventions. It moves beyond "black-box" classification by employing a Human-Centric Explainable AI (HXAI) approach.

## 🧠 The 4-Layer HXAI Architecture
1. **Layer 1: Disease Detection & Severity** - High-accuracy classification across a 4-tier spectrum (Healthy, Mild, Moderate, Severe) with calibrated confidence scoring.
2. **Layer 2: Visual Explainability** - Interactive Grad-CAM heatmaps highlighting the exact morphological features driving the model's prediction.
3. **Layer 3: Natural Language Reasoning** - "Why this prediction?" and "Why not another disease?" counterfactual explanations mapping latent features to human-readable symptoms.
4. **Layer 4: Decision Support** - Contextualized, actionable treatment plans broken down into immediate, organic, and chemical interventions.

## 🛠 Tech Stack
* **Frontend Space:** Next.js, Tailwind CSS, Framer Motion (Deep dark-mode UI with reactive state-driven lighting).
* **Backend Space:** FastAPI, Python, PostgreSQL.
* **Machine Learning Space:** PyTorch, EfficientNetV2, Grad-CAM.
* **Datasets:** PlantVillage, PlantDoc, Mendeley Data.

## 🚀 Getting Started

To run this project locally, you will need to boot up both the frontend and backend environments in separate terminal windows.

### 1. Backend Setup (FastAPI & PyTorch)
```bash
cd backend
pip install -r requirements.txt
# Start the FastAPI server
uvicorn app.api.main:app --reload