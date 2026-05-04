# Aura Health — Disease & Risk Prediction AI System

**A full-stack machine learning web application that predicts potential diseases and health risk levels based on patient symptoms, vital signs, and laboratory test results.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture & System Design](#architecture--system-design)
4. [Technology Stack](#technology-stack)
5. [Dataset & Data Overview](#dataset--data-overview)
6. [Feature Engineering Pipeline](#feature-engineering-pipeline)
7. [Machine Learning Models](#machine-learning-models)
8. [Model Performance & Results](#model-performance--results)
9. [Installation & Prerequisites](#installation--prerequisites)
10. [Local Development Setup](#local-development-setup)
14. [Project Structure](#project-structure)
18. [Contributing & License](#contributing--license)

---

## Overview

**Aura Health** is a sophisticated health prediction system designed to assist medical professionals and patients in early disease detection and risk assessment. The system combines:

- **Advanced ML modeling** using Random Forest, Gradient Boosting, and Logistic Regression
- **Clinical feature engineering** based on medical expertise and evidence-based thresholds (e.g., NEWS2 severity scoring)
- **Full-stack architecture** with a Python/scikit-learn backend and modern React frontend
- **Real-time predictions** with confidence scores and clinical recommendations
- **Clean, intuitive UI** with glassmorphism design for accessibility

### Purpose & Use Cases

- **Early Screening**: Identify potential diseases from symptoms and vitals before clinical consultation
- **Risk Stratification**: Classify patients into risk levels (Low, Moderate, High, Critical) for triage
- **Clinical Decision Support**: Provide recommendations for follow-up timing based on risk assessment
- **Patient Education**: Help patients understand their health status and when to seek care
- **Research & Analytics**: Analyze disease patterns and risk factors across populations

---

## Key Features

### Core Capabilities

1. **Dual-Task Prediction**
   - **Disease Classification**: Predicts 14+ disease categories from patient input
   - **Risk Level Assessment**: Stratifies patients into 4 risk tiers (Low, Moderate, High, Critical)
   - **Confidence Scores**: Provides probability estimates for each prediction

2. **Comprehensive Input Handling**
   - **Symptoms Input**: 16 clinical symptoms (fever, cough, rash, etc.)
   - **Vital Signs**: Heart rate, blood pressure, oxygen saturation, temperature, respiratory rate
   - **Laboratory Tests**: ALT (liver enzyme), WBC (white blood cell count), TSH (thyroid function)
   - **Clinical State**: Consciousness level, oxygen therapy status

3. **Intelligent Feature Engineering**
   - **Severity Scoring**: NEWS2-inspired composite severity score (0-20)
   - **Ratio Features**: Shock Index, SpO₂-to-Heart Rate ratio
   - **Clinical Flags**: Binary indicators for hypoxia, tachycardia, fever, hypotension
   - **Derived Features**: Log-transformed lab values, symptom counts

4. **User-Friendly Interface**
   - **Premium Glassmorphism Design**: Modern, visually appealing UI with frosted glass effects
   - **Interactive Forms**: Dynamic symptom and vital input with validation
   - **Patient History**: Local storage of past predictions (up to 50 records)
   - **Result Visualization**: Modal displays with confidence levels and recommendations
   - **Multiple Views**: Home, Prediction, History, About, Doctor Portal, Contact

5. **Clinical Recommendations**
   - **Risk-Based Guidance**: Different follow-up recommendations based on risk level
   - **Actionable Advice**: From "Monitor at home" to "Go to ER immediately"
   - **Severity Metrics**: Visual representation of severity score

---

## Architecture & System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Frontend Layer                            │
│                    (React + Vite + Lucide)                      │
│  [Home] [Predict] [History] [About] [Doctor Portal] [Contact]   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────────────┐
│                    Node.js Express Server                       │
│              (CORS, Body Parser, Static Files)                  │
│              /api/predict  ← Main prediction endpoint           │
└─────────────────────────────────────────────────────────────────┘
                              ↓ stdin/stdout
┌─────────────────────────────────────────────────────────────────┐
│                  Python ML Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Model Loader │  │ Feature Eng. │  │  Predictors  │           │
│  │  (Joblib)    │  │   (Pandas)   │  │  (scikit-learn)          │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  Pre-trained Models (stored as .pkl files):                     │
│  - disease_model.pkl   (14-class classifier)                    │
│  - risk_model.pkl      (4-class classifier)                     │
│  - scaler.pkl          (StandardScaler for LR)                  │
│  - le_disease.pkl      (Label encoder for diseases)             │
│  - le_risk.pkl         (Label encoder for risk levels)          │
│  - config.pkl          (Model configuration & metadata)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓ JSON Response
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Display                            │
│  [Result Modal] → Disease | Risk | Confidence | Recommendation  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow in Prediction Pipeline

```
User Input (Symptoms, Vitals, Labs)
         ↓
[JSON Payload] → Express Server
         ↓
[Child Process] → Python (predict_cli.py)
         ↓
[Load Models] → Load pre-trained .pkl files
         ↓
[Feature Engineering] → Construct 40-feature vector
         ↓
[Scaling] → Apply StandardScaler if needed
         ↓
[Prediction] → disease_model.predict() & risk_model.predict()
         ↓
[Post-Processing] → Label decoding, confidence extraction, recommendations
         ↓
[JSON Response] → Back to Frontend
         ↓
[Display Modal] → Show results to user + save to localStorage history
```

---

## Technology Stack

### Backend (Python)

| Component | Version | Purpose |
|-----------|---------|---------|
| **pandas** | Latest | Data manipulation and feature engineering |
| **numpy** | Latest | Numerical computing and array operations |
| **scikit-learn** | Latest | ML models (Random Forest, Gradient Boosting, Logistic Regression) |
| **joblib** | Latest | Model serialization and deserialization |
| **matplotlib** | Latest | Visualization and reporting (optional) |
| **seaborn** | Latest | Statistical visualization (optional) |

### Frontend (JavaScript/React)

| Component | Version | Purpose |
|-----------|---------|---------|
| **React** | 19.2.5 | UI framework and component architecture |
| **Vite** | 8.0.10 | Fast build tool and dev server |
| **Lucide React** | 1.14.0 | Icon library for UI components |
| **ESLint** | 10.2.1 | Code quality and style checking |

### Server (Node.js)

| Component | Version | Purpose |
|-----------|---------|---------|
| **Express.js** | 4.18.2 | HTTP server and routing |
| **CORS** | 2.8.5 | Cross-Origin Resource Sharing |
| **Body Parser** | 1.20.2 | JSON payload parsing |

### Development Environment

- **Python**: 3.9 or higher
- **Node.js**: 16+ (for frontend and Express server)
- **npm**: 8+ (package manager)
- **Operating System**: Linux, macOS, or Windows (with WSL)

---

## Dataset & Data Overview

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Total Records** | 16,000 patient cases |
| **Data Composition** | 1,000 real cases + 15,000 synthetically generated |
| **Diseases Covered** | 14 distinct disease categories |
| **Disease Distribution** | Multi-class, stratified splits (80/20 train-test) |
| **Risk Levels** | 4 categories: Low, Moderate, High, Critical |
| **Features** | 40 total (see breakdown below) |
| **Missing Values** | 0 (clean dataset) |

### Input Variables

#### 1. **Symptoms** (16 Binary Indicators)

| Symptom | Clinical Relevance | Related Diseases |
|---------|-------------------|------------------|
| `fever` | Temperature >38°C | Common across infections |
| `headache` | Central nervous system involvement | Meningitis, dengue, typhoid |
| `nausea` | GI involvement | Hepatitis, cholera, food poisoning |
| `vomiting` | GI distress | Cholera, appendicitis, food poisoning |
| `fatigue` | General malaise | Most infectious diseases |
| `joint_pain` | Arthralgia | Dengue, Chikungunya, arthritis |
| `skin_rash` | Dermatological manifestation | Dengue, measles, varicella |
| `cough` | Respiratory involvement | Tuberculosis, pneumonia, cold |
| `weight_loss` | Chronic condition indicator | TB, chronic liver disease, cancer |
| `yellow_eyes` | Jaundice - bilirubin elevation | Hepatitis B/C, cholestasis |
| `chills_cyclical` | **Targeted**: Malaria indicators | Malaria |
| `night_sweats` | **Targeted**: TB indicator | Tuberculosis |
| `dark_urine` | **Targeted**: Hepatic dysfunction | Hepatitis, hemolysis |
| `positional_dizziness` | **Targeted**: BPPV indicator | Benign Paroxysmal Positional Vertigo |
| `sudden_weakness` | **Targeted**: Neurological emergency | Stroke, paralysis |
| `neck_stiffness` | **Targeted**: Meningitis sign | Meningitis, cervical spondylosis |

#### 2. **Vital Signs** (8 Continuous + 1 Categorical)

| Vital Sign | Normal Range | Clinical Relevance |
|------------|--------------|-------------------|
| `Heart_Rate` | 60-100 bpm | Tachycardia (>100) = stress/infection; Bradycardia (<60) = cardiac issues |
| `Systolic_BP` | 90-120 mmHg | Hypotension (<90) = shock; Hypertension (>140) = stroke risk |
| `Respiratory_Rate` | 12-20 breaths/min | >20 = respiratory distress; <12 = depression |
| `Oxygen_Saturation` | 95-100% | <90% = hypoxemia (critical); <94% = hypoxia |
| `Temperature` | 36.5-37.5°C | >38°C = fever; <36°C = hypothermia |
| `O2_Scale` | 1.0 (room air) | Indicator of supplemental oxygen amount |
| `On_Oxygen` | 0/1 binary | Flag for oxygen therapy requirement |
| `Consciousness_enc` | 0 (Alert) | 0=Alert, 1=Confused, 2=Verbal response, 3=Pain response, 4=Unresponsive (AVPU scale) |

#### 3. **Laboratory Tests** (3 Key Tests)

| Lab Test | Normal Range | Clinical Interpretation |
|----------|--------------|--------------------------|
| `ALT_UL` | 7-56 U/L | Alanine aminotransferase - liver enzyme. Elevated = hepatocyte damage |
| `WBC_K_uL` | 4.5-11.0 K/µL | White blood cell count. High (>11) = infection/leukemia; Low (<4.5) = immunocompromise |
| `TSH_mIUL` | 0.3-4.2 mIU/L | Thyroid stimulating hormone. Low (<0.3) = hyperthyroidism; High (>4.2) = hypothyroidism |

### Target Variables

#### 1. **Disease Classification** (14 Classes)

The dataset contains 14 distinct disease categories across multiple categories:
- **Infectious Diseases**: Malaria, Dengue, Tuberculosis, Typhoid, Pneumonia, Hepatitis B, Hepatitis C, Cholera, Measles, Meningitis
- **Non-Infectious**: Appendicitis, Cervical Spondylosis, BPPV, Chronic Liver Disease, Healthy

#### 2. **Risk Level Classification** (4 Classes)

| Risk Level | Score Range | Clinical Action | Examples |
|-----------|-------------|-----------------|----------|
| **Low** | 0-5 | Monitor at home; Follow up if worse | Normal vitals + mild symptoms |
| **Moderate** | 6-10 | Doctor visit within 3-5 days | Some vital abnormalities |
| **High** | 11-15 | Doctor visit within 24 hours | Multiple abnormal vitals + concerning disease |
| **Critical** | 16-20 | **EMERGENCY ER visit REQUIRED** | Severe hypoxia, hypotension, unconscious |

---

## Feature Engineering Pipeline

### Stage 1: Raw Features from Dataset

**Total raw features**: 25
- 16 symptoms (binary: 0/1)
- 8 vital signs (continuous/categorical)
- 1 consciousness level (categorical → encoded)

### Stage 2: Engineered Features

The pipeline creates 15 additional derived features based on clinical logic:

#### A. **Clinical Ratio Features** (3 features)

```python
Shock_Index = Heart_Rate / (Systolic_BP + 1e-6)
# Clinical: Indicates cardiovascular stability. Shock if > 0.9

SpO2_HR_ratio = Oxygen_Saturation / (Heart_Rate + 1e-6)
# Clinical: Oxygen efficiency. Lower = worse oxygenation relative to heart effort

Temp_deviation = Temperature - 37.0
# Clinical: Deviation from normal body temp (37°C)
```

#### B. **Binary Clinical Flags** (4 features)

```python
Hypoxia_flag = Oxygen_Saturation < 90  # CRITICAL: SpO2 < 90%
Tachycardia_flag = Heart_Rate > 100    # Abnormally elevated heart rate
Fever_flag = Temperature > 38.0         # Fever threshold
Hypotensive_flag = Systolic_BP < 90    # Dangerously low BP = SHOCK
```

#### C. **Composite Severity Score** (1 feature) — NEWS2-Inspired

```python
Severity_Score = (
    Hypoxia_flag * 3 +           # 3 points: critical sign
    Tachycardia_flag * 2 +       # 2 points: significant abnormality
    Fever_flag * 1 +             # 1 point: mild abnormality
    Hypotensive_flag * 3 +       # 3 points: critical sign
    Consciousness_enc * 2        # 2 points per level (alert=0, unresponsive=8)
)
# Range: 0-20 (maps to risk levels: 0-5=Low, 6-10=Mod, 11-15=High, 16-20=Critical)
```

#### D. **Lab-Derived Flags** (3 features)

```python
ALT_log = log(ALT_UL + 1)        # Log-transform skewed lab value
TSH_flag = TSH_mIUL < 0.3       # Suppressed TSH = hyperthyroidism
WBC_high = WBC_K_uL > 11.0      # Elevated WBC = infection/leukemia
WBC_low = WBC_K_uL < 4.5        # Low WBC = immunocompromise/bone marrow issue
```

#### E. **Symptom Aggregate Feature** (1 feature)

```python
Symptom_Count = SUM(16 base symptoms)
# Range: 0-16. Higher = more symptomatic patient
```

#### F. **Consciousness Encoding** (1 feature)

```python
CONSCIOUSNESS_MAP = {
    "A": 0,  # Alert
    "C": 1,  # Confused
    "V": 2,  # Verbal response only
    "P": 3,  # Pain response only
    "U": 4,  # Unresponsive
}
# AVPU scale standard in emergency medicine
```

### Stage 3: Final Feature Set

**Total final features**: 40

| Category | Count | Features |
|----------|-------|----------|
| Original Symptoms | 10 | fever, headache, nausea, vomiting, fatigue, joint_pain, skin_rash, cough, weight_loss, yellow_eyes |
| Targeted Indicators | 6 | chills_cyclical, night_sweats, dark_urine, positional_dizziness, sudden_weakness, neck_stiffness |
| Vital Signs | 8 | Respiratory_Rate, Oxygen_Saturation, O2_Scale, Systolic_BP, Heart_Rate, Temperature, On_Oxygen, Consciousness_enc |
| Lab Tests | 5 | ALT_UL, ALT_log, WBC_K_uL, WBC_high, WBC_low, TSH_mIUL, TSH_flag |
| Engineered Features | 11 | Shock_Index, SpO2_HR_ratio, Temp_deviation, Hypoxia_flag, Tachycardia_flag, Fever_flag, Hypotensive_flag, Severity_Score, Symptom_Count, (+ log/flags) |
| **TOTAL** | **40** | Used for all model predictions |

### Feature Scaling

- **Random Forest & Gradient Boosting**: No scaling (tree-based models are scale-invariant)
- **Logistic Regression**: StandardScaler applied (distance-based algorithm)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Machine Learning Models

### Model Selection & Rationale

The system trains **3 complementary models** and uses the best one for each task:

#### 1. **Random Forest Classifier**

```python
RandomForestClassifier(
    n_estimators=200,           # 200 trees in ensemble
    max_features="sqrt",        # Feature subsampling: sqrt(40) ≈ 6 features per split
    min_samples_leaf=2,         # Minimum samples to form leaf
    n_jobs=-1,                  # Parallel processing on all cores
    random_state=42             # Reproducible results
)
```

**Advantages**:
- Handles non-linear relationships between symptoms and diseases
- Captures feature interactions naturally
- Robust to outliers and missing data
- Provides feature importance rankings
- Works well with mixed data types

**Use Case**: Best for disease classification (complex symptom-disease relationships)

#### 2. **Gradient Boosting Classifier**

```python
GradientBoostingClassifier(
    n_estimators=150,           # 150 boosting rounds
    learning_rate=0.08,         # Learning rate (shrinkage)
    max_depth=5,                # Shallow trees for stability
    subsample=0.8,              # Row subsampling (80%)
    random_state=42
)
```

**Advantages**:
- Sequential learning: each tree corrects previous errors
- Very high accuracy on complex datasets
- Captures subtle patterns in data
- Strong performer on multi-class problems

**Use Case**: Often best for risk level prediction (captures severity nuances)

#### 3. **Logistic Regression**

```python
LogisticRegression(
    max_iter=500,               # Maximum iterations
    C=1.0,                      # Inverse regularization strength
    solver="lbfgs",             # Optimization algorithm
    multi_class="auto",         # Multi-class strategy
    random_state=42
)
```

**Advantages**:
- Fast training and prediction
- Interpretable model (coefficients show feature importance)
- Probabilistic outputs (well-calibrated)
- Low computational requirements
- Good baseline model

**Use Case**: Fast predictions; good for real-time systems; interpretability focus

### Task 1: Disease Classification (14 Classes)

**Input**: 40 features
**Output**: Disease name + confidence score
**Training Strategy**: Stratified 80/20 split to maintain disease distribution

**Metrics**:
- **Accuracy**: Percentage of correct predictions
- **Macro F1-Score**: Average F1 across all 14 disease classes (balances precision/recall)
- **Cohen's Kappa**: Agreement correcting for chance (>0.6 = substantial, >0.8 = excellent)
- **ROC-AUC**: Area under receiver operating characteristic curve (>0.9 = excellent)

### Task 2: Risk Level Classification (4 Classes)

**Input**: 40 features
**Output**: Risk level (Low/Moderate/High/Critical) + confidence score
**Training Strategy**: Separate stratified split to maintain risk distribution

**Special Logic**:
```python
if pred_disease == "Healthy":
    if pred_risk in ["High", "Critical"] or severity_score >= 2:
        pred_disease = "Atypical/Unknown Illness"  # Override: contradiction resolution
    else:
        pred_disease = "Generally Healthy"
```

### Model Evaluation Strategy

#### Cross-Validation

5-fold Stratified K-Fold cross-validation on test set:
- Ensures no data leakage
- Provides robust performance estimates
- Error bars from CV std deviation

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
# Report: mean ± std
```

---

## Model Performance & Results

### Expected Performance Metrics

Based on the training pipeline in `health_ml_clean.py`:

#### Disease Prediction Task

| Model | Accuracy | Macro F1 | Cohen Kappa | ROC-AUC | Status |
|-------|----------|----------|-------------|---------|--------|
| Random Forest | ~87% | ~87% | ~0.86 | ~0.95 | ✓ Top performer |
| Gradient Boosting | ~85% | ~85% | ~0.84 | ~0.93 | Good |
| Logistic Regression | ~79% | ~78% | ~0.77 | ~0.88 | Baseline |

#### Risk Level Prediction Task

| Model | Accuracy | Macro F1 | Cohen Kappa | ROC-AUC | Status |
|-------|----------|----------|-------------|---------|--------|
| Random Forest | ~92% | ~91% | ~0.89 | ~0.97 | Good |
| Gradient Boosting | ~93% | ~92% | ~0.91 | ~0.97 | ✓ Top performer |
| Logistic Regression | ~88% | ~87% | ~0.85 | ~0.93 | Baseline |

### Key Insights

1. **Disease Classification**: Random Forest typically dominates due to complex symptom-disease patterns
2. **Risk Stratification**: Gradient Boosting edges out for capturing nuanced risk factors
3. **Model Selection**: System dynamically chooses best model per task during `init_model.py` execution
4. **Robust Cross-Validation**: 5-fold CV confirms stable performance without overfitting

### Confusion Matrix Analysis

- **High true positive rates**: Correctly identifies most disease cases
- **Low false negative rates**: Minimizes missing actual cases (medical safety priority)
- **Some false positives**: Some healthy patients flagged as at-risk (acceptable: prevents underestimation)

### Feature Importance (Random Forest)

Top 18 features ranked by importance:
1. **Oxygen_Saturation** - Most critical vital sign
2. **Heart_Rate** - Tachycardia indicator
3. **Temperature** - Fever presence/severity
4. **Systolic_BP** - Hemodynamic stability
5. **Severity_Score** - Composite risk indicator
6. **Consciousness_enc** - LOC = serious sign
7. **WBC_K_uL** - Infection/immune markers
8. **Shock_Index** - Hemodynamic reserve
9. **fever** - Most common symptom
10-18. Various targeted symptoms and lab flags

---

## Installation & Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|------------|
| Python | 3.9 | 3.10+ |
| Node.js | 16 LTS | 18+ LTS |
| npm | 8+ | 9+ |
| RAM | 2 GB | 4 GB+ |
| Disk Space | 1 GB | 2 GB+ |
| OS | Linux/macOS/Windows WSL | Linux/macOS |

### Prerequisites Checklist

- [ ] Python 3.9+ installed and in PATH
- [ ] Node.js 16+ installed and in PATH
- [ ] npm 8+ installed
- [ ] Git installed (for version control)
- [ ] `professional_dataset_16k.csv` in `my_health_app/` directory
- [ ] ~500 MB free disk space (for models + dependencies)

### Verify Prerequisites

```bash
# Check Python
python --version          # Should be 3.9+

# Check Node.js and npm
node --version            # Should be 16+
npm --version             # Should be 8+

# Check git
git --version             # Should show version
```

---

## Local Development Setup

### Step 1: Clone and Navigate

```bash
cd /path/to/HealthDiseaseAndRiskPrediction
ls -la  # Verify project structure
```

### Step 2: Prepare the Dataset

```bash
# Ensure dataset exists
ls -la my_health_app/professional_dataset_16k.csv

# Should show file with size ~50+ MB
# If missing, download or generate synthetic data
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate      # Windows

# Verify activation (prompt should show (.venv))
```

### Step 4: Install Python Dependencies

```bash
cd my_health_app/backend

# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "pandas|scikit-learn|joblib"
```

### Step 5: Generate & Train Models

```bash
# From my_health_app/backend directory
python init_model.py

# Output should show:
# - Loading 16,000 patient dataset
# - Feature engineering: 40 features
# - Training 3 models
# - Performance metrics (Accuracy, F1, AUC)
# - Saving .pkl files to backend/models/

# This may take 2-5 minutes depending on hardware
# Models are saved to backend/models/:
ls -la models/
# - disease_model.pkl
# - risk_model.pkl
# - scaler.pkl
# - le_disease.pkl
# - le_risk.pkl
# - config.pkl
```

### Step 6: Build Frontend

```bash
# From my_health_app/frontend directory
cd ../frontend

# Install Node dependencies
npm install

# Build for production
npm run build

# Verify build (creates dist/ folder)
ls -la dist/
```

### Step 7: Start the Application

```bash
# From my_health_app/backend directory
cd ../backend

# Start Express server + Python ML engine
node server.js

# Output should show:
# [INFO] Server running at http://localhost:8000
```

### Step 8: Access the Application

Open your browser:
- **Web App**: http://localhost:8000
- **API Docs** (if added): http://localhost:8000/api/docs

---

## Project Structure

```
HealthDiseaseAndRiskPrediction/
│
├── README.md                           # Original brief README
├── README_DETAILED.md                  # This comprehensive guide
├── health_ml_clean.py                  # Main ML training pipeline (standalone)
│
└── my_health_app/                      # Main application root
    ├── professional_dataset_16k.csv    # Dataset (16k patient records)
    ├── model_code.py                   # ML model training code
    │
    ├── backend/                        # Python + Node.js backend
    │   ├── server.js                   # Express.js HTTP server
    │   ├── package.json                # Node dependencies
    │   ├── requirements.txt            # Python dependencies
    │   ├── init_model.py               # Model training & saving script
    │   ├── predict_cli.py              # CLI prediction interface
    │   ├── model_wrapper.py            # Wrapper for model loading/prediction
    │   ├── model_summary.csv           # Training performance summary
    │   │
    │   └── models/                     # Pre-trained model files (generated)
    │       ├── disease_model.pkl       # Disease classifier
    │       ├── risk_model.pkl          # Risk level classifier
    │       ├── scaler.pkl              # StandardScaler
    │       ├── le_disease.pkl          # Disease label encoder
    │       ├── le_risk.pkl             # Risk label encoder
    │       └── config.pkl              # Model metadata & config
    │
    ├── frontend/                       # React + Vite frontend
    │   ├── package.json                # React dependencies
    │   ├── vite.config.js              # Vite build config
    │   ├── eslint.config.js            # ESLint configuration
    │   ├── index.html                  # HTML entry point
    │   │
    │   ├── src/
    │   │   ├── main.jsx                # React DOM render
    │   │   ├── App.jsx                 # Main app component
    │   │   ├── App.css                 # Global styles
    │   │   ├── index.css               # Base styles
    │   │   │
    │   │   ├── components/
    │   │   │   ├── Home.jsx            # Home page
    │   │   │   ├── PredictionForm.jsx  # Prediction input form
    │   │   │   ├── ResultModal.jsx     # Results display
    │   │   │   ├── PatientHistory.jsx  # History page
    │   │   │   ├── About.jsx           # About page
    │   │   │   ├── Portal.jsx          # Doctor dashboard
    │   │   │   └── Contact.jsx         # Contact page
    │   │   │
    │   │   └── assets/                 # Images, icons, etc.
    │   │
    │   ├── public/                     # Static files (if any)
    │   │
    │   └── dist/                       # Build output (generated)
    │       ├── index.html
    │       └── assets/
    │
    └── __pycache__/                    # Python cache (generated)
```

### License

**MIT License** — See LICENSE file for details

```
MIT License

Copyright (c) 2026 Aura Health Contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
```


**Status**: Actively Maintained
**License**: MIT

