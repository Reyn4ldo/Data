# AMR Machine Learning Pipeline

A comprehensive machine learning pipeline for antimicrobial resistance (AMR) prediction using bacterial isolate data from environmental and fish samples.

## Overview

This pipeline performs multi-class and binary classification to predict:
1. **Bacterial Species** - Classify isolates by bacterial species
2. **Resistance Category** - Predict Low/Moderate/High resistance based on MAR index
3. **Species from AST Only** - Predict species using only antibiotic susceptibility test (AST) patterns
4. **MDR Status** - Binary prediction of Multi-Drug Resistant status with probability scores

## Project Structure

```
├── data/
│   ├── raw_data.csv              # Original bacterial isolate data
│   └── processed_data.csv        # Cleaned and engineered features
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_preprocessing.py     # Data cleaning and preparation
│   ├── feature_engineering.py    # Feature creation and encoding
│   ├── model_training.py         # ML model training with cross-validation
│   ├── model_evaluation.py       # Model evaluation and comparison
│   ├── model_interpretation.py   # SHAP values and feature importance
│   └── utils.py                  # Utility functions
├── models/                       # Saved trained models
├── results/
│   ├── figures/                  # Visualization plots
│   └── metrics/                  # Evaluation metrics (JSON, CSV)
├── notebooks/                    # Optional Jupyter notebooks
├── main.py                       # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Data
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

The pipeline will:
1. Load and preprocess the raw data
2. Engineer features (encode S/I/R values, create one-hot encodings, etc.)
3. Train 6 different ML algorithms for each task
4. Perform 5-fold group-stratified cross-validation
5. Evaluate models and select the best performer
6. Generate visualizations (confusion matrices, ROC curves, SHAP plots)
7. Save trained models and metrics

## Input Data

The input data (`data/raw_data.csv`) contains:
- **Organism**: Bacterial species identification
- **Isolate_Code**: Unique identifier for grouping replicates
- **Location data**: National_Site, Local_Site, Sample_Source
- **ESBL status**: Extended-spectrum beta-lactamase production
- **Antibiotic susceptibility**: MIC values and interpretations (S/I/R) for ~22 antibiotics
- **Resistance metrics**: Scored_Resistance, MAR_INDEX

## Machine Learning Tasks

### Task 1: Species Classification
- **Target**: Bacterial species (multi-class)
- **Features**: All available features including AST patterns, sample source, location

### Task 2: Resistance Category Classification  
- **Target**: Resistance category (Low/Moderate/High)
- **Categories**:
  - Low: MAR index < 0.2
  - Moderate: 0.2 ≤ MAR index < 0.4
  - High: MAR index ≥ 0.4

### Task 3: Species from AST Only
- **Target**: Bacterial species
- **Features**: Only AST interpretation columns (S/I/R patterns)

### Task 4: MDR Prediction
- **Target**: MDR (Multi-Drug Resistant) vs Non-MDR
- **MDR Definition**: Resistance to ≥3 antibiotic classes OR MAR index ≥ 0.2
- **Output**: MDR probability (0-1) and binary classification

## Algorithms

The pipeline trains and evaluates 6 different algorithms:
1. Random Forest
2. LightGBM (Gradient Boosting)
3. Logistic Regression
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes

## Evaluation Metrics

### Multi-class Tasks
- Accuracy
- Balanced Accuracy
- Macro F1-score (primary metric)
- Weighted F1-score
- Macro Precision/Recall
- Macro ROC-AUC (One-vs-Rest)
- Confusion Matrix

### Binary MDR Task
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- PR-AUC (Precision-Recall AUC)
- Confusion Matrix
- Probability Calibration Curves

## Model Selection Criteria

Best model is selected based on:
1. Highest macro F1 score
2. Highest ROC-AUC
3. Lowest overfitting gap (train vs validation)
4. Stability across CV folds

## Outputs

After running the pipeline:

### Data
- `data/processed_data.csv`: Cleaned and engineered dataset

### Models
Saved in `models/` directory:
- `{task}_{model_name}.joblib`: Trained models
- `{task}_scaler.joblib`: Feature scalers
- `{task}_label_encoder.joblib`: Label encoders

### Metrics
Saved in `results/metrics/`:
- `{task}_metrics.json`: Detailed metrics for all models
- `{task}_comparison.csv`: Model comparison table
- `{task}_best_model.json`: Best model selection info

### Figures
Saved in `results/figures/`:
- `{task}_{model}_confusion.png`: Confusion matrices
- `{task}_roc_curves.png`: ROC curves for all models
- `{task}_{model}_importance.png`: Feature importance plots
- `{task}_{model}_shap_bar.png`: SHAP value plots
- `{task}_{model}_calibration.png`: Calibration curves (MDR only)

## Antibiotic Classes

The pipeline tracks resistance patterns across antibiotic classes:
- **Penicillins**: Ampicillin
- **Beta-lactam combinations**: Amoxicillin/Clavulanic acid
- **Cephalosporins**: Ceftaroline, Cefalexin, Cefalotin, Cefpodoxime, Cefotaxime, Cefovecin, Ceftiofur, Ceftazidime/Avibactam
- **Carbapenems**: Imipenem
- **Aminoglycosides**: Amikacin, Gentamicin, Neomycin
- **Quinolones**: Nalidixic acid, Enrofloxacin, Marbofloxacin, Pradofloxacin
- **Tetracyclines**: Doxycycline, Tetracycline
- **Nitrofurans**: Nitrofurantoin
- **Phenicols**: Chloramphenicol
- **Folate pathway inhibitors**: Trimethoprim/Sulfamethazole

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- shap >= 0.44.0
- matplotlib >= 3.7.0
- seaborn >= 0.13.0
- imbalanced-learn >= 0.11.0
- joblib >= 1.3.0

## License

This project is for research purposes only.

## Author

AMR ML Pipeline Development Team
