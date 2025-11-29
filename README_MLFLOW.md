# Boston Housing Price Prediction with MLflow

Complete ML pipeline for predicting Boston housing prices using Random Forest with MLflow experiment tracking.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline with:
- **Data Version Control (DVC)** for dataset management
- **MLflow** for experiment tracking, model logging, and metrics visualization
- **Random Forest Regressor** for price prediction
- **Modular pipeline** with reusable components

## 🏗️ Project Structure

```
mlops-kubeflow-assignment/
├── data/
│   ├── raw/                      # Raw dataset
│   └── processed/                # Preprocessed train/test splits
├── models/                       # Trained model files
├── results/                      # Evaluation metrics and outputs
├── src/
│   └── mlflow_pipeline_components.py  # Pipeline components with MLflow tracking
├── run_pipeline.py              # Main pipeline execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run with default parameters
python run_pipeline.py

# Run with custom parameters
python run_pipeline.py --test-size 0.3 --n-estimators 200 --max-depth 15
```

### 3. View Results in MLflow UI

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

## 📊 Pipeline Components

### 1. Data Extraction
- Downloads Boston Housing dataset
- Saves raw data to `data/raw/boston.csv`
- Logs dataset metadata to MLflow

### 2. Data Preprocessing
- Splits data into train/test sets
- Applies StandardScaler normalization
- Saves processed data to `data/processed/`
- Logs preprocessing parameters to MLflow

### 3. Model Training
- Trains Random Forest Regressor
- Logs model parameters and training metrics
- Saves model to `models/` and MLflow

### 4. Model Evaluation
- Evaluates model on test set
- Calculates RMSE, MAE, R², MSE, MAPE
- Logs metrics and artifacts to MLflow
- Saves detailed report to `results/metrics.txt`

## 🎯 MLflow Features

### Experiment Tracking
- All pipeline runs tracked automatically
- Parameters, metrics, and artifacts logged
- Compare multiple runs in UI

### Model Registry
- Models logged with sklearn flavor
- Easy model deployment and versioning
- Model artifacts stored with metadata

### Visualization
- Metric plots across runs
- Parameter importance analysis
- Run comparison charts

## 📈 Expected Results

Typical model performance:
- **RMSE**: ~3.5-4.5 (thousands of dollars)
- **MAE**: ~2.5-3.5
- **R² Score**: ~0.75-0.85
- **MAPE**: ~15-20%

## 🔧 Configuration Options

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test-size` | 0.2 | Proportion of test set (0.0-1.0) |
| `--n-estimators` | 100 | Number of trees in forest |
| `--max-depth` | 10 | Maximum tree depth |
| `--random-state` | 42 | Random seed for reproducibility |
| `--experiment-name` | "Boston Housing Prediction" | MLflow experiment name |

### Example: Run Multiple Experiments

```bash
# Experiment 1: Baseline
python run_pipeline.py --n-estimators 50 --max-depth 5

# Experiment 2: More trees
python run_pipeline.py --n-estimators 200 --max-depth 10

# Experiment 3: Deeper trees
python run_pipeline.py --n-estimators 100 --max-depth 20
```

## 📸 Screenshots for Deliverable

1. **Pipeline Execution**: Terminal output showing all steps
2. **MLflow Experiments**: Main experiments page
3. **Run Details**: Individual run with parameters and metrics
4. **Model Artifacts**: Logged model and files
5. **Metrics Comparison**: Chart comparing multiple runs

## 🐛 Troubleshooting

### Port Already in Use
```bash
# MLflow UI port 5000 busy
mlflow ui --port 5001
```

### Module Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Data Download Issues
- Check internet connection
- Dataset URL: http://lib.stat.cmu.edu/datasets/boston
- Alternative: Use pre-downloaded data in `data/raw/`

## 📚 Dataset Information

**Boston Housing Dataset**
- 506 samples
- 13 features (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- Target: MEDV (Median home value in $1000s)

## 🔗 Technologies Used

- **Python 3.11**
- **MLflow 2.9+** - Experiment tracking and model management
- **scikit-learn** - Machine learning algorithms
- **pandas & numpy** - Data manipulation
- **DVC 3.0+** - Data version control
- **joblib** - Model serialization

## 📝 Assignment Deliverables

✅ **Task 1**: Project structure with DVC for data versioning  
✅ **Task 2**: Modular pipeline components with clear separation  
✅ **Task 3**: Complete pipeline with MLflow experiment tracking  
✅ **Task 4**: Documentation and reproducible execution  

## 🎓 Learning Outcomes

- MLOps best practices with MLflow
- Reproducible ML pipelines
- Experiment tracking and model versioning
- Data version control with DVC
- Model evaluation and metrics logging

## 📄 License

This project is for educational purposes (MLOps course assignment).

## 👤 Author

Haider Niaz  
MLOps Assignment - Semester 7
