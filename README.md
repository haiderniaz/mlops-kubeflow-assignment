# MLOps Pipeline - Boston Housing Price Prediction

## 📋 Project Overview

This project implements a complete MLOps pipeline for predicting Boston housing prices using machine learning. The pipeline demonstrates industry-standard practices including:

- **ML Problem**: Regression task to predict median house values based on 13 features (crime rate, number of rooms, proximity to employment centers, etc.)
- **Dataset**: Boston Housing Dataset (506 samples, 13 features)
- **Model**: Random Forest Regressor with hyperparameter tuning
- **MLOps Tools**: MLflow for experiment tracking, DVC for data versioning, GitHub Actions for CI/CD

### 🎯 Key Features

- ✅ **Experiment Tracking**: MLflow for logging parameters, metrics, and models
- ✅ **Data Versioning**: DVC for tracking dataset versions
- ✅ **CI/CD Pipeline**: Automated testing and validation with GitHub Actions
- ✅ **Reproducibility**: Parameterized pipeline with configurable hyperparameters
- ✅ **Model Registry**: Automatic model logging and versioning

### 📊 Model Performance

Current best model performance:
- **RMSE**: 2.79
- **MAE**: 2.02
- **R² Score**: 0.894
- **MAPE**: 10.99%

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- Git
- GitHub account
- DVC (Data Version Control)

### 1. Clone the Repository

```bash
git clone https://github.com/haiderniaz/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
- `mlflow>=2.9.0` - Experiment tracking and model registry
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `dvc>=3.0.0` - Data version control
- `joblib>=1.3.0` - Model serialization

### 4. Configure DVC Remote Storage (Optional)

If you want to use DVC for data versioning with remote storage:

```bash
# Initialize DVC (already done in this project)
dvc init

# Configure remote storage (example with Google Drive)
dvc remote add -d storage gdrive://YOUR_FOLDER_ID

# Or use local remote for testing
dvc remote add -d storage /path/to/local/storage

# Pull data from remote
dvc pull
```

### 5. Verify Installation

```bash
# Check Python version
python --version

# Verify key packages
pip list | grep -E "(mlflow|scikit-learn|dvc)"

# Test import
python -c "import mlflow, sklearn, pandas, numpy, dvc; print('✓ All packages installed successfully')"
```

---

## 📦 Project Structure

```
mlops-kubeflow-assignment/
│
├── .github/
│   └── workflows/
│       └── pipeline-ci.yml          # GitHub Actions CI/CD workflow
│
├── data/
│   ├── raw/                         # Raw data from data extraction
│   │   └── boston.csv
│   └── processed/                   # Processed train/test splits
│       ├── train.csv
│       └── test.csv
│
├── src/
│   └── mlflow_pipeline_components.py  # All 4 pipeline components
│
├── models/
│   └── random_forest_model.joblib   # Trained model artifacts
│
├── results/
│   └── metrics.txt                  # Evaluation metrics
│
├── mlruns/                          # MLflow tracking data (auto-generated)
│
├── run_pipeline.py                  # Main pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── Jenkinsfile                      # Jenkins CI/CD pipeline (alternative)
├── CI_CD_DOCUMENTATION.md          # CI/CD setup guide
├── README_MLFLOW.md                # Detailed MLflow documentation
├── .gitignore                      # Git ignore rules
├── .dvcignore                      # DVC ignore rules
└── README.md                       # This file
```

---

## 🔄 Pipeline Walkthrough

### Pipeline Architecture

The ML pipeline consists of 4 stages, each implemented as a Python function with MLflow tracking:

```
┌─────────────────────┐
│ 1. Data Extraction  │ → Downloads Boston Housing dataset
│                     │   Logs: data_source, dataset_size
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 2. Preprocessing    │ → Train/test split, feature scaling
│                     │   Logs: test_size, scaler, train/test samples
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 3. Model Training   │ → Random Forest training
│                     │   Logs: n_estimators, max_depth, model, train metrics
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 4. Model Evaluation │ → Calculate RMSE, MAE, R², MAPE
│                     │   Logs: all metrics, metrics.txt artifact
└─────────────────────┘
```

### Running the Pipeline

#### Basic Execution

```bash
# Run with default parameters
python run_pipeline.py
```

This will execute the full pipeline with default settings:
- `test_size=0.2` (20% of data for testing)
- `n_estimators=100` (100 trees in Random Forest)
- `max_depth=10` (maximum tree depth)
- `random_state=42` (for reproducibility)

#### Custom Parameters

```bash
# Run with custom hyperparameters
python run_pipeline.py \
  --experiment-name "Boston Housing ML Pipeline" \
  --test-size 0.3 \
  --n-estimators 200 \
  --max-depth 15 \
  --random-state 42
```

**Available Parameters:**

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `--experiment-name` | MLflow experiment name | "Boston Housing Prediction" | string |
| `--test-size` | Test set proportion | 0.2 | float (0-1) |
| `--n-estimators` | Number of trees in Random Forest | 100 | int |
| `--max-depth` | Maximum tree depth | 10 | int |
| `--random-state` | Random seed for reproducibility | 42 | int |

#### Example Experiments

```bash
# Experiment 1: More trees, deeper trees
python run_pipeline.py --n-estimators 200 --max-depth 20

# Experiment 2: Smaller model for faster training
python run_pipeline.py --n-estimators 50 --max-depth 5

# Experiment 3: Different train/test split
python run_pipeline.py --test-size 0.3

# Experiment 4: Quick test run
python run_pipeline.py --n-estimators 10 --max-depth 3 --experiment-name "Quick Test"
```

### Viewing Results in MLflow UI

1. **Start MLflow UI:**
```bash
mlflow ui
```

2. **Access the UI:**
   - Open browser to `http://127.0.0.1:5000`
   - Or use custom port: `mlflow ui --port 8080`

3. **Navigate the UI:**
   - **Experiments Tab**: View all experiments and runs
   - **Run Details**: Click any run to see parameters, metrics, artifacts
   - **Compare Runs**: Select multiple runs to compare performance
   - **Model Registry**: View registered models and versions

4. **Key Features:**
   - 📊 **Metrics Visualization**: Interactive charts for RMSE, MAE, R²
   - 🔍 **Run Comparison**: Side-by-side parameter and metric comparison
   - 📁 **Artifact Browser**: Download models, metrics, and visualizations
   - 🏷️ **Tags & Notes**: Add metadata to organize experiments

---

## 🤖 CI/CD Pipeline

### GitHub Actions Workflow

The project includes an automated CI/CD pipeline with 4 stages:

#### **Stage 1: Environment Setup**
- Checkout code
- Set up Python 3.11
- Cache dependencies
- Install requirements
- Verify installation

#### **Stage 2: Pipeline Validation**
- Syntax check all Python files
- Validate component imports
- Check pipeline structure

#### **Stage 3: Pipeline Execution Test**
- Run pipeline with test parameters
- Verify outputs (data, model, metrics)
- Check MLflow tracking

#### **Stage 4: Report Generation**
- Generate CI summary
- Upload artifacts (model, metrics)

### Triggering the CI/CD Pipeline

The workflow automatically runs on:
- **Push to main branch**
- **Pull requests to main**
- **Manual trigger** (workflow_dispatch)

### Viewing CI/CD Results

1. Go to: `https://github.com/haiderniaz/mlops-kubeflow-assignment/actions`
2. Click on the latest workflow run
3. View logs for each stage
4. Download artifacts from the "Artifacts" section

### Jenkins Alternative

For Jenkins setup, refer to `CI_CD_DOCUMENTATION.md` and use the provided `Jenkinsfile`.

---

## 📊 Data Versioning with DVC

### Current Data Tracking

The Boston Housing dataset is tracked with DVC:

```bash
# View DVC-tracked files
dvc list . data

# Check data status
dvc status

# Pull latest data version
dvc pull

# Push new data version (if you modify data)
dvc add data/boston.csv
git add data/boston.csv.dvc .gitignore
git commit -m "Update dataset"
dvc push
```

### DVC Remote Setup

```bash
# View current remote
dvc remote list

# Add new remote (Google Drive example)
dvc remote add -d mystorage gdrive://FOLDER_ID

# Configure remote
dvc remote modify mystorage gdrive_acknowledge_abuse true

# Push data to remote
dvc push -r mystorage
```

---

## 🧪 Running Tests

### Manual Testing

```bash
# Test individual components
python -c "from src.mlflow_pipeline_components import data_extraction_component; data_extraction_component()"

# Test full pipeline
python run_pipeline.py --n-estimators 10 --max-depth 3
```

### Automated Testing (CI/CD)

Tests run automatically on every push via GitHub Actions. To run locally:

```bash
# Syntax validation
python -m py_compile src/mlflow_pipeline_components.py
python -m py_compile run_pipeline.py

# Import validation
python -c "from src.mlflow_pipeline_components import *; print('✓ All imports successful')"

# Pipeline execution test
python run_pipeline.py --n-estimators 10 --max-depth 3 --experiment-name "CI-Test"
```

---

## 📝 Pipeline Components Details

### 1. Data Extraction Component
**File**: `src/mlflow_pipeline_components.py:data_extraction_component()`

**Purpose**: Downloads and prepares the Boston Housing dataset

**MLflow Logging**:
- `data_source`: "sklearn.datasets.load_boston"
- `dataset_size`: Number of samples
- `feature_names`: List of features

**Outputs**: `data/raw/boston.csv`

---

### 2. Data Preprocessing Component
**File**: `src/mlflow_pipeline_components.py:data_preprocessing_component()`

**Purpose**: Splits data and applies feature scaling

**MLflow Logging**:
- `test_size`: Proportion of test data
- `train_samples`, `test_samples`: Dataset sizes
- `scaling_method`: "StandardScaler"
- `scaler` artifact: Fitted scaler for inference

**Outputs**: 
- `data/processed/train.csv`
- `data/processed/test.csv`

---

### 3. Model Training Component
**File**: `src/mlflow_pipeline_components.py:model_training_component()`

**Purpose**: Trains Random Forest Regressor

**MLflow Logging**:
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `random_state`: Random seed
- `train_rmse`, `train_r2`: Training metrics
- Model artifact: Trained Random Forest

**Outputs**: `models/random_forest_model.joblib`

---

### 4. Model Evaluation Component
**File**: `src/mlflow_pipeline_components.py:model_evaluation_component()`

**Purpose**: Evaluates model on test set

**MLflow Logging**:
- `test_rmse`: Root Mean Squared Error
- `test_mae`: Mean Absolute Error
- `test_r2`: R² Score
- `test_mse`: Mean Squared Error
- `test_mape`: Mean Absolute Percentage Error
- Metrics artifact: `results/metrics.txt`

**Outputs**: `results/metrics.txt`

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **MLflow UI not starting**
```bash
# Kill existing MLflow processes
Get-Process | Where-Object {$_.ProcessName -like "*mlflow*"} | Stop-Process -Force

# Restart MLflow UI
mlflow ui
```

#### 2. **DVC remote not configured**
```bash
# Set up local remote for testing
dvc remote add -d local_storage .dvc/storage
dvc push
```

#### 3. **Import errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python -c "import mlflow; print(mlflow.__version__)"
```

#### 4. **Pipeline execution fails**
```bash
# Check Python environment
python --version
pip list

# Run with verbose logging
python run_pipeline.py --n-estimators 10 --max-depth 3
```

#### 5. **GitHub Actions workflow fails**
- Check the Actions tab on GitHub
- Review error logs for each stage
- Common fixes:
  - Update deprecated actions (already done: `upload-artifact@v4`)
  - Check Python version compatibility
  - Verify `requirements.txt` is complete

---

## 📚 Additional Documentation

- **`README_MLFLOW.md`**: Detailed MLflow usage and features
- **`CI_CD_DOCUMENTATION.md`**: Complete CI/CD setup guide
- **`Jenkinsfile`**: Jenkins pipeline configuration
- **`.github/workflows/pipeline-ci.yml`**: GitHub Actions workflow

---

## 🎓 Assignment Tasks Completed

✅ **Task 1**: Project structure and DVC setup  
✅ **Task 2**: Pipeline components with MLflow tracking  
✅ **Task 3**: Pipeline orchestration and execution  
✅ **Task 4**: CI/CD with GitHub Actions and Jenkins  
✅ **Task 5**: Comprehensive documentation (this README)  

---

## 📞 Repository Information

- **GitHub Repository**: [https://github.com/haiderniaz/mlops-kubeflow-assignment](https://github.com/haiderniaz/mlops-kubeflow-assignment)
- **Owner**: haiderniaz
- **Branch**: main
- **Last Updated**: November 29, 2025

---

## 🙏 Acknowledgments

- **Dataset**: Boston Housing Dataset from scikit-learn
- **MLOps Tools**: MLflow, DVC, GitHub Actions
- **ML Framework**: scikit-learn

---

## 📄 License

This project is for educational purposes as part of an MLOps course assignment.

---

## 🚀 Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/haiderniaz/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Run pipeline
python run_pipeline.py

# 3. View results
mlflow ui
# Open http://127.0.0.1:5000

# 4. Experiment with parameters
python run_pipeline.py --n-estimators 200 --max-depth 15
```

---

**Built with ❤️ for MLOps Assignment**
