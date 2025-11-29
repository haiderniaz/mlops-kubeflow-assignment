# Kubeflow Pipeline Components - Documentation

## Overview
This document explains the inputs and outputs defined for each pipeline component.

## Component Definitions

### 1. Data Extraction Component

**Purpose:** Fetches versioned dataset from DVC remote storage

**Inputs:**
- `dvc_remote_url` (str): URL or path to DVC remote storage
- `output_data_path` (str): Path where extracted data will be saved

**Outputs:**
- Returns `str`: Path to the extracted dataset file

**Description:**
This component retrieves the versioned dataset from DVC remote storage. It ensures data consistency across pipeline runs by fetching the exact version of data tracked by DVC.

---

### 2. Data Preprocessing Component

**Purpose:** Handles data cleaning, scaling, and train/test splitting

**Inputs:**
- `input_data_path` (str): Path to raw input data CSV file
- `train_data_path` (str): Path to save processed training data
- `test_data_path` (str): Path to save processed test data
- `test_size` (float): Proportion of data for testing (default: 0.2)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Outputs:**
- Returns `dict` containing:
  - `train_path` (str): Path to training data
  - `test_path` (str): Path to test data
  - `train_samples` (int): Number of training samples
  - `test_samples` (int): Number of test samples
  - `feature_count` (int): Number of features

**Processing Steps:**
1. Remove duplicate rows
2. Handle missing values (dropna)
3. Separate features (X) and target (y)
4. Split data into train/test sets
5. Apply StandardScaler for feature normalization
6. Save processed datasets

---

### 3. Model Training Component ⭐

**Purpose:** Trains a Random Forest classifier and saves model artifact

#### **INPUTS:**

1. **train_data_path** (str) - **Required**
   - Description: Path to the training data CSV file
   - Type: STRING parameter
   - Example: `/data/processed/train.csv`
   - Purpose: Specifies location of preprocessed training data

2. **model_output_path** (str) - **Required**
   - Description: Path where the trained model will be saved
   - Type: STRING parameter
   - Format: `.joblib` file
   - Example: `/models/random_forest_model.joblib`
   - Purpose: Defines output location for model artifact

3. **n_estimators** (int) - **Optional**
   - Description: Number of trees in the Random Forest
   - Type: NUMBER_INTEGER parameter
   - Default: 100
   - Range: Typically 10-500
   - Purpose: Controls model complexity and training time

4. **max_depth** (int) - **Optional**
   - Description: Maximum depth of each decision tree
   - Type: NUMBER_INTEGER parameter
   - Default: 10
   - Range: Typically 5-30
   - Purpose: Controls overfitting by limiting tree depth

5. **random_state** (int) - **Optional**
   - Description: Seed for random number generator
   - Type: NUMBER_INTEGER parameter
   - Default: 42
   - Purpose: Ensures reproducibility across runs

#### **OUTPUTS:**

Returns `dict` (STRUCT parameter type) containing:

1. **model_path** (str)
   - Path to the saved model file
   - Used by evaluation component to load model

2. **train_rmse** (float)
   - Root Mean Squared Error on training set
   - Measures prediction accuracy during training

3. **train_mae** (float)
   - Mean Absolute Error on training set
   - Average magnitude of prediction errors

4. **train_r2** (float)
   - R² (R-squared) score on training set
   - Proportion of variance explained by model
   - Range: 0 to 1 (higher is better)

5. **n_estimators** (int)
   - Echoes input parameter for tracking

6. **max_depth** (int)
   - Echoes input parameter for tracking

7. **feature_count** (int)
   - Number of features used in training
   - For validation and debugging

#### **Training Process:**
1. Load training data from CSV
2. Separate features (X) and target (y - MEDV)
3. Initialize RandomForestRegressor with specified parameters
4. Fit model on training data
5. Calculate training metrics (RMSE, MAE, R²)
6. Extract feature importances
7. Save model using joblib
8. Return metrics and metadata

#### **Why These Inputs/Outputs?**

**Inputs Rationale:**
- **train_data_path**: Essential - tells component where to find data
- **model_output_path**: Essential - defines where to persist trained model
- **Hyperparameters** (n_estimators, max_depth): Allow tuning without code changes
- **random_state**: Ensures reproducible results for testing and validation

**Outputs Rationale:**
- **model_path**: Next component (evaluation) needs to load this model
- **Training metrics**: Monitor training performance and detect overfitting
- **Metadata**: Track experiment configuration for MLOps best practices
- **feature_count**: Validate data consistency across pipeline runs

---

### 4. Model Evaluation Component

**Purpose:** Evaluates trained model on test set and saves metrics

**Inputs:**
- `model_path` (str): Path to trained model file (.joblib)
- `test_data_path` (str): Path to test data CSV file
- `metrics_output_path` (str): Path to save evaluation metrics JSON

**Outputs:**
- Returns `dict` containing:
  - `rmse` (float): Root Mean Squared Error
  - `mae` (float): Mean Absolute Error
  - `r2_score` (float): R² Score
  - `mape` (float): Mean Absolute Percentage Error
  - `mse` (float): Mean Squared Error
  - `test_samples` (int): Number of test samples
  - `predictions_mean` (float): Mean of predictions
  - `predictions_std` (float): Std dev of predictions
  - `actual_mean` (float): Mean of actual values
  - `actual_std` (float): Std dev of actual values

**Evaluation Metrics:**
1. Load trained model from joblib file
2. Make predictions on test set
3. Calculate comprehensive evaluation metrics
4. Save metrics to JSON file
5. Return metrics dictionary

---

## Pipeline Flow

```
1. Data Extraction → Raw Dataset
2. Data Preprocessing → Train/Test Datasets (scaled)
3. Model Training → Trained Model + Training Metrics
4. Model Evaluation → Evaluation Metrics + JSON Report
```

## Key Design Decisions

1. **Component Isolation**: Each component is self-contained with clear inputs/outputs
2. **Reproducibility**: Random seeds and version control ensure consistent results
3. **Scalability**: Components can run in containers with specified dependencies
4. **Monitoring**: Comprehensive metrics at each stage for observability
5. **Flexibility**: Hyperparameters exposed as inputs for easy tuning

## Usage Example

```python
from kfp import dsl

@dsl.pipeline(name='boston-housing-pipeline')
def ml_pipeline():
    # Step 1: Extract data
    extract_task = data_extraction_component(
        dvc_remote_url='../dvc_remote',
        output_data_path='/data/raw/boston.csv'
    )
    
    # Step 2: Preprocess data
    preprocess_task = data_preprocessing_component(
        input_data_path=extract_task.output,
        train_data_path='/data/processed/train.csv',
        test_data_path='/data/processed/test.csv'
    )
    
    # Step 3: Train model
    train_task = model_training_component(
        train_data_path=preprocess_task.outputs['train_path'],
        model_output_path='/models/rf_model.joblib',
        n_estimators=100,
        max_depth=10
    )
    
    # Step 4: Evaluate model
    eval_task = model_evaluation_component(
        model_path=train_task.outputs['model_path'],
        test_data_path=preprocess_task.outputs['test_path'],
        metrics_output_path='/metrics/evaluation.json'
    )
```

## Component YAML Files

All components have been compiled to YAML format:
- `data_extraction_component.yaml`
- `data_preprocessing_component.yaml`
- `model_training_component.yaml`
- `model_evaluation_component.yaml`

These YAML files can be used directly in Kubeflow Pipelines SDK v2.
