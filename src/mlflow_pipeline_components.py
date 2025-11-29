"""
MLflow Pipeline Components for Boston Housing Price Prediction
This module contains pipeline components with MLflow tracking.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn


def data_extraction_component(output_data_path='data/raw/boston.csv'):
    """
    Data Extraction Component
    
    Fetches the Boston Housing dataset from the source.
    
    Args:
        output_data_path: Path where extracted data will be saved
        
    Returns:
        str: Path to the extracted dataset
    """
    print("="*60)
    print("STEP 1: DATA EXTRACTION")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    
    # Download Boston Housing dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    
    try:
        print(f"Downloading data from: {data_url}")
        raw_df = pd.read_csv(data_url, sep=r'\s+', skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names)
        df['MEDV'] = target
        
        # Save to CSV
        df.to_csv(output_data_path, index=False)
        
        print(f"✓ Data extracted successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Saved to: {output_data_path}")
        
        # Log to MLflow
        mlflow.log_param("data_source", data_url)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("n_features", len(feature_names))
        
        return output_data_path
        
    except Exception as e:
        print(f"✗ Error during data extraction: {str(e)}")
        raise


def data_preprocessing_component(
    input_data_path='data/raw/boston.csv',
    train_data_path='data/processed/train.csv',
    test_data_path='data/processed/test.csv',
    test_size=0.2,
    random_state=42
):
    """
    Data Preprocessing Component
    
    Splits data into train/test sets and applies feature scaling.
    
    Args:
        input_data_path: Path to raw data
        train_data_path: Path to save training data
        test_data_path: Path to save test data
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Paths to processed data
    """
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    
    # Load data
    print(f"Loading data from: {input_data_path}")
    df = pd.read_csv(input_data_path)
    
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Add target back
    train_df = X_train_scaled.copy()
    train_df['MEDV'] = y_train.values
    
    test_df = X_test_scaled.copy()
    test_df['MEDV'] = y_test.values
    
    # Save processed data
    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    
    print(f"\n✓ Preprocessing completed!")
    print(f"  Training data saved to: {train_data_path}")
    print(f"  Test data saved to: {test_data_path}")
    
    # Log to MLflow
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("train_samples", len(train_df))
    mlflow.log_param("test_samples", len(test_df))
    mlflow.log_param("scaling_method", "StandardScaler")
    
    return {
        'train_path': train_data_path,
        'test_path': test_data_path,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }


def model_training_component(
    train_data_path='data/processed/train.csv',
    model_output_path='models/random_forest_model.joblib',
    n_estimators=100,
    max_depth=10,
    random_state=42
):
    """
    Model Training Component
    
    Trains a Random Forest model and saves it.
    
    Args:
        train_data_path: Path to training data
        model_output_path: Path to save trained model
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed
        
    Returns:
        dict: Training metrics and model path
    """
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Load training data
    print(f"Loading training data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)
    
    X_train = train_df.drop('MEDV', axis=1)
    y_train = train_df['MEDV']
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Train model
    print(f"\nTraining Random Forest Regressor...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  random_state: {random_state}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    train_predictions = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    train_r2 = r2_score(y_train, train_predictions)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Training R²: {train_r2:.4f}")
    
    # Save model
    joblib.dump(model, model_output_path)
    print(f"\n✓ Model saved to: {model_output_path}")
    
    # Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_r2", train_r2)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")
    
    return {
        'model_path': model_output_path,
        'train_rmse': train_rmse,
        'train_r2': train_r2
    }


def model_evaluation_component(
    model_path='models/random_forest_model.joblib',
    test_data_path='data/processed/test.csv',
    metrics_output_path='results/metrics.txt'
):
    """
    Model Evaluation Component
    
    Evaluates the trained model on test data.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        metrics_output_path: Path to save evaluation metrics
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    X_test = test_df.drop('MEDV', axis=1)
    y_test = test_df['MEDV']
    
    print(f"  Test samples: {len(X_test)}")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"R² Score:                       {r2:.4f}")
    print(f"Mean Squared Error (MSE):       {mse:.4f}")
    print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
    print(f"{'='*60}")
    
    # Save metrics to file
    metrics_text = f"""
Boston Housing Price Prediction - Model Evaluation Results
{'='*60}

Model: Random Forest Regressor
Test Set Size: {len(X_test)} samples

Performance Metrics:
-------------------
Root Mean Squared Error (RMSE): {rmse:.4f}
Mean Absolute Error (MAE):      {mae:.4f}
R² Score:                       {r2:.4f}
Mean Squared Error (MSE):       {mse:.4f}
Mean Absolute % Error (MAPE):   {mape:.2f}%

Interpretation:
--------------
- RMSE of {rmse:.4f} means predictions are off by ~${rmse*1000:.0f} on average
- R² of {r2:.4f} means the model explains {r2*100:.1f}% of the variance
- MAPE of {mape:.2f}% indicates average percentage error

{'='*60}
"""
    
    with open(metrics_output_path, 'w') as f:
        f.write(metrics_text)
    
    print(f"\n✓ Metrics saved to: {metrics_output_path}")
    
    # Log to MLflow
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mape", mape)
    
    # Log metrics file as artifact
    mlflow.log_artifact(metrics_output_path)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse,
        'mape': mape,
        'metrics_path': metrics_output_path
    }
