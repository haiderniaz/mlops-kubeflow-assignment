"""
Kubeflow Pipeline Components for Boston Housing Price Prediction
This module contains reusable pipeline components for ML workflow.
"""

from kfp.dsl import component


@component(
    base_image="python:3.9",
    packages_to_install=["dvc", "pandas"]
)
def data_extraction_component(
    dvc_remote_url: str,
    output_data_path: str
) -> str:
    """
    Data Extraction Component
    
    Fetches the versioned dataset from DVC remote storage.
    
    Args:
        dvc_remote_url: URL or path to DVC remote storage
        output_data_path: Path where extracted data will be saved
        
    Returns:
        str: Path to the extracted dataset
    """
    import os
    import subprocess
    import pandas as pd
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    
    # For this implementation, we'll simulate DVC get
    # In production, you would use: dvc get <repo> <path> -o <output>
    print(f"Extracting data from DVC remote: {dvc_remote_url}")
    print(f"Output path: {output_data_path}")
    
    # For demonstration, we're creating a sample dataset
    # In production, this would fetch from actual DVC remote
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    
    try:
        import numpy as np
        raw_df = pd.read_csv(data_url, sep=r'\s+', skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        df = pd.DataFrame(data, columns=feature_names)
        df['MEDV'] = target
        
        df.to_csv(output_data_path, index=False)
        print(f"Data extracted successfully: {len(df)} rows")
        
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        raise
    
    return output_data_path


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def data_preprocessing_component(
    input_data_path: str,
    train_data_path: str,
    test_data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Data Preprocessing Component
    
    Handles cleaning, scaling, and splitting data into train/test sets.
    
    Args:
        input_data_path: Path to raw input data
        train_data_path: Path to save training data
        test_data_path: Path to save test data
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        dict: Dictionary with paths and statistics
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import os
    
    print(f"Loading data from: {input_data_path}")
    df = pd.read_csv(input_data_path)
    
    # Data Cleaning
    print("Performing data cleaning...")
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    print(f"Removed {missing_before} missing values")
    
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split data
    print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature Scaling
    print("Applying StandardScaler to features...")
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
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    
    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    
    print(f"Training data saved: {train_data_path} ({len(train_df)} rows)")
    print(f"Test data saved: {test_data_path} ({len(test_df)} rows)")
    
    return {
        "train_path": train_data_path,
        "test_path": test_data_path,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "feature_count": len(X.columns)
    }


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"]
)
def model_training_component(
    train_data_path: str,
    model_output_path: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> dict:
    """
    Model Training Component
    
    Trains a Random Forest classifier and saves the model artifact.
    
    Args:
        train_data_path: Path to training data CSV file
        model_output_path: Path to save trained model (.joblib)
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum depth of trees (default: 10)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        dict: Dictionary containing model path and training metrics
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
    import os
    
    print(f"Loading training data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)
    
    # Separate features and target
    X_train = train_df.drop('MEDV', axis=1)
    y_train = train_df['MEDV']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Initialize and train Random Forest model
    print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Training set predictions for metrics
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Training Metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")
    
    # Get feature importances
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 important features: {top_features}")
    
    return {
        "model_path": model_output_path,
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "train_r2": float(train_r2),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "feature_count": len(X_train.columns)
    }


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"]
)
def model_evaluation_component(
    model_path: str,
    test_data_path: str,
    metrics_output_path: str
) -> dict:
    """
    Model Evaluation Component
    
    Loads trained model, evaluates on test set, and saves metrics.
    
    Args:
        model_path: Path to trained model file (.joblib)
        test_data_path: Path to test data CSV file
        metrics_output_path: Path to save evaluation metrics (JSON)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        mean_squared_error, 
        r2_score, 
        mean_absolute_error,
        mean_absolute_percentage_error
    )
    import joblib
    import json
    import os
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_test = test_df.drop('MEDV', axis=1)
    y_test = test_df['MEDV']
    
    print(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "mape": float(mape),
        "mse": float(mse),
        "test_samples": len(y_test),
        "predictions_mean": float(np.mean(y_pred)),
        "predictions_std": float(np.std(y_pred)),
        "actual_mean": float(np.mean(y_test)),
        "actual_std": float(np.std(y_test))
    }
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("="*50)
    
    # Save metrics to file
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_output_path}")
    
    return metrics