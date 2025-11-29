"""
Boston Housing Price Prediction - Kubeflow Pipeline
This module defines the complete ML pipeline orchestrating all components.
"""

from kfp import dsl
from kfp import compiler
import os
import sys

# Add src to path to import components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)


@dsl.pipeline(
    name='Boston Housing Price Prediction Pipeline',
    description='End-to-end ML pipeline for predicting Boston housing prices using Random Forest'
)
def boston_housing_pipeline(
    # Pipeline parameters
    dvc_remote_url: str = '../dvc_remote',
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
):
    """
    Complete ML Pipeline for Boston Housing Price Prediction
    
    This pipeline orchestrates four main components:
    1. Data Extraction - Fetch versioned dataset from DVC
    2. Data Preprocessing - Clean, scale, and split data
    3. Model Training - Train Random Forest model
    4. Model Evaluation - Evaluate model performance
    
    Args:
        dvc_remote_url: URL or path to DVC remote storage
        test_size: Proportion of data for testing (0.0-1.0)
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of decision trees
        random_state: Random seed for reproducibility
    """
    
    # Step 1: Data Extraction
    # Fetches the versioned dataset from DVC remote storage
    extract_task = data_extraction_component(
        dvc_remote_url=dvc_remote_url,
        output_data_path='/data/raw/boston.csv'
    )
    extract_task.set_display_name('Extract Data from DVC')
    
    # Step 2: Data Preprocessing
    # Cleans, scales, and splits data into train/test sets
    preprocess_task = data_preprocessing_component(
        input_data_path=extract_task.output,
        train_data_path='/data/processed/train.csv',
        test_data_path='/data/processed/test.csv',
        test_size=test_size,
        random_state=random_state
    )
    preprocess_task.set_display_name('Preprocess & Split Data')
    preprocess_task.after(extract_task)
    
    # Step 3: Model Training
    # Trains Random Forest model on training data
    train_task = model_training_component(
        train_data_path='/data/processed/train.csv',
        model_output_path='/models/random_forest_model.joblib',
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    train_task.set_display_name('Train Random Forest Model')
    train_task.after(preprocess_task)
    
    # Step 4: Model Evaluation
    # Evaluates trained model on test data
    eval_task = model_evaluation_component(
        model_path='/models/random_forest_model.joblib',
        test_data_path='/data/processed/test.csv',
        metrics_output_path='/metrics/evaluation_metrics.json'
    )
    eval_task.set_display_name('Evaluate Model Performance')
    eval_task.after(train_task)
    
    # Print pipeline info
    print(f"Pipeline configured with:")
    print(f"  - Test size: {test_size}")
    print(f"  - Random Forest trees: {n_estimators}")
    print(f"  - Max tree depth: {max_depth}")
    print(f"  - Random seed: {random_state}")


def compile_pipeline(output_path='pipeline.yaml'):
    """
    Compile the pipeline to a YAML file for deployment.
    
    Args:
        output_path: Path where the compiled pipeline YAML will be saved
    """
    print(f"Compiling pipeline to: {output_path}")
    
    compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path=output_path
    )
    
    print(f"✓ Pipeline compiled successfully!")
    print(f"✓ Output file: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Upload {output_path} to Kubeflow Pipelines UI")
    print(f"2. Create a new run with desired parameters")
    print(f"3. Monitor the execution in the UI")


if __name__ == '__main__':
    # Compile the pipeline when script is run directly
    compile_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE COMPILATION COMPLETE")
    print("="*60)
    print("\nPipeline Structure:")
    print("  1. Data Extraction Component")
    print("     └─> Fetches Boston housing dataset from DVC")
    print("  2. Data Preprocessing Component")
    print("     └─> Cleans, scales, splits into train/test")
    print("  3. Model Training Component")
    print("     └─> Trains Random Forest Regressor")
    print("  4. Model Evaluation Component")
    print("     └─> Evaluates model and saves metrics")
    print("="*60)