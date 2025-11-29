"""
MLflow Pipeline Runner for Boston Housing Price Prediction
This script executes the full ML pipeline with MLflow tracking.
"""

import mlflow
import sys
from datetime import datetime
from src.mlflow_pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)


def run_pipeline(
    experiment_name="Boston Housing Prediction",
    test_size=0.2,
    n_estimators=100,
    max_depth=10,
    random_state=42
):
    """
    Execute the complete ML pipeline with MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        test_size: Proportion of test set (0.0 to 1.0)
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
    """
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    print("\n" + "="*70)
    print(" "*15 + "BOSTON HOUSING PRICE PREDICTION PIPELINE")
    print("="*70)
    print(f"\nExperiment: {experiment_name}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPipeline Parameters:")
    print(f"  - Test Size: {test_size}")
    print(f"  - N Estimators: {n_estimators}")
    print(f"  - Max Depth: {max_depth}")
    print(f"  - Random State: {random_state}")
    print("="*70)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        try:
            # Step 1: Data Extraction
            data_path = data_extraction_component(
                output_data_path='data/raw/boston.csv'
            )
            
            # Step 2: Data Preprocessing
            preprocess_result = data_preprocessing_component(
                input_data_path=data_path,
                train_data_path='data/processed/train.csv',
                test_data_path='data/processed/test.csv',
                test_size=test_size,
                random_state=random_state
            )
            
            # Step 3: Model Training
            train_result = model_training_component(
                train_data_path=preprocess_result['train_path'],
                model_output_path='models/random_forest_model.joblib',
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
            # Step 4: Model Evaluation
            eval_result = model_evaluation_component(
                model_path=train_result['model_path'],
                test_data_path=preprocess_result['test_path'],
                metrics_output_path='results/metrics.txt'
            )
            
            # Log final summary
            print("\n" + "="*70)
            print(" "*25 + "PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nFinal Model Performance:")
            print(f"  RMSE: {eval_result['rmse']:.4f}")
            print(f"  MAE:  {eval_result['mae']:.4f}")
            print(f"  R²:   {eval_result['r2']:.4f}")
            print(f"  MAPE: {eval_result['mape']:.2f}%")
            print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            
            # Tag the run
            mlflow.set_tag("pipeline_status", "success")
            mlflow.set_tag("pipeline_version", "1.0")
            
            print(f"\n✓ MLflow Run ID: {mlflow.active_run().info.run_id}")
            print(f"✓ View results in MLflow UI: http://localhost:5000")
            print("\nTo start MLflow UI, run:")
            print("  mlflow ui")
            
            return {
                'status': 'success',
                'run_id': mlflow.active_run().info.run_id,
                'metrics': eval_result
            }
            
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {str(e)}")
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("error_message", str(e))
            raise


def main():
    """Main execution function with argument parsing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Boston Housing Price Prediction Pipeline with MLflow'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Random Forest (default: 100)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Maximum depth of trees (default: 10)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='Boston Housing Prediction',
        help='MLflow experiment name (default: "Boston Housing Prediction")'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        result = run_pipeline(
            experiment_name=args.experiment_name,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
