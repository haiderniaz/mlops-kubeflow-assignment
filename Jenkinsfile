pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.11'
    }
    
    stages {
        stage('Environment Setup') {
            steps {
                echo '=========================================='
                echo 'STAGE 1: ENVIRONMENT SETUP'
                echo '=========================================='
                
                // Checkout code from GitHub
                checkout scm
                
                // Install Python dependencies
                sh '''
                    echo "Installing Python dependencies..."
                    python3 -m pip install --upgrade pip
                    pip install -r requirements.txt
                    echo "✓ Dependencies installed successfully"
                '''
                
                // Verify installation
                sh '''
                    echo "Verifying environment setup..."
                    pip list | grep -E "(mlflow|scikit-learn|pandas|numpy|dvc)"
                    python3 --version
                    echo "✓ Environment setup complete"
                '''
            }
        }
        
        stage('Pipeline Compilation & Validation') {
            steps {
                echo '=========================================='
                echo 'STAGE 2: PIPELINE COMPILATION & VALIDATION'
                echo '=========================================='
                
                // Validate Python syntax
                sh '''
                    echo "Validating pipeline components syntax..."
                    python3 -m py_compile src/mlflow_pipeline_components.py
                    python3 -m py_compile run_pipeline.py
                    echo "✓ Pipeline components are syntactically correct"
                '''
                
                // Check pipeline structure
                sh '''
                    echo "Checking pipeline structure..."
                    python3 -c "
from src.mlflow_pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)
print('✓ All pipeline components imported successfully')
print('✓ Pipeline components validated:')
print('  - data_extraction_component')
print('  - data_preprocessing_component')
print('  - model_training_component')
print('  - model_evaluation_component')
                    "
                '''
                
                // Validate pipeline runner
                sh '''
                    echo "Validating pipeline runner..."
                    python3 -c "
import sys
sys.path.insert(0, '.')
from run_pipeline import run_pipeline
print('✓ Pipeline runner validated successfully')
print('✓ Pipeline is ready for execution')
                    "
                '''
            }
        }
        
        stage('Pipeline Execution Test') {
            steps {
                echo '=========================================='
                echo 'STAGE 3: PIPELINE EXECUTION TEST'
                echo '=========================================='
                
                // Run pipeline with test parameters
                sh '''
                    echo "Executing pipeline with test parameters..."
                    python3 run_pipeline.py \
                        --experiment-name "Jenkins-CI-Test" \
                        --n-estimators 10 \
                        --max-depth 3 \
                        --test-size 0.2
                    echo "✓ Pipeline executed successfully"
                '''
                
                // Verify outputs
                sh '''
                    echo "Verifying pipeline outputs..."
                    
                    # Check data files
                    if [ -f "data/raw/boston.csv" ]; then
                        echo "✓ Raw data extracted"
                    else
                        echo "✗ Raw data missing"
                        exit 1
                    fi
                    
                    if [ -f "data/processed/train.csv" ] && [ -f "data/processed/test.csv" ]; then
                        echo "✓ Processed data created"
                    else
                        echo "✗ Processed data missing"
                        exit 1
                    fi
                    
                    # Check model file
                    if [ -f "models/random_forest_model.joblib" ]; then
                        echo "✓ Model trained and saved"
                    else
                        echo "✗ Model file missing"
                        exit 1
                    fi
                    
                    # Check metrics file
                    if [ -f "results/metrics.txt" ]; then
                        echo "✓ Evaluation metrics generated"
                        echo "--- Metrics Preview ---"
                        head -n 15 results/metrics.txt
                    else
                        echo "✗ Metrics file missing"
                        exit 1
                    fi
                    
                    echo "✓ All pipeline outputs verified successfully"
                '''
                
                // Archive artifacts
                archiveArtifacts artifacts: 'results/metrics.txt, models/*.joblib', 
                                 fingerprint: true,
                                 allowEmptyArchive: false
            }
        }
    }
    
    post {
        success {
            echo '=========================================='
            echo 'PIPELINE CI/CD COMPLETED SUCCESSFULLY!'
            echo '=========================================='
            echo '✓ Stage 1: Environment Setup - PASSED'
            echo '✓ Stage 2: Pipeline Compilation & Validation - PASSED'
            echo '✓ Stage 3: Pipeline Execution Test - PASSED'
            echo '=========================================='
        }
        
        failure {
            echo '=========================================='
            echo '✗ PIPELINE CI/CD FAILED'
            echo '=========================================='
            echo 'Please check the logs above for error details.'
        }
        
        always {
            // Clean up workspace
            echo 'Cleaning up workspace...'
            cleanWs()
        }
    }
}
