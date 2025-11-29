"""
Script to submit pipeline to Kubeflow Pipelines via Python SDK
"""

import kfp
from kfp import compiler
import sys
from datetime import datetime


def check_kfp_connection(host='http://localhost:8080'):
    """Check if KFP server is accessible."""
    try:
        client = kfp.Client(host=host)
        print(f"✓ Connected to Kubeflow Pipelines at {host}")
        return client
    except Exception as e:
        print(f"✗ Failed to connect to Kubeflow Pipelines: {str(e)}")
        print("\nMake sure:")
        print("1. Minikube is running: minikube status")
        print("2. Port forwarding is active: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")
        return None


def upload_and_run_pipeline(client, pipeline_file='pipeline.yaml'):
    """Upload pipeline and create a run."""
    
    try:
        # Create experiment
        experiment_name = 'Boston Housing Prediction'
        try:
            experiment = client.create_experiment(name=experiment_name)
            print(f"✓ Created experiment: {experiment_name}")
        except:
            # Experiment might already exist
            experiment = client.get_experiment(experiment_name=experiment_name)
            print(f"✓ Using existing experiment: {experiment_name}")
        
        # Upload pipeline
        pipeline_name = f'Boston Housing Pipeline {datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        print(f"\nUploading pipeline from: {pipeline_file}")
        pipeline = client.upload_pipeline(
            pipeline_package_path=pipeline_file,
            pipeline_name=pipeline_name
        )
        print(f"✓ Pipeline uploaded successfully!")
        print(f"  Pipeline ID: {pipeline.id}")
        
        # Create run
        run_name = f'boston-housing-run-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        print(f"\nCreating pipeline run: {run_name}")
        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name,
            pipeline_id=pipeline.id,
            params={
                'dvc_remote_url': '../dvc_remote',
                'test_size': 0.2,
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        )
        
        print(f"✓ Pipeline run created successfully!")
        print(f"  Run ID: {run.id}")
        print(f"  Run Name: {run_name}")
        print(f"\n{'='*60}")
        print(f"View run in UI:")
        print(f"  http://localhost:8080/#/runs/details/{run.id}")
        print(f"{'='*60}")
        
        return run
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def list_pipelines(client):
    """List all existing pipelines."""
    try:
        pipelines = client.list_pipelines(page_size=10)
        if pipelines.pipelines:
            print("\nExisting Pipelines:")
            print("-" * 60)
            for p in pipelines.pipelines:
                print(f"  Name: {p.name}")
                print(f"  ID: {p.id}")
                print(f"  Created: {p.created_at}")
                print("-" * 60)
        else:
            print("\nNo pipelines found.")
    except Exception as e:
        print(f"Error listing pipelines: {str(e)}")


def main():
    """Main execution function."""
    
    print("="*60)
    print("KUBEFLOW PIPELINES - PIPELINE SUBMISSION")
    print("="*60)
    
    # Check connection
    client = check_kfp_connection()
    if not client:
        sys.exit(1)
    
    # List existing pipelines
    list_pipelines(client)
    
    # Ask user if they want to upload
    print("\n" + "="*60)
    response = input("Upload and run pipeline.yaml? (yes/no): ").lower()
    
    if response in ['yes', 'y']:
        run = upload_and_run_pipeline(client)
        
        if run:
            print("\n✓ SUCCESS! Pipeline is now running.")
            print("\nNext steps:")
            print("1. Open the URL above in your browser")
            print("2. Watch the pipeline execute")
            print("3. Take screenshots for your deliverable")
            print("   - Pipeline graph with all components")
            print("   - Completed run with green checkmarks")
            print("   - Evaluation metrics output")
    else:
        print("\nOperation cancelled.")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
