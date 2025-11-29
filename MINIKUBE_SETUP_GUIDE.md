# Task 3: Minikube & Kubeflow Setup Guide

## Prerequisites
- Windows 10/11 with WSL2 or Hyper-V enabled
- Docker Desktop installed and running
- At least 8GB RAM available
- 20GB free disk space

---

## Step 1: Install Minikube

### Option A: Using Chocolatey (Recommended for Windows)
```powershell
# Run PowerShell as Administrator
choco install minikube
```

### Option B: Manual Installation
```powershell
# Download Minikube installer
New-Item -Path 'c:\' -Name 'minikube' -ItemType Directory -Force
Invoke-WebRequest -OutFile 'c:\minikube\minikube.exe' -Uri 'https://github.com/kubernetes/minikube/releases/latest/download/minikube-windows-amd64.exe' -UseBasicParsing

# Add to PATH
$oldPath = [Environment]::GetEnvironmentVariable('Path', [EnvironmentVariableTarget]::Machine)
if ($oldPath.Split(';') -inotcontains 'C:\minikube'){
  [Environment]::SetEnvironmentVariable('Path', $('{0};C:\minikube' -f $oldPath), [EnvironmentVariableTarget]::Machine)
}
```

### Option C: Using Winget
```powershell
winget install Kubernetes.minikube
```

---

## Step 2: Start Minikube Cluster

```powershell
# Start Minikube with sufficient resources for Kubeflow
minikube start --cpus=4 --memory=8192 --disk-size=40g --driver=docker

# Verify status (FOR SCREENSHOT)
minikube status

# Check cluster info
kubectl cluster-info

# Enable ingress addon (required for Kubeflow)
minikube addons enable ingress
```

**Expected Output for `minikube status`:**
```
minikube
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured
```

---

## Step 3: Install Kubeflow Pipelines

### Option A: Standalone Kubeflow Pipelines (Recommended - Faster & Lighter)

```powershell
# Install standalone KFP version 2.0.5
$KFP_VERSION = "2.0.5"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$KFP_VERSION"

# Wait for all pods to be ready (this may take 5-10 minutes)
kubectl wait --for=condition=ready --timeout=600s pods --all -n kubeflow

# Check pod status
kubectl get pods -n kubeflow
```

### Option B: Full Kubeflow Deployment (More Features, Heavier)

```powershell
# Install kustomize
choco install kustomize

# Clone Kubeflow manifests
git clone https://github.com/kubeflow/manifests.git
cd manifests

# Install Kubeflow (this takes 15-20 minutes)
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

# Wait for all pods
kubectl wait --for=condition=ready --timeout=1800s pods --all -n kubeflow
```

---

## Step 4: Access Kubeflow Pipelines UI

### Port Forward to Local Machine

```powershell
# Forward KFP UI port (FOR SCREENSHOT ACCESS)
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# In a new terminal, you can also forward the API server
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
```

**Access the UI:**
- Open browser: http://localhost:8080
- For full Kubeflow: http://localhost:8080/_/pipeline

### Alternative: Use Minikube Tunnel

```powershell
# In a separate PowerShell window (run as Administrator)
minikube tunnel
```

---

## Step 5: Verify Kubeflow Pipelines Installation

```powershell
# Check all KFP components are running
kubectl get pods -n kubeflow | findstr pipeline

# Expected pods:
# - ml-pipeline-*
# - ml-pipeline-ui-*
# - ml-pipeline-visualizationserver-*
# - ml-pipeline-persistenceagent-*
# - ml-pipeline-scheduledworkflow-*
```

---

## Step 6: Upload and Run Pipeline

### Via UI (Recommended for Screenshots):

1. **Access KFP UI**: http://localhost:8080

2. **Upload Pipeline**:
   - Click "Pipelines" in left menu
   - Click "+ Upload pipeline"
   - Select `pipeline.yaml` from your project
   - Give it a name: "Boston Housing Price Prediction"
   - Click "Create"

3. **Create a Run**:
   - Click on your pipeline
   - Click "+ Create run"
   - Fill in parameters:
     - dvc_remote_url: `../dvc_remote`
     - test_size: `0.2`
     - n_estimators: `100`
     - max_depth: `10`
     - random_state: `42`
   - Click "Start"

4. **Monitor Execution** (FOR SCREENSHOTS):
   - View the graph showing all 4 components
   - Watch as each step completes (green checkmarks)
   - Click on "Model Evaluation" step to see outputs
   - View metrics (RMSE, MAE, R² score)

### Via Python SDK (Alternative):

```python
import kfp
from kfp import compiler

# Connect to KFP
client = kfp.Client(host='http://localhost:8080')

# Upload pipeline
pipeline_id = client.upload_pipeline(
    pipeline_package_path='pipeline.yaml',
    pipeline_name='Boston Housing Price Prediction'
)

# Create and run experiment
experiment = client.create_experiment(name='Boston Housing Experiment')

run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name='boston-housing-run-1',
    pipeline_id=pipeline_id,
    params={
        'dvc_remote_url': '../dvc_remote',
        'test_size': 0.2,
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
)

print(f"Run created: {run.id}")
print(f"View at: http://localhost:8080/#/runs/details/{run.id}")
```

---

## Screenshots Required for Deliverable 3:

### Screenshot 1: Minikube Status
```powershell
minikube status
```
**Should show:**
- minikube: Running
- host: Running
- kubelet: Running
- apiserver: Running

### Screenshot 2: KFP UI - Pipeline Graph
- Navigate to your run in the UI
- Click on "Graph" tab
- Shows all 4 components connected:
  1. Extract Data from DVC
  2. Preprocess & Split Data
  3. Train Random Forest Model
  4. Evaluate Model Performance
- All should have green checkmarks

### Screenshot 3: Pipeline Run Details - Outputs
- Click on "Evaluate Model Performance" component
- Click "Logs" or "Output" tab
- Should show:
  - RMSE: ~3.5-4.5
  - MAE: ~2.5-3.5
  - R² Score: ~0.75-0.85
  - Test samples: ~101-102
  - Other evaluation metrics

---

## Troubleshooting

### Issue: Pods stuck in Pending
```powershell
# Check events
kubectl get events -n kubeflow --sort-by='.lastTimestamp'

# Increase resources
minikube delete
minikube start --cpus=6 --memory=12288 --disk-size=50g
```

### Issue: Port forward fails
```powershell
# Kill existing port forwards
Get-Process -Name kubectl | Stop-Process -Force

# Restart port forward
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

### Issue: Cannot access UI
```powershell
# Check service is running
kubectl get svc -n kubeflow | findstr ml-pipeline-ui

# Try different port
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 3000:80
# Access at http://localhost:3000
```

### Issue: Pipeline fails to run
```powershell
# Check logs
kubectl logs -n kubeflow -l app=ml-pipeline

# Verify components can pull images
minikube ssh docker pull python:3.9
```

---

## Cleanup (After Screenshots)

```powershell
# Stop Minikube
minikube stop

# Delete cluster (optional)
minikube delete

# Or just pause
minikube pause
```

---

## Additional Commands for Verification

```powershell
# View all namespaces
kubectl get namespaces

# View all pods across namespaces
kubectl get pods --all-namespaces

# Get detailed info about KFP deployment
kubectl describe deployment ml-pipeline -n kubeflow

# View pipeline runs
kubectl get workflows -n kubeflow

# Get logs from a specific pod
kubectl logs <pod-name> -n kubeflow

# Access Minikube dashboard
minikube dashboard
```

---

## Expected Timeline

- Minikube installation: 5-10 minutes
- Cluster start: 2-5 minutes  
- KFP installation: 5-15 minutes (standalone) or 15-30 minutes (full)
- Pipeline upload and run: 5-10 minutes
- **Total: 20-60 minutes**

---

## Success Criteria

✅ `minikube status` shows all components running
✅ Can access KFP UI at http://localhost:8080
✅ Pipeline uploads successfully
✅ All 4 components execute in sequence
✅ Model evaluation shows reasonable metrics
✅ Can view outputs and logs for each component

---

## Resources

- Minikube Docs: https://minikube.sigs.k8s.io/docs/
- Kubeflow Pipelines: https://www.kubeflow.org/docs/components/pipelines/
- KFP SDK: https://kubeflow-pipelines.readthedocs.io/
- Kubernetes Docs: https://kubernetes.io/docs/
