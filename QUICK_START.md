# Quick Start Guide - Task 3 Deployment

## 📋 Prerequisites Checklist
- [ ] Docker Desktop is installed and running
- [ ] At least 8GB RAM available
- [ ] 20GB free disk space
- [ ] PowerShell with Administrator privileges

---

## 🚀 Quick Setup (15-20 minutes)

### 1. Install & Start Minikube (5 minutes)

```powershell
# Install Minikube (choose one method)
choco install minikube          # OR
winget install Kubernetes.minikube

# Start cluster with required resources
minikube start

# ✅ SCREENSHOT 1: Verify status
minikube status
```

### 2. Deploy Kubeflow Pipelines (10 minutes)

```powershell
# Install standalone KFP
$KFP_VERSION = "2.0.5"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$KFP_VERSION"

# Wait for pods to be ready
kubectl wait --for=condition=ready --timeout=600s pods --all -n kubeflow

# Verify installation
kubectl get pods -n kubeflow
```

### 3. Access KFP UI (2 minutes)

```powershell
# Forward KFP UI to localhost
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Open in browser: http://localhost:8080
```

---

## 📤 Upload & Run Pipeline

### Method A: Using UI (Recommended for Screenshots)

1. **Open KFP UI**: http://localhost:8080

2. **Upload Pipeline**:
   - Click "Pipelines" → "+ Upload pipeline"
   - Select: `pipeline.yaml`
   - Name: "Boston Housing Price Prediction"
   - Click "Create"

3. **Create Run**:
   - Click your pipeline → "+ Create run"
   - Use default parameters:
     - dvc_remote_url: `../dvc_remote`
     - test_size: `0.2`
     - n_estimators: `100`
     - max_depth: `10`
     - random_state: `42`
   - Click "Start"

4. **✅ SCREENSHOT 2: Pipeline Graph**
   - View the graph showing 4 connected components
   - All steps should turn green when complete

5. **✅ SCREENSHOT 3: Outputs**
   - Click "Evaluate Model Performance" step
   - View metrics in logs/outputs

### Method B: Using Python Script

```powershell
# Run submission script
python submit_pipeline.py
```

---

## 📸 Required Screenshots

### Screenshot 1: Minikube Status
```powershell
minikube status
```
**Must show:**
- ✅ minikube: Running
- ✅ host: Running
- ✅ kubelet: Running
- ✅ apiserver: Running

### Screenshot 2: KFP UI - Pipeline Graph
**Navigate to:** Run details → Graph tab
**Must show:**
1. Extract Data from DVC (green ✓)
2. Preprocess & Split Data (green ✓)
3. Train Random Forest Model (green ✓)
4. Evaluate Model Performance (green ✓)
**All connected with arrows**

### Screenshot 3: Model Evaluation Outputs
**Click:** "Evaluate Model Performance" → Logs/Output tab
**Must show:**
- RMSE: ~3.5-4.5
- MAE: ~2.5-3.5
- R² Score: ~0.75-0.85
- Test samples count
- Other metrics

---

## ⚡ Quick Commands Reference

```powershell
# Check Minikube status
minikube status

# View all KFP pods
kubectl get pods -n kubeflow

# Port forward KFP UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# View pipeline logs
kubectl logs -n kubeflow -l app=ml-pipeline

# View all running workflows
kubectl get workflows -n kubeflow

# Stop Minikube (after screenshots)
minikube stop

# Delete cluster (if needed)
minikube delete
```

---

## 🐛 Troubleshooting

### Issue: Minikube won't start
```powershell
# Clean start
minikube delete
minikube start --cpus=4 --memory=8192 --disk-size=40g --driver=docker
```

### Issue: Can't access UI
```powershell
# Kill existing port forwards
Get-Process -Name kubectl | Stop-Process -Force

# Restart port forward
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

### Issue: Pods stuck in Pending
```powershell
# Increase resources
minikube stop
minikube start --cpus=6 --memory=12288 --disk-size=50g
```

### Issue: Pipeline fails
```powershell
# Check pod logs
kubectl get pods -n kubeflow
kubectl logs <pod-name> -n kubeflow

# Verify docker can pull images
minikube ssh docker pull python:3.9
```

---

## 🎯 Expected Results

**Pipeline Execution Time:** 5-10 minutes
**All Components:** Should complete successfully
**Model Performance:**
- RMSE: 3.5-4.5 (lower is better)
- MAE: 2.5-3.5 (lower is better)
- R² Score: 0.75-0.85 (higher is better, max 1.0)

---

## 📚 Files Generated

- ✅ `pipeline.yaml` - Compiled Kubeflow pipeline
- ✅ `submit_pipeline.py` - Python script for submission
- ✅ `MINIKUBE_SETUP_GUIDE.md` - Detailed setup instructions
- ✅ All component YAML files in `components/` directory

---

## ✨ Success Checklist

- [x] Pipeline compiled to YAML
- [ ] Minikube running and verified
- [ ] Kubeflow Pipelines deployed
- [ ] Can access KFP UI at localhost:8080
- [ ] Pipeline uploaded successfully
- [ ] Pipeline executed completely (all 4 steps)
- [ ] All screenshots captured
- [ ] Model metrics are reasonable

---

## 🔗 Resources

- **Full Setup Guide**: See `MINIKUBE_SETUP_GUIDE.md`
- **Component Docs**: See `COMPONENT_DOCUMENTATION.md`
- **Minikube Docs**: https://minikube.sigs.k8s.io/docs/
- **KFP Docs**: https://www.kubeflow.org/docs/components/pipelines/

---

**🎉 Good luck with your deployment!**
