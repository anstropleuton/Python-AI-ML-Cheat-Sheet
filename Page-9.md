# PAGE 9 — MLOps & Deployment

---

# 1️⃣ Model Saving

---

## Pickle (Generic Python)

```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

⚠ Not safe for untrusted sources.

---

## Joblib (Better for Sklearn)

```python
import joblib

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

Use for:

- Large numpy arrays
- sklearn pipelines

---

## Torch Save

Save weights only:

```python
import torch

torch.save(model.state_dict(), "model.pt")
```

Load:

```python
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

---

## Save Full Checkpoint

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch
}, "ckpt.pt")
```

---

## Recommended Practice

✔ Save:

- model
- config
- tokenizer
- scaler
- feature columns

---

# 2️⃣ REST API — FastAPI

---

## Install

```bash
pip install fastapi uvicorn
```

---

## Minimal Server

```python
# app.py

from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.joblib")

class Request(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: Request):
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)
    return {"prediction": int(pred[0])}
```

---

## Run Server

```bash
uvicorn app:app --reload
```

Visit:

```
http://127.0.0.1:8000/docs
```

Automatic Swagger UI.

---

## Torch Inference Endpoint

```python
@app.post("/predict")
def predict(req: Request):
    model.eval()
    X = torch.tensor(req.features).float().unsqueeze(0)

    with torch.no_grad():
        out = model(X)
        pred = torch.argmax(out, dim=1)

    return {"prediction": int(pred.item())}
```

---

## Async Endpoint

```python
@app.post("/predict")
async def predict(req: Request):
    ...
```

Use when:

- I/O bound
- Calling external APIs

---

# 3️⃣ Docker Basics

---

## Dockerfile

```dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## requirements.txt

```
fastapi
uvicorn
scikit-learn
torch
joblib
```

---

## Build Image

```bash
docker build -t ml-api .
```

---

## Run Container

```bash
docker run -p 8000:8000 ml-api
```

---

## Production Tip

Use:

```bash
uvicorn app:app --workers 4
```

Or:

```
gunicorn + uvicorn workers
```

---

# 4️⃣ Experiment Tracking — MLflow

---

## Install

```bash
pip install mlflow
```

---

## Basic Usage

```python
import mlflow

mlflow.set_experiment("my-exp")

with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", 0.92)
```

---

## Log Model

Sklearn:

```python
import mlflow.sklearn

mlflow.sklearn.log_model(model, "model")
```

PyTorch:

```python
mlflow.pytorch.log_model(model, "model")
```

---

## Log Artifacts

```python
mlflow.log_artifact("confusion_matrix.png")
```

---

## Start UI

```bash
mlflow ui
```

Visit:

```
http://127.0.0.1:5000
```

---

# 5️⃣ Model Versioning

---

## Naming Convention

```
model_v1.pkl
model_v2.pkl
model_2026_02_19.pt
```

Better:

```
model_task_dataset_metric.pt
```

---

## Directory Structure

```
project/
│
├── models/
│   ├── v1/
│   ├── v2/
│
├── data/
├── src/
```

---

## Best Checkpoint Pattern (Torch)

```python
best_loss = float("inf")

if val_loss < best_loss:
    best_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
```

---

## Load Specific Version

```python
model.load_state_dict(
    torch.load("models/v2/best_model.pt")
)
```

---

# 6️⃣ Production Checklist

---

✔ Model deterministic (set seed)
✔ Model in eval mode
✔ No gradients in inference
✔ Validate request shape
✔ Handle bad inputs
✔ Log predictions
✔ Monitor latency
✔ Monitor drift

---

# 7️⃣ Logging (Minimal)

```python
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Model loaded")
logging.error("Prediction failed")
```

---

# 8️⃣ Basic Monitoring Idea

Track:

- Latency
- Input distribution
- Prediction distribution
- Error rate

Save to:

- Logs
- Database
- Monitoring tool

---

# 9️⃣ Minimal End-to-End Deployment

Train → Save → Serve → Containerize

```python
# train.py
joblib.dump(model, "model.joblib")
```

```bash
uvicorn app:app
```

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

---

# 🔥 Real-World Upgrade Path

Local:

- FastAPI

Small scale:

- Docker + VM

Medium:

- Kubernetes

Cloud:

- AWS SageMaker
- GCP Vertex AI
- Azure ML

---

# ⚡ Common Pitfalls

❌ Forget model.eval()
❌ Forget scaler at inference
❌ Different preprocessing in prod
❌ Large model without batching
❌ Blocking server with heavy sync calls

---

# 🔥 Mastery Outcome

If you master this page:

- You can deploy models as APIs
- You can containerize ML systems
- You can track experiments
- You understand versioning
- You can move from notebook → production

---
