# PAGE 10 — Full Workflow Blueprint

---

# 1️⃣ End-to-End ML Pipeline

---

## 1. Problem Definition

Define:

```python
TASK = "binary_classification"
METRIC = "roc_auc"
CONSTRAINTS = {
    "latency_ms": 50,
    "memory_mb": 500
}
```

Checklist:

- Input?
- Output?
- Business metric?
- Offline metric?
- Real-time or batch?

---

## 2. Data Ingestion

CSV:

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

Database:

```python
import sqlalchemy

engine = sqlalchemy.create_engine(DB_URL)
df = pd.read_sql("SELECT * FROM table", engine)
```

API:

```python
import requests
data = requests.get(URL).json()
```

Validate early:

```python
assert df.shape[0] > 0
assert "target" in df.columns
```

---

## 3. EDA (Fast Triage)

```python
df.head()
df.info()
df.describe()
df.isna().mean()
```

Target balance:

```python
df["target"].value_counts(normalize=True)
```

Correlation:

```python
df.corr(numeric_only=True)["target"]
```

---

## 4. Feature Engineering

Split:

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Encoding:

```python
X = pd.get_dummies(X, drop_first=True)
```

Scaling:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Save scaler:

```python
import joblib
joblib.dump(scaler, "scaler.joblib")
```

---

## 5. Modeling

Baseline first:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Upgrade:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=300)
```

---

## 6. Validation

Split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

Metric:

```python
from sklearn.metrics import roc_auc_score

pred = model.predict_proba(X_test)[:,1]
roc_auc_score(y_test, pred)
```

Cross-val:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
```

---

## 7. Deployment

Save:

```python
joblib.dump(model, "model.joblib")
```

Serve → FastAPI
Containerize → Docker
Monitor → Logs + metrics

---

# 2️⃣ Common Failure Modes

---

## Data Leakage

Wrong:

```python
scaler.fit(X)   # before split ❌
```

Correct:

```python
scaler.fit(X_train)
```

Leak sources:

- Target encoding using full dataset
- Time-series future data
- Feature created from label

---

## Overfitting

Symptoms:

- Train AUC = 0.99
- Test AUC = 0.72

Fix:

- Regularization
- More data
- Simpler model
- Cross-validation

---

## Underfitting

Symptoms:

- Train AUC = 0.60
- Test AUC = 0.58

Fix:

- More features
- More complex model
- Feature engineering

---

## Distribution Shift

Train:

```
avg_income = 50k
```

Prod:

```
avg_income = 80k
```

Detect:

```python
df_train.mean()
df_prod.mean()
```

Monitor:

- Feature drift
- Prediction drift

---

# 3️⃣ Production Concerns

---

## Latency

Measure:

```python
import time

start = time.time()
model.predict(X_batch)
latency = time.time() - start
```

Optimize:

- Smaller model
- ONNX export
- Batch inference

---

## Monitoring

Track:

```python
{
  "timestamp": ...,
  "latency_ms": ...,
  "prediction": ...
}
```

Log distribution:

```python
pred.mean()
```

---

## Drift Detection (Simple)

Compare means:

```python
abs(train_mean - prod_mean)
```

Advanced:

- KS test
- PSI (Population Stability Index)

---

## Retraining Loop

Pseudo:

```python
if performance < threshold:
    retrain()
    redeploy()
```

Automate with:

- Scheduled jobs
- CI/CD pipeline

---

# 4️⃣ Scaling

---

## Batching

Bad:

```python
for x in data:
    model.predict(x)
```

Good:

```python
model.predict(batch)
```

---

## Parallelism (CPU)

```python
model = RandomForestClassifier(n_jobs=-1)
```

Multiprocessing:

```python
from multiprocessing import Pool
```

---

## GPU (PyTorch)

```python
device = "cuda"
model.to(device)
```

Mixed precision:

```python
from torch.cuda.amp import autocast
```

---

## Async API Scaling

```bash
uvicorn app:app --workers 4
```

Horizontal scaling:

- Multiple containers
- Load balancer

---

# 5️⃣ Debug Checklist

---

### Model not learning?

- Check labels
- Check scaling
- Overfit small batch
- Print gradients
- Check class imbalance

---

### API crashing?

- Validate input schema
- Check tensor shapes
- Ensure model.eval()

---

### Bad performance in prod?

- Compare train vs prod features
- Check preprocessing
- Check drift

---

# 6️⃣ Minimal ML Project Template

---

```
ml-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── features.py
│   └── utils.py
│
├── models/
│
├── app.py
├── requirements.txt
└── Dockerfile
```

---

# 7️⃣ Minimal Starter Training Script

```python
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)

    save(model)

if __name__ == "__main__":
    main()
```

---

# 8️⃣ Essential Libraries List

Core:

- numpy
- pandas
- scikit-learn
- matplotlib

DL:

- torch
- transformers

MLOps:

- fastapi
- uvicorn
- mlflow
- joblib

Optional:

- xgboost
- lightgbm
- catboost
- optuna

---

# 9️⃣ Golden Rules

✔ Baseline first
✔ Split before transform
✔ Log everything
✔ Save preprocessing
✔ Monitor in production
✔ Reproduce experiments
✔ Keep models simple

---

# 🔥 The Complete Mental Model

1. Define problem
2. Clean data
3. Build baseline
4. Improve features
5. Cross-validate
6. Save + deploy
7. Monitor
8. Retrain

Repeat loop.

---

# 🔥 If You Master All 10 Pages

You can:

- Build ML systems from scratch
- Train deep learning models
- Fine-tune transformers
- Deploy APIs
- Containerize services
- Monitor production models
- Design scalable ML systems

---
