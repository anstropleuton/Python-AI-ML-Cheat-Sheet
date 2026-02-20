# PAGE 6 — Scikit-Learn Workflow

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
```

---

# 1️⃣ Train / Test Split

---

## Basic

```python
X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## Stratified (For Classification)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

⚠ Always stratify if class imbalance.

---

# 2️⃣ Preprocessing

---

## Scaling

### StandardScaler (mean=0, std=1)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

⚠ Never fit on test.

---

### MinMaxScaler (0–1)

```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

---

## Encoding Categorical

```python
encoder = OneHotEncoder(handle_unknown="ignore")

X_cat = encoder.fit_transform(df[["category"]])
```

Dense:

```python
encoder = OneHotEncoder(sparse_output=False)
```

---

## ColumnTransformer (Mixed Data)

```python
num_cols = ["age", "income"]
cat_cols = ["city"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
```

---

## Pipeline (CRITICAL)

```python
pipe = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression())
])

pipe.fit(df[num_cols + cat_cols], y)
```

Predict:

```python
y_pred = pipe.predict(X_test)
```

⚠ Pipeline prevents leakage.

---

# 3️⃣ Models

---

## Regression

### Linear Regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Coefficients:

```python
model.coef_
model.intercept_
```

---

## Classification

---

### Logistic Regression

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Probabilities:

```python
model.predict_proba(X_test)
```

---

### KNN

```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

⚠ Needs scaling.

---

### SVM

```python
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)
```

Linear:

```python
SVC(kernel="linear")
```

---

### Decision Tree

```python
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
```

---

### Random Forest

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)
```

Feature importance:

```python
model.feature_importances_
```

---

### Gradient Boosting

```python
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

---

# 4️⃣ Evaluation

---

## Predictions

```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
```

---

## Accuracy

```python
accuracy_score(y_test, y_pred)
```

---

## Precision / Recall / F1

```python
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
```

Full report:

```python
print(classification_report(y_test, y_pred))
```

---

## ROC-AUC

```python
roc_auc_score(y_test, y_proba)
```

⚠ Use probabilities.

---

## Confusion Matrix

```python
confusion_matrix(y_test, y_pred)
```

Normalized:

```python
cm = confusion_matrix(y_test, y_pred)
cm / cm.sum(axis=1, keepdims=True)
```

---

## Cross-Validation

```python
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring="f1"
)

scores.mean()
```

Stratified default for classification.

---

# 5️⃣ Hyperparameter Tuning

---

## GridSearch

```python
param_grid = {
    "n_estimators": [100,200],
    "max_depth": [None,5,10]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)
```

Best:

```python
grid.best_params_
grid.best_score_
```

---

## RandomizedSearch (Faster)

```python
param_dist = {
    "n_estimators": np.arange(50,300),
    "max_depth": [None,5,10,20]
}

search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=20,
    cv=5,
    random_state=42
)

search.fit(X_train, y_train)
```

---

## Tuning with Pipeline

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", SVC())
])

param_grid = {
    "model__C": [0.1,1,10],
    "model__kernel": ["linear","rbf"]
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

⚠ Use `model__param`.

---

# 🔥 Complete Template

```python
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("F1:", f1_score(y_test, y_pred))
```

---

# ⚡ Practical Rules

✔ Always stratify classification
✔ Always use Pipeline
✔ Scale for KNN/SVM/LogReg
✔ Trees don’t need scaling
✔ Use cross-validation
✔ Tune hyperparameters
✔ Don’t peek at test set

---

# 🔥 Mastery Outcome

If you master this page:

- You can build any classical ML pipeline
- You avoid leakage
- You evaluate properly
- You tune efficiently
- You understand model tradeoffs

---
