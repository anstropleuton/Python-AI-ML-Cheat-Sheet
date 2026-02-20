# PAGE 5 — Visualization & EDA

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# 1️⃣ Matplotlib (Low-Level Control)

---

## Basic Setup

```python
plt.figure(figsize=(6,4))
plt.plot([1,2,3], [1,4,9])
plt.show()
```

Add labels:

```python
plt.xlabel("x")
plt.ylabel("y")
plt.title("Title")
plt.legend()
```

Grid:

```python
plt.grid(True)
```

---

## Line Plot (Time Series)

```python
plt.plot(df["date"], df["value"], label="value")
plt.xticks(rotation=45)
plt.tight_layout()
```

Multiple lines:

```python
plt.plot(x, y1, label="y1")
plt.plot(x, y2, label="y2")
plt.legend()
```

---

## Scatter (Feature vs Target)

```python
plt.scatter(df["age"], df["income"], alpha=0.5)
```

Color by class:

```python
plt.scatter(X[:,0], X[:,1], c=y, cmap="viridis")
plt.colorbar()
```

---

## Histogram (Distribution Check)

```python
plt.hist(df["age"], bins=30)
```

Density:

```python
plt.hist(df["age"], bins=30, density=True)
```

---

## Subplots (Compare Many Features)

```python
fig, ax = plt.subplots(2,2, figsize=(8,6))
ax[0,0].hist(df["age"])
ax[0,1].hist(df["income"])
plt.tight_layout()
```

Loop:

```python
fig, axes = plt.subplots(3,3, figsize=(8,8))
for i, col in enumerate(df.columns[:9]):
    axes[i//3, i%3].hist(df[col])
```

---

## Styles

```python
plt.style.use("ggplot")
```

List:

```python
plt.style.available
```

---

# 2️⃣ Seaborn (High-Level EDA)

```python
sns.set(style="whitegrid")
```

---

## Heatmap (Correlation)

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm")
```

With annotations:

```python
sns.heatmap(corr, annot=True, fmt=".2f")
```

---

## Pairplot (Quick Feature Exploration)

```python
sns.pairplot(df, hue="target")
```

Numeric only:

```python
sns.pairplot(df.select_dtypes("number"))
```

⚠ Slow for large data.

---

## Distribution Plots

Histogram + KDE:

```python
sns.histplot(df["age"], kde=True)
```

Boxplot:

```python
sns.boxplot(x="target", y="income", data=df)
```

Violin:

```python
sns.violinplot(x="target", y="income", data=df)
```

---

## Scatter with Regression Line

```python
sns.regplot(x="age", y="income", data=df)
```

Grouped:

```python
sns.lmplot(x="age", y="income", hue="target", data=df)
```

---

## Count Plot (Categorical Balance)

```python
sns.countplot(x="target", data=df)
```

---

# 3️⃣ EDA CHECKLIST (DO THIS EVERY PROJECT)

---

# 🔎 Target Distribution

Classification:

```python
df["target"].value_counts()
sns.countplot(x="target", data=df)
```

Regression:

```python
sns.histplot(df["target"], kde=True)
```

Check imbalance.

---

# 🕳 Missing Pattern

```python
df.isna().sum().sort_values(ascending=False)
```

Visual:

```python
sns.heatmap(df.isna(), cbar=False)
```

Feature missing vs target:

```python
df.groupby(df["feature"].isna())["target"].mean()
```

---

# 📈 Feature vs Target

Numeric:

```python
sns.scatterplot(x="feature", y="target", data=df)
```

Boxplot by class:

```python
sns.boxplot(x="target", y="feature", data=df)
```

Group mean:

```python
df.groupby("target")["feature"].mean()
```

---

# 🚨 Outliers

Z-score:

```python
z = (df["feature"] - df["feature"].mean()) / df["feature"].std()
df[np.abs(z) > 3]
```

IQR:

```python
q1 = df["feature"].quantile(0.25)
q3 = df["feature"].quantile(0.75)
iqr = q3 - q1

df[(df["feature"] < q1 - 1.5*iqr) |
   (df["feature"] > q3 + 1.5*iqr)]
```

Visual:

```python
sns.boxplot(y=df["feature"])
```

---

# 🔗 Correlation

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr)
```

Highly correlated features:

```python
corr[abs(corr) > 0.9]
```

Drop multicollinearity:

```python
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
df = df.drop(columns=to_drop)
```

---

# 🧨 Leakage Detection

Red flags:

- Feature almost perfectly correlated with target
- Post-event information
- ID-like columns

Check:

```python
corr["target"].sort_values(ascending=False)
```

If correlation ≈ 1 → suspicious.

---

# 🧠 Distribution Shift Check (Train vs Test)

```python
sns.kdeplot(train["feature"], label="train")
sns.kdeplot(test["feature"], label="test")
plt.legend()
```

---

# 📊 Feature Importance Quick Check (Tree Model)

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

pd.Series(model.feature_importances_,
          index=df.drop("target", axis=1).columns
          ).sort_values().plot(kind="barh")
```

---

# 🔥 Fast EDA Template

```python
df = pd.read_csv("data.csv")

print(df.shape)
print(df.dtypes)

sns.countplot(x="target", data=df)

corr = df.corr(numeric_only=True)
sns.heatmap(corr)

for col in df.select_dtypes("number"):
    sns.histplot(df[col])
    plt.show()
```

---

# ⚡ Performance Notes

✔ Sample large data:

```python
df_sample = df.sample(10000)
```

✔ Avoid pairplot on 100k rows
✔ Use `.select_dtypes()`
✔ Disable annotation on huge heatmaps

---

# 🔥 Mastery Outcome

If you master this page:

- You detect leakage early
- You understand feature relationships fast
- You diagnose imbalance
- You detect outliers properly
- You prevent training garbage models

---
