# PAGE 4 — Pandas + Data Handling

```python
import pandas as pd
import numpy as np
```

---

# 1️⃣ DataFrame Basics

---

## Creation

### From Dict

```python
df = pd.DataFrame({
    "age": [20,25,30],
    "income": [1000,2000,3000]
})
```

### From List of Dict

```python
data = [{"a":1,"b":2},{"a":3,"b":4}]
df = pd.DataFrame(data)
```

---

## From CSV

```python
df = pd.read_csv("data.csv")
```

Common params:

```python
pd.read_csv("data.csv",
            index_col=0,
            parse_dates=["date"],
            dtype={"age":"int32"},
            na_values=["NA","?"])
```

Write:

```python
df.to_csv("out.csv", index=False)
```

---

## Inspect

```python
df.head()
df.tail()
df.shape
df.columns
df.dtypes
df.info()
df.describe()
```

---

## Column Selection

```python
df["age"]          # Series
df[["age","income"]]  # DataFrame
```

Assign:

```python
df["new"] = df["age"] * 2
```

---

## Row Selection

### `.iloc` (position)

```python
df.iloc[0]
df.iloc[0:3]
df.iloc[:, 0]
df.iloc[0,1]
```

### `.loc` (label)

```python
df.loc[0]
df.loc[0:2]
df.loc[:, "age"]
df.loc[df["age"] > 20]
```

---

# 2️⃣ Cleaning

---

## Missing Values

```python
df.isna()
df.isna().sum()
df.notna()
```

---

### `fillna`

```python
df["age"].fillna(0)
df.fillna({"age":0, "income":df["income"].mean()})
df.fillna(method="ffill")
```

---

### `dropna`

```python
df.dropna()
df.dropna(axis=1)
df.dropna(subset=["age"])
```

---

### Replace

```python
df.replace("?", np.nan)
df["col"].replace({0:np.nan})
```

---

### Rename

```python
df.rename(columns={"old":"new"}, inplace=True)
df.rename(index={0:"row0"})
```

---

### Apply (Use Carefully)

Row-wise:

```python
df.apply(lambda row: row["a"] + row["b"], axis=1)
```

Column-wise:

```python
df.apply(np.mean)
```

⚠ Slow if heavy logic.

Vectorized preferred:

```python
df["a"] + df["b"]
```

---

# 3️⃣ Filtering

---

## Boolean Indexing

```python
df[df["age"] > 25]
df[(df["age"] > 20) & (df["income"] < 3000)]
```

OR:

```python
df[(df["a"] > 1) | (df["b"] < 5)]
```

---

## `query` (Cleaner Syntax)

```python
df.query("age > 20 and income < 3000")
```

Using variables:

```python
threshold = 25
df.query("age > @threshold")
```

---

## Sorting

```python
df.sort_values("age")
df.sort_values(["age","income"], ascending=[True,False])
df.sort_index()
```

---

# 4️⃣ Groupby (Feature Engineering Core)

---

## Basic Aggregation

```python
df.groupby("category")["income"].mean()
```

Multiple:

```python
df.groupby("cat").agg({
    "income":["mean","sum"],
    "age":"max"
})
```

Named:

```python
df.groupby("cat").agg(
    avg_income=("income","mean"),
    max_age=("age","max")
)
```

---

## Multi-index Result

```python
g = df.groupby(["cat","region"])["income"].mean()
g.reset_index()
```

Flatten:

```python
g.columns = ["_".join(col) for col in g.columns]
```

---

## Transform (Return Same Shape)

```python
df["group_mean"] = df.groupby("cat")["income"].transform("mean")
```

---

## Pivot Table

```python
pd.pivot_table(df,
               values="income",
               index="cat",
               columns="region",
               aggfunc="mean")
```

Equivalent:

```python
df.pivot_table(...)
```

---

# 5️⃣ Feature Engineering

---

## Encoding Categories

### Label Encoding

```python
df["cat"] = df["cat"].astype("category")
df["cat_code"] = df["cat"].cat.codes
```

---

### One-Hot Encoding

```python
pd.get_dummies(df, columns=["cat"])
```

Drop first:

```python
pd.get_dummies(df, drop_first=True)
```

---

## Binning (Discretization)

```python
pd.cut(df["age"], bins=5)
pd.qcut(df["income"], q=4)
```

Custom:

```python
pd.cut(df["age"], bins=[0,18,30,50,100])
```

---

## Datetime Features

```python
df["date"] = pd.to_datetime(df["date"])
```

Extract:

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
```

Time diff:

```python
df["delta"] = df["date"] - df["date"].min()
```

---

## Rolling (Time Series Prep)

```python
df["rolling_mean"] = df["value"].rolling(7).mean()
df["expanding"] = df["value"].expanding().mean()
```

---

## Shift (Lag Features)

```python
df["lag1"] = df["value"].shift(1)
```

---

# 6️⃣ Performance Tricks

---

## Vectorization > Apply > Loop

Bad:

```python
for i,row in df.iterrows():
    ...
```

Better:

```python
df["c"] = df["a"] + df["b"]
```

Avoid:

```python
df.apply(...)
```

Unless necessary.

---

## Use `astype` for Memory

```python
df["age"] = df["age"].astype("int32")
df["cat"] = df["cat"].astype("category")
```

Check memory:

```python
df.memory_usage(deep=True)
```

---

## Efficient Filtering

Convert strings to category:

```python
df["cat"] = df["cat"].astype("category")
```

---

## Avoid Chained Assignment

Bad:

```python
df[df["a"] > 0]["b"] = 1
```

Correct:

```python
df.loc[df["a"] > 0, "b"] = 1
```

---

## Read Large CSV Efficiently

```python
pd.read_csv("big.csv", chunksize=10000)
```

Select columns:

```python
pd.read_csv("big.csv", usecols=["a","b"])
```

---

# 🔥 ML Workflow Pattern in Pandas

```python
df = pd.read_csv("data.csv")

df = df.dropna(subset=["target"])
df["cat"] = df["cat"].astype("category")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("target", axis=1).values
y = df["target"].values
```

---

# 🔥 Mastery Outcome

If you master this page:

- You clean messy datasets
- You engineer features fast
- You avoid performance traps
- You prepare data for sklearn / torch
- You understand real-world data wrangling

---
