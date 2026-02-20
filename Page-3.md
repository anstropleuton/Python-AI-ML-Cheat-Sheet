# PAGE 3 — NumPy & Numerical Computing

```python
import numpy as np
```

---

# 1️⃣ ndarray BASICS

---

## Creation

### From Python Objects

```python
a = np.array([1,2,3])
b = np.array([[1,2],[3,4]])
```

Specify dtype:

```python
np.array([1,2,3], dtype=np.float32)
```

---

### Zeros / Ones

```python
np.zeros((3,4))
np.ones((2,2))
np.eye(3)           # identity
```

---

### arange (step-based)

```python
np.arange(0,10,2)
```

### linspace (count-based)

```python
np.linspace(0,1,5)
```

---

### Random (Quick)

```python
np.random.rand(3,4)        # uniform [0,1)
np.random.randn(3,4)       # normal(0,1)
np.random.randint(0,10,(2,3))
```

---

## Shape & Structure

```python
a.shape
a.ndim
a.size
a.dtype
```

---

### Reshape

```python
a = np.arange(12)
a.reshape(3,4)
a.reshape(-1,2)     # infer dimension
```

⚠ Must preserve total size.

---

### Flatten

```python
a.flatten()     # copy
a.ravel()       # view (if possible)
```

---

### Transpose

```python
A.T
np.transpose(A)
A.swapaxes(0,1)
```

For 3D:

```python
A.transpose(0,2,1)
```

---

# 2️⃣ Indexing (CRITICAL FOR ML)

---

## Basic Slicing

```python
a = np.arange(10)

a[0]
a[2:5]
a[:3]
a[-1]
a[::2]
```

2D:

```python
A[0,1]
A[:,0]
A[1,:]
A[:2,:2]
```

---

## Boolean Masking

```python
a = np.array([1,2,3,4])
mask = a > 2
a[mask]
```

Inline:

```python
a[a % 2 == 0]
```

Modify:

```python
a[a < 0] = 0
```

---

## Fancy Indexing

```python
a = np.array([10,20,30,40])
a[[0,3,1]]
```

2D:

```python
A[[0,2], [1,3]]   # pairwise
```

Select rows:

```python
A[[0,2], :]
```

---

## Views vs Copies (VERY IMPORTANT)

Slice → view:

```python
b = a[2:5]
b[0] = 100   # modifies original
```

Fancy → copy:

```python
b = a[[1,3]]
```

Check:

```python
b.base is a
```

---

# 3️⃣ Vectorized Operations

---

## Element-wise

```python
a + b
a - b
a * b
a / b
a ** 2
np.sqrt(a)
np.exp(a)
np.log(a)
```

Comparison:

```python
a > 0
```

---

## Broadcasting Rules (Essential)

Rule:

- Align from right
- Dimensions equal OR one is 1

Example:

```python
X = np.random.randn(5,3)
b = np.random.randn(3)

X + b
```

Column vector:

```python
b = np.random.randn(5,1)
X + b
```

Add scalar:

```python
X + 10
```

Force reshape:

```python
b.reshape(1,-1)
```

---

## Matrix Multiplication

```python
A @ B
np.matmul(A,B)
```

Dot product (1D):

```python
np.dot(a,b)
a @ b
```

Batch matmul (3D):

```python
np.matmul(A,B)
```

---

## `einsum` (Advanced Weapon)

Dot:

```python
np.einsum("i,i->", a, b)
```

Matrix multiply:

```python
np.einsum("ij,jk->ik", A, B)
```

Batch:

```python
np.einsum("bij,bjk->bik", A, B)
```

Sum axis:

```python
np.einsum("ij->i", A)
```

---

# 4️⃣ Aggregations

---

## Basic

```python
a.sum()
a.mean()
a.std()
a.min()
a.max()
```

---

## Axis Logic

Given:

```python
X.shape = (5,3)
```

```python
X.sum(axis=0)   # column-wise
X.sum(axis=1)   # row-wise
```

---

## keepdims

```python
X.mean(axis=1, keepdims=True)
```

Keeps shape for broadcasting.

---

## Cumulative

```python
np.cumsum(a)
np.cumprod(a)
```

---

## Arg Ops

```python
np.argmax(a)
np.argmin(a)
np.argsort(a)
```

---

# 5️⃣ Random (ML Critical)

---

## Set Seed

```python
np.random.seed(42)
```

⚠ Global state (legacy API)

Modern:

```python
rng = np.random.default_rng(42)
rng.normal(size=(3,3))
```

---

## Distributions

```python
np.random.normal(0,1,(3,3))
np.random.uniform(0,1,(3,3))
np.random.binomial(10,0.5,100)
```

---

## Sampling

```python
np.random.choice(10, size=5)
np.random.choice(a, size=3, replace=False)
```

Shuffle:

```python
np.random.shuffle(a)
```

---

# 6️⃣ Linear Algebra Essentials

```python
from numpy import linalg as LA
```

---

## Dot Product

```python
a @ b
np.dot(a,b)
```

---

## Norm

```python
LA.norm(a)
LA.norm(a, ord=1)
```

---

## Inverse

```python
LA.inv(A)
```

⚠ Prefer solve:

```python
LA.solve(A,b)
```

---

## Determinant

```python
LA.det(A)
```

---

## Eigenvalues

```python
vals, vecs = LA.eig(A)
```

Symmetric matrix:

```python
LA.eigh(A)
```

---

## SVD (VERY IMPORTANT IN ML)

```python
U, S, Vt = LA.svd(A)
```

Reconstruct:

```python
U @ np.diag(S) @ Vt
```

---

## Pseudo-inverse

```python
LA.pinv(A)
```

---

# 🔥 Performance Micro-Rules

✔ Use vectorization
✔ Avoid Python loops
✔ Prefer `@` over manual loops
✔ Use broadcasting
✔ Use `float32` for deep learning
✔ Avoid repeated reshapes inside loop

Bad:

```python
for i in range(n):
    X = X.reshape(...)
```

---

# 🔥 Mental Model Summary

NumPy =

- Homogeneous typed arrays
- C-level speed
- Vectorized computation
- Broadcasting for free parallelism
- Linear algebra backbone of ML

Master this →
You unlock:

- Scikit-learn internals
- PyTorch tensor intuition
- Gradient math
- Efficient preprocessing

---
