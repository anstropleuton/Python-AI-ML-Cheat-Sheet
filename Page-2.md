# PAGE 2 — Advanced Python for AI/ML

---

## 1️⃣ OOP (Only What Matters for ML Code)

### Basic Class

```python
class Model:
    def __init__(self, lr: float):
        self.lr = lr
        self.weights = None

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

### `__repr__` (Debugging Essential)

```python
def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr})"
```

### Instance vs Class Variables

```python
class A:
    count = 0          # class var

    def __init__(self):
        A.count += 1   # shared
```

---

### `@staticmethod`

No access to instance.

```python
class Utils:
    @staticmethod
    def normalize(x):
        return (x - min(x)) / (max(x) - min(x))
```

### `@classmethod`

Access class.

```python
class Model:
    def __init__(self, lr):
        self.lr = lr

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(lr=cfg["lr"])
```

---

### Common Dunder Methods

```python
__len__
__getitem__
__setitem__
__call__
__iter__
__add__
__eq__
```

Example:

```python
class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

Callable object:

```python
class Linear:
    def __call__(self, x):
        return x * 2
```

---

### `@dataclass` (Clean Config Objects)

```python
from dataclasses import dataclass

@dataclass
class Config:
    lr: float
    epochs: int = 10
```

With defaults:

```python
@dataclass(frozen=True)
class Params:
    dim: int
```

---

### Composition > Inheritance

Prefer:

```python
class Trainer:
    def __init__(self, model):
        self.model = model
```

Avoid deep inheritance chains.

---

## 2️⃣ Modules & Packaging

### Project Structure (ML Template)

```
project/
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
│
├── tests/
├── pyproject.toml
└── README.md
```

### Absolute Import (Preferred)

```python
from src.model import Model
```

### Relative Import (Inside Package)

```python
from .model import Model
```

---

### `__init__.py`

Expose clean API:

```python
from .model import Model
from .data import load_data
```

---

### Editable Install (Dev Mode)

```bash
pip install -e .
```

Now import anywhere:

```python
from project.model import Model
```

---

## 3️⃣ Performance

### List vs Generator

List:

```python
[x*x for x in range(10_000_000)]
```

Generator:

```python
(x*x for x in range(10_000_000))
```

Memory difference:

- List → stores everything
- Generator → lazy

---

### Timing

#### `time`

```python
import time
t0 = time.time()
run()
print(time.time() - t0)
```

#### `timeit`

```python
import timeit
timeit.timeit("sum(range(1000))", number=1000)
```

#### `cProfile`

```bash
python -m cProfile train.py
```

Inside:

```python
import cProfile
cProfile.run("train()")
```

---

### Memory Profiling (Quick Trick)

```python
import sys
sys.getsizeof(obj)
```

Better:

```bash
pip install memory-profiler
```

---

### Big-O Quick Intuition

| Operation    | Complexity |
| ------------ | ---------- |
| list append  | O(1)       |
| dict lookup  | O(1) avg   |
| sorting      | O(n log n) |
| nested loops | O(n²)      |

Avoid:

```python
for x in list:
    if x in list:  # O(n²)
```

Use set:

```python
s = set(list)
```

---

## 4️⃣ Decorators (Power Feature)

### Basic Pattern

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper
```

Usage:

```python
@decorator
def train():
    ...
```

---

### Timing Decorator

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        print(time.time() - t0)
        return out
    return wrapper
```

---

### Decorator with Arguments

```python
def repeat(n):
    def deco(func):
        def wrapper(*a, **kw):
            for _ in range(n):
                result = func(*a, **kw)
            return result
        return wrapper
    return deco

@repeat(3)
def f():
    print("hi")
```

---

### Preserve Metadata

```python
from functools import wraps

def deco(func):
    @wraps(func)
    def wrapper(*a, **kw):
        return func(*a, **kw)
    return wrapper
```

---

## 5️⃣ Typing (Useful Subset for ML)

```python
from typing import List, Dict, Tuple, Union, Optional, Callable
```

### Basic

```python
def f(x: List[int]) -> Dict[str, float]:
    ...
```

### Union

```python
def f(x: int | float) -> float:
    ...
```

### Optional

```python
def load(path: Optional[str] = None):
```

### Callable

```python
def train(loss_fn: Callable[[float, float], float]):
```

---

### TypedDict (Structured Dict)

```python
from typing import TypedDict

class Batch(TypedDict):
    x: list[float]
    y: list[int]
```

---

### Protocol (Duck Typing Contracts)

```python
from typing import Protocol

class ModelLike(Protocol):
    def predict(self, X) -> list: ...
```

Accepts any object implementing `predict`.

---

⚠ Type hints:

- Not runtime enforced
- Use `mypy` for checking

---

## 6️⃣ Async (Minimal, Practical)

Use when:

- API calls
- Multiple inference requests
- IO-bound workloads

---

### Basic

```python
import asyncio

async def fetch():
    await asyncio.sleep(1)
    return 42

async def main():
    result = await fetch()
    print(result)

asyncio.run(main())
```

---

### Parallel Tasks

```python
async def main():
    tasks = [fetch() for _ in range(5)]
    results = await asyncio.gather(*tasks)
```

---

Blocking inside async = BAD:

```python
time.sleep(1)  # blocks event loop
```

---

## 7️⃣ Numpy Mental Model (Critical for ML)

### Why Arrays > Python Loops

Slow:

```python
result = []
for x in xs:
    result.append(x * 2)
```

Fast (vectorized):

```python
import numpy as np
xs = np.array(xs)
result = xs * 2
```

Reason:

- C backend
- SIMD
- No Python loop overhead

---

### Broadcasting Intuition

Shapes:

```
(5,3) + (3,) → works
(5,3) + (5,) → error
```

Rule:

- Compare from right
- Equal OR one is 1

Example:

```python
X = np.random.randn(5,3)
b = np.random.randn(3)
X + b
```

---

### Axis Logic

2D:

```
axis=0 → down rows
axis=1 → across columns
```

```python
X.mean(axis=0)  # per column
```

---

### Vectorization Mindset

Replace:

```python
for i in range(n):
    y[i] = a*x[i] + b
```

With:

```python
y = a*x + b
```

Avoid Python loops in ML core math.

---

# 🔥 PAGE 2 MASTERY CHECK

If you master this page:

- You write clean ML modules
- You profile bottlenecks
- You structure reusable packages
- You use decorators for logging/timing
- You understand vectorization
- You prepare for Numpy / PyTorch / Sklearn internals

---
