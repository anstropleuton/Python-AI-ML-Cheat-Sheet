# PAGE 1 — Python Core Essentials for AI/ML

---

## 1️⃣ Syntax & Core Constructs (Ultra-Fast Recap)

### Variables & Dynamic Typing

```python
x = 10
x = "now string"     # dynamic typing
type(x)

a: int = 5           # type hint (not enforced at runtime)
b: float | None = 3.2
```

### Multiple Assignment

```python
a, b = 1, 2
a, b = b, a           # swap
x = y = 0             # chained
a, *rest = [1,2,3,4]  # unpacking
```

### f-Strings

```python
name = "AI"
score = 0.98765
f"{name=}, {score:.3f}"
```

### Ternary

```python
label = "pos" if y > 0 else "neg"
```

### Walrus `:=`

```python
if (n := len(data)) > 100:
    print(n)
```

### Truthy / Falsy

Falsy:

```python
False, None, 0, 0.0, "", [], {}, set()
```

### `None` vs `False`

```python
if x is None:  # identity check
if not x:      # falsy check (0, "", etc.)
```

### Type Hints (Minimal Useful Set)

```python
from typing import List, Dict, Tuple, Optional, Union, Callable

def f(x: int, y: float) -> float:
    return x + y

def g(xs: List[int]) -> Dict[str, float]:
    ...
```

---

## 2️⃣ Data Structures (Performance-Oriented)

### List

```python
lst = [1,2,3]
lst.append(4)
lst.extend([5,6])
lst.pop()
lst.sort(reverse=True)
```

### Tuple (immutable, hashable)

```python
t = (1,2,3)
d = {(1,2): "point"}  # tuple as dict key
```

Use tuple when:

- Fixed structure
- Dictionary key
- Want immutability

### Set

```python
s = {1,2,3}
s.add(4)
s.remove(2)
s1 & s2    # intersection
s1 | s2    # union
```

### Dict

```python
d = {"a":1, "b":2}
d["c"] = 3
d.get("x", 0)
d.keys(); d.values(); d.items()
```

### Dict Merge (3.9+)

```python
d3 = d1 | d2
```

---

### Comprehensions (Essential)

```python
[x*x for x in range(10)]
{x: x*x for x in range(5)}
{x for x in range(10) if x%2==0}
```

Nested:

```python
[(i,j) for i in range(3) for j in range(3)]
```

Conditional:

```python
[x if x%2==0 else 0 for x in range(10)]
```

---

### `collections` (AI-Friendly Tools)

```python
from collections import defaultdict, Counter, deque, namedtuple
```

#### defaultdict

```python
dd = defaultdict(int)
dd["a"] += 1
```

#### Counter

```python
c = Counter(["a","b","a"])
c.most_common(1)
```

#### deque (fast queue)

```python
dq = deque([1,2])
dq.appendleft(0)
dq.pop()
```

#### namedtuple

```python
Point = namedtuple("Point", "x y")
p = Point(1,2)
p.x
```

---

## 3️⃣ Functions

### Basic

```python
def f(x, y=0):
    return x + y
```

### `*args`, `**kwargs`

```python
def f(*args, **kwargs):
    print(args)
    print(kwargs)

f(1,2,a=3)
```

### Keyword-Only Args

```python
def f(x, *, lr=0.01):
    ...
```

### Default Mutable Trap

```python
def bad(lst=[]):   # DON'T
    lst.append(1)

def good(lst=None):
    if lst is None:
        lst = []
```

### Lambda

```python
f = lambda x: x**2
sorted(data, key=lambda x: x[1])
```

### Closures

```python
def outer(a):
    def inner(b):
        return a + b
    return inner

add5 = outer(5)
add5(3)
```

### Function Annotations

```python
def train(X: list[float]) -> float:
    ...
```

### Minimal Docstring

```python
def f(x):
    """Compute squared value."""
    return x*x
```

---

## 4️⃣ Iteration & Functional Patterns

### enumerate

```python
for i, val in enumerate(data):
    ...
```

### zip

```python
for x, y in zip(xs, ys):
    ...
```

### map / filter

```python
list(map(str, nums))
list(filter(lambda x: x>0, nums))
```

### any / all

```python
any(x > 0 for x in nums)
all(x > 0 for x in nums)
```

### sorted with key

```python
sorted(data, key=len)
sorted(data, key=lambda x: x[1])
```

Custom multi-key:

```python
sorted(data, key=lambda x: (x[1], -x[2]))
```

---

### itertools (Power Tools)

```python
import itertools as it
```

Cartesian product:

```python
list(it.product([1,2], [3,4]))
```

Permutations:

```python
list(it.permutations([1,2,3], 2))
```

Combinations:

```python
list(it.combinations([1,2,3], 2))
```

Chain:

```python
list(it.chain([1,2],[3,4]))
```

Cycle:

```python
c = it.cycle([1,2])
next(c)
```

Islice:

```python
list(it.islice(range(100), 5))
```

---

## 5️⃣ Exception Handling

### Basic Pattern

```python
try:
    risky()
except ValueError as e:
    print(e)
else:
    print("success")
finally:
    cleanup()
```

Multiple:

```python
except (TypeError, KeyError):
```

### Custom Exception

```python
class DataError(Exception):
    pass

raise DataError("bad dataset")
```

Re-raise:

```python
except Exception:
    raise
```

---

## 6️⃣ File IO (Data Pipeline Basics)

### Read Text

```python
with open("file.txt") as f:
    data = f.read()
```

Line-by-line:

```python
for line in open("file.txt"):
    ...
```

Write:

```python
with open("out.txt","w") as f:
    f.write("hello")
```

---

### JSON

```python
import json

with open("data.json") as f:
    obj = json.load(f)

json.dumps(obj)
```

---

### CSV

```python
import csv

with open("data.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        ...
```

---

### pathlib (modern path handling)

```python
from pathlib import Path

p = Path("data") / "file.txt"
p.exists()
p.read_text()
```

---

## 7️⃣ Virtual Environments (ML Hygiene)

### Create venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install

```bash
pip install numpy pandas scikit-learn
```

### Freeze

```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

### Minimal `pyproject.toml`

```toml
[project]
name = "ml-project"
version = "0.1.0"
dependencies = ["numpy", "pandas"]
```

---

# 🔥 PAGE 1 MENTAL MODEL

If you master this page:

- You can write clean ML scripts
- Build data pipelines
- Avoid common Python bugs
- Manipulate data efficiently
- Structure ML projects properly

---
