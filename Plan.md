Design assumptions:

- 2-column layout
- Tiny font, minimal spacing
- Code-heavy
- Minimal theory (only what unlocks practice)
- Max coverage, maximal surface area
- Focus: doing, not philosophizing

---

# PAGE 1 BLUEPRINT — Python Core Essentials for AI/ML

### 1. Syntax & Core Constructs (Ultra Fast Recap)

- Variables, dynamic typing
- Multiple assignment
- f-strings
- Ternary expressions
- Walrus operator `:=`
- Truthy/falsy values
- `None` vs `False`
- Type hints (`typing` basics)

### 2. Data Structures (Performance-Oriented)

- `list`, `tuple`, `set`, `dict`
- Comprehensions (list/dict/set)
- Nested comprehensions
- `collections`:
  - `defaultdict`
  - `Counter`
  - `deque`
  - `namedtuple`

- When to use tuple vs list
- Dict merging (`|`)

### 3. Functions

- `*args`, `**kwargs`
- Keyword-only args
- Default mutable arg trap
- Lambda
- Closures
- Function annotations
- Docstrings (minimal)

### 4. Iteration & Functional Patterns

- `enumerate`
- `zip`
- `map`, `filter`
- `any`, `all`
- `sorted` with `key`
- Custom key functions
- `itertools`:
  - `product`
  - `permutations`
  - `combinations`
  - `chain`
  - `cycle`
  - `islice`

### 5. Exception Handling

- `try/except/else/finally`
- Custom exceptions
- Raising errors properly

### 6. File IO

- Reading/writing text
- Reading JSON
- CSV basics
- `pathlib`
- Context managers

### 7. Virtual Environments

- `venv`
- `pip`
- `requirements.txt`
- `pyproject.toml` minimal intro

---

# PAGE 2 BLUEPRINT — Advanced Python for AI/ML

### 1. OOP (Only What Matters)

- Classes
- `__init__`
- `__repr__`
- `@staticmethod`
- `@classmethod`
- Dunder methods
- Data classes (`@dataclass`)
- Composition vs inheritance

### 2. Modules & Packaging

- Project structure
- `__init__.py`
- Relative imports
- Absolute imports
- Editable installs

### 3. Performance

- List vs generator
- Memory profiling
- Time profiling:
  - `time`
  - `timeit`
  - `cProfile`

- Big-O quick intuition

### 4. Decorators

- Basic decorator pattern
- Decorator with arguments
- Use cases: logging, timing

### 5. Typing (Useful Subset)

- `List`, `Dict`, `Tuple`
- `Union`
- `Optional`
- `Callable`
- `TypedDict`
- `Protocol`
- Runtime checking caveat

### 6. Async (Minimal)

- `async def`
- `await`
- `asyncio.run`
- When to use (APIs, inference servers)

### 7. Numpy Mental Model

- Why arrays beat Python loops
- Broadcasting intuition
- Vectorization mindset

---

# PAGE 3 BLUEPRINT — Numpy & Numerical Computing

### 1. ndarray Basics

- Creation:
  - `array`
  - `zeros`
  - `ones`
  - `arange`
  - `linspace`
  - `random`

- Shape
- Reshape
- Flatten
- Transpose

### 2. Indexing

- Basic slicing
- Boolean masking
- Fancy indexing
- Views vs copies

### 3. Vectorized Ops

- Element-wise ops
- Broadcasting rules
- Matrix multiplication
- `einsum`

### 4. Aggregations

- `sum`, `mean`, `std`
- `axis` logic
- `keepdims`

### 5. Random

- Seeds
- Normal/uniform
- Random sampling

### 6. Linear Algebra

- Dot product
- Inverse
- Determinant
- Eigenvalues
- SVD

---

# PAGE 4 BLUEPRINT — Pandas + Data Handling

### 1. DataFrame Basics

- Creation
- From CSV
- From dict
- Column selection
- Row selection
- `.loc`, `.iloc`

### 2. Cleaning

- Missing values
- `fillna`
- `dropna`
- Replace
- Rename
- Apply

### 3. Filtering

- Boolean indexing
- Query
- Sorting

### 4. Groupby

- Aggregation
- Multi-index
- Pivot tables

### 5. Feature Engineering

- Encoding categories
- One-hot
- Binning
- Datetime features

### 6. Performance Tricks

- Vectorization
- Avoid loops
- `astype` for memory

---

# PAGE 5 BLUEPRINT — Visualization & EDA

### 1. Matplotlib

- Line
- Scatter
- Histogram
- Subplots
- Styles

### 2. Seaborn

- Heatmap
- Pairplot
- Distribution plots
- Correlation matrix

### 3. EDA Checklist

- Target distribution
- Missing pattern
- Feature-target relationship
- Outliers
- Correlation
- Leakage detection

---

# PAGE 6 BLUEPRINT — Scikit-Learn Workflow

### 1. Train/Test Split

- `train_test_split`
- Stratify

### 2. Preprocessing

- Scaling:
  - StandardScaler
  - MinMaxScaler

- Encoding
- Pipelines
- ColumnTransformer

### 3. Models

- Linear Regression
- Logistic Regression
- KNN
- SVM
- Decision Trees
- Random Forest
- Gradient Boosting

### 4. Evaluation

- Accuracy
- Precision/Recall
- F1
- ROC-AUC
- Confusion matrix
- Cross-validation

### 5. Hyperparameter Tuning

- GridSearch
- RandomizedSearch
- CV strategy

---

# PAGE 7 BLUEPRINT — Deep Learning with PyTorch

### 1. Tensors

- Creation
- Device (CPU/GPU)
- Autograd

### 2. Model Definition

- `nn.Module`
- `forward`
- Sequential vs custom

### 3. Training Loop

- Loss
- Optimizer
- Backprop
- Zero grad
- Epoch structure

### 4. Dataset & DataLoader

- Custom Dataset
- Batching
- Shuffle

### 5. Saving/Loading

- `state_dict`
- Checkpoints

### 6. Debugging Tips

- Overfit small batch
- Check gradients
- Print shapes

---

# PAGE 8 BLUEPRINT — NLP & Transformers

### 1. Text Preprocessing

- Tokenization
- Lowercasing
- Stopwords
- Lemmatization

### 2. Vectorization

- Bag of words
- TF-IDF
- Word embeddings

### 3. Transformers

- Tokenizer
- Model loading
- Inference
- Fine-tuning skeleton

### 4. Prompt Engineering Basics

- Zero-shot
- Few-shot
- System prompts

---

# PAGE 9 BLUEPRINT — MLOps & Deployment

### 1. Model Saving

- Pickle
- Joblib
- Torch save

### 2. REST API

- FastAPI minimal server
- Inference endpoint
- Request validation

### 3. Docker Basics

- Dockerfile structure
- Build & run

### 4. Experiment Tracking

- MLflow
- Logging metrics
- Logging artifacts

### 5. Model Versioning

- Checkpoints
- Naming conventions

---

# PAGE 10 BLUEPRINT — Full Workflow Blueprint

### 1. End-to-End Pipeline

- Problem definition
- Data ingestion
- EDA
- Feature engineering
- Modeling
- Validation
- Deployment

### 2. Common Failure Modes

- Data leakage
- Overfitting
- Underfitting
- Distribution shift

### 3. Production Concerns

- Latency
- Monitoring
- Drift detection
- Retraining loops

### 4. Scaling

- Batching
- Parallelism
- GPU usage

### 5. Cheat Sheet Appendix

- Debug checklist
- ML project template
- Minimal starter repo structure
- Essential libraries list
