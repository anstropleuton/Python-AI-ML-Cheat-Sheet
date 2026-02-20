# PAGE 7 — Deep Learning with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

---

# 1️⃣ Tensors

---

## Creation

From list:

```python
x = torch.tensor([1,2,3], dtype=torch.float32)
```

Zeros / Ones:

```python
torch.zeros(3,4)
torch.ones(2,2)
torch.eye(3)
```

Random:

```python
torch.randn(3,4)      # normal
torch.rand(3,4)       # uniform
torch.randint(0,10,(2,3))
```

Like:

```python
torch.zeros_like(x)
torch.randn_like(x)
```

---

## Shape Ops

```python
x.shape
x.view(3,4)           # reshape (contiguous)
x.reshape(3,4)
x.flatten()
x.permute(1,0)
x.unsqueeze(0)
x.squeeze()
```

---

## Device (CPU / GPU)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device)
model = model.to(device)
```

⚠ Always move both data and model.

---

## NumPy ↔ Torch

```python
x.numpy()                   # CPU only
torch.from_numpy(arr)
```

---

## Autograd

```python
x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x
y.backward()

x.grad
```

Disable grad:

```python
with torch.no_grad():
    ...
```

Detach:

```python
x_det = x.detach()
```

---

# 2️⃣ Model Definition

---

## Basic `nn.Module`

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

Instantiate:

```python
model = MLP(10, 64, 1)
```

---

## Sequential (Quick Prototype)

```python
model = nn.Sequential(
    nn.Linear(10,64),
    nn.ReLU(),
    nn.Linear(64,1)
)
```

Use custom when:

- Multiple inputs
- Skip connections
- Complex logic

---

## Inspect

```python
print(model)
sum(p.numel() for p in model.parameters())
```

Freeze:

```python
for p in model.parameters():
    p.requires_grad = False
```

---

# 3️⃣ Training Loop (Core Pattern)

---

## Setup

```python
model = MLP(10,64,1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

---

## One Training Step

```python
model.train()

optimizer.zero_grad()

outputs = model(X_batch)
loss = criterion(outputs, y_batch)

loss.backward()

optimizer.step()
```

⚠ Order matters.

---

## Full Epoch Loop

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(epoch, total_loss/len(loader))
```

---

## Evaluation

```python
model.eval()

with torch.no_grad():
    preds = model(X_test.to(device))
```

⚠ Always disable grad in eval.

---

# 4️⃣ Dataset & DataLoader

---

## Custom Dataset

```python
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## DataLoader

```python
dataset = MyDataset(X_train, y_train)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
```

Loop:

```python
for xb, yb in loader:
    ...
```

---

## Train / Val Split

```python
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [800,200])
```

---

# 5️⃣ Saving / Loading

---

## Save Weights

```python
torch.save(model.state_dict(), "model.pt")
```

Load:

```python
model = MLP(10,64,1)
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

Load:

```python
ckpt = torch.load("ckpt.pt")

model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
```

---

# 6️⃣ Debugging & Sanity Tricks

---

## 1. Overfit Small Batch

```python
small_loader = DataLoader(dataset, batch_size=10)

# train until near-zero loss
```

If cannot overfit → bug.

---

## 2. Print Shapes

```python
print(X.shape)
print(outputs.shape)
```

Common error:

```
Expected (N, C) but got (C,)
```

---

## 3. Check Gradients

```python
for name, p in model.named_parameters():
    print(name, p.grad.abs().mean())
```

If grad = 0 → dead network.

---

## 4. Detect Exploding Gradients

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 5. Initialization

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)
```

---

## 6. Classification Setup

Binary:

```python
criterion = nn.BCEWithLogitsLoss()
```

Output:

```python
nn.Linear(hidden,1)
```

Multiclass:

```python
criterion = nn.CrossEntropyLoss()
```

Output:

```python
nn.Linear(hidden,num_classes)
```

⚠ Do NOT apply softmax before CrossEntropyLoss.

---

## 7. Accuracy

```python
pred = torch.argmax(outputs, dim=1)
acc = (pred == y_batch).float().mean()
```

Binary:

```python
pred = torch.sigmoid(outputs) > 0.5
```

---

# ⚡ Performance Tips

✔ Use `pin_memory=True` (GPU)
✔ Use larger batch for GPU
✔ Avoid CPU↔GPU transfer inside loop
✔ Use `.item()` only when needed
✔ Set `torch.backends.cudnn.benchmark = True`

---

# 🔥 Minimal Complete Template

```python
model = MLP(10,64,1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out,yb)
        loss.backward()
        optimizer.step()
```

---

# 🔥 Mastery Outcome

If you master this page:

- You can implement any deep learning model
- You understand autograd
- You can train on GPU
- You can debug exploding/vanishing gradients
- You can save & resume training

---
