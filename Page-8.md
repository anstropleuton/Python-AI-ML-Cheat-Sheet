# PAGE 8 — NLP & Transformers

```python
# Core
import re
import numpy as np
```

```python
# Sklearn vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

```python
# HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
```

---

# 1️⃣ Text Preprocessing

---

## Basic Cleaning

```python
text = "Hello!!! This is NLP, version 2.0 :)"

text = text.lower()
text = re.sub(r"[^a-z0-9\s]", "", text)
```

---

## Tokenization (Manual)

```python
tokens = text.split()
```

---

## NLTK Example

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokens = word_tokenize(text)

stop_words = set(stopwords.words("english"))
tokens = [t for t in tokens if t not in stop_words]

lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(t) for t in tokens]
```

---

## When NOT to preprocess heavily

⚠ For transformers:

- DO NOT remove stopwords
- DO NOT lemmatize
- DO NOT lowercase (if model is cased)

Let tokenizer handle it.

---

# 2️⃣ Vectorization

---

## Bag of Words

```python
docs = ["i love ai", "ai loves python"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.vocabulary_)
print(X.toarray())
```

---

## TF-IDF

```python
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = tfidf.fit_transform(docs)
```

---

## N-grams

```python
CountVectorizer(ngram_range=(1,3))
```

---

## Word Embeddings (Static)

```python
# Using pretrained GloVe via gensim
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("glove.txt", binary=False)

vec = model["king"]
```

---

## Sentence Embeddings (Modern)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

emb = model.encode(["hello world"])
```

---

# 3️⃣ Transformers (HuggingFace)

---

## Load Tokenizer

```python
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## Tokenize

```python
inputs = tokenizer(
    "Hello world",
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(inputs["input_ids"])
print(inputs["attention_mask"])
```

---

## Load Model (Embeddings)

```python
model = AutoModel.from_pretrained(model_name)

outputs = model(**inputs)

last_hidden = outputs.last_hidden_state
cls_embedding = last_hidden[:,0,:]
```

---

## Sequence Classification

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

outputs = model(**inputs)
logits = outputs.logits
```

Prediction:

```python
pred = torch.argmax(logits, dim=1)
```

---

## Inference Template

```python
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    return probs
```

---

# 4️⃣ Fine-Tuning Skeleton

---

## Dataset Format

```python
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```

---

## Trainer API (Fastest Way)

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
```

---

## Manual PyTorch Loop (More Control)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for batch in loader:
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
```

---

## Freeze Base Model

```python
for param in model.base_model.parameters():
    param.requires_grad = False
```

---

# 5️⃣ Attention Intuition (Minimal)

Transformer input:

```
[input_ids]
[attention_mask]
[token_type_ids]
```

Output:

```
last_hidden_state  → token embeddings
pooler_output      → CLS pooled
logits             → classification
```

---

# 6️⃣ Prompt Engineering Basics

---

## Zero-Shot

```python
prompt = """
Classify sentiment:

Text: I love this movie
Sentiment:
"""
```

---

## Few-Shot

```python
prompt = """
Classify sentiment:

Text: I hate it
Sentiment: Negative

Text: Amazing experience
Sentiment: Positive

Text: It was fine
Sentiment:
"""
```

---

## System Prompt (LLMs)

```
You are a helpful AI assistant specialized in finance.
Always answer concisely.
```

---

## Instruction Format

```
Instruction:
Summarize the text.

Input:
<text>

Output:
```

---

## Chain-of-Thought (When Needed)

```
Solve step by step.
Explain reasoning before final answer.
```

⚠ Use only when reasoning tasks required.

---

# 7️⃣ Practical NLP Workflow

1. Define task
2. Choose baseline (TF-IDF + Logistic)
3. Upgrade → Transformer
4. Fine-tune
5. Evaluate
6. Optimize inference

---

# 8️⃣ Common Pitfalls

❌ Forget truncation
❌ Batch too large → OOM
❌ Applying softmax before CrossEntropy
❌ Over-cleaning text for transformers
❌ Not using attention_mask

---

# ⚡ Performance Tips

✔ Use `torch.cuda.amp.autocast()`
✔ Use `gradient_accumulation_steps`
✔ Use smaller models first
✔ Cache tokenized dataset
✔ Use `datasets` library

---

# 🔥 Minimal End-to-End Template

```python
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = TextDataset(texts, labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

---

# 🔥 Mastery Outcome

If you master this page:

- You can build NLP pipelines
- You understand TF-IDF vs embeddings
- You can fine-tune transformers
- You can deploy inference
- You understand prompting basics

---
