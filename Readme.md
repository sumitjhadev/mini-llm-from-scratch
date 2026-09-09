# 🧠 Mini LLM From Scratch

### A Modern Transformer Language Model Built From First Principles in PyTorch

> **A compact 2.76M-parameter autoregressive language model implementing modern Transformer components — built from scratch, trained on Shakespeare, and capable of generating text character-by-character.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
</p>

---

## ✨ What Is This?

This project is an **LLM implementation built from scratch** to understand what happens inside a modern language model — from tokenization and attention to autoregressive generation.

Instead of relying on high-level Transformer libraries, the core architecture is implemented directly using **PyTorch tensors and neural network primitives**.

The model combines several techniques used in modern LLM architectures:

* 🔤 Character-level tokenization
* ⚡ Grouped Query Attention (GQA)
* 🔄 Rotary Positional Embeddings (RoPE)
* 📐 RMSNorm
* 🧩 SwiGLU Feed Forward Network
* 🔗 Weight Tying
* 🎯 Causal Language Modeling
* 📝 Autoregressive Text Generation

---

# 🏗️ Architecture

The model follows a modern Transformer decoder architecture:

```text
                     Input Text
                         │
                         ▼
              Character-Level Tokens
                         │
                         ▼
                 Token Embeddings
                         │
                         ▼
                    ┌─────────┐
                    │ RMSNorm │
                    └────┬────┘
                         │
                         ▼
              Grouped Query Attention
                         │
                    + RoPE
                         │
                         ▼
                 Residual Connection
                         │
                         ▼
                    ┌─────────┐
                    │ RMSNorm │
                    └────┬────┘
                         │
                         ▼
                       SwiGLU
                         │
                         ▼
                 Residual Connection
                         │
                         ▼
                    ┌─────────┐
                    │ RMSNorm │
                    └────┬────┘
                         │
                         ▼
                  Language Model
                       Head
                         │
                         ▼
                  Next Token Logits
                         │
                         ▼
                Autoregressive Sampling
                         │
                         ▼
                    Generated Text
```

---

# ⚙️ Model Configuration

| Component              |            Configuration |
| :--------------------- | -----------------------: |
| 🧠 Parameters          |                **2.76M** |
| 🧱 Transformer Layers  |                    **4** |
| 📐 Embedding Dimension |                  **256** |
| 👁️ Query Heads        |                    **8** |
| 🔑 KV Heads            |                    **2** |
| 📚 Context Length      |                  **256** |
| 🔤 Tokenization        |          Character-level |
| 🎯 Objective           | Causal Language Modeling |
| 🖥️ Framework          |                  PyTorch |

### Why GQA?

The model uses **8 query heads but only 2 key/value heads**, reducing KV-cache memory requirements while retaining multiple query representations.

```text
Query Heads:       8
                   │
                   ├── Q ──┐
                   ├── Q ──┤
                   ├── Q ──┤
                   ├── Q ──┤
                   ├── Q ──┤
                   ├── Q ──┤
                   ├── Q ──┤
                   └── Q ──┘
                          │
KV Heads:             2 K/V
                          │
                          ▼
                    Attention Output
```

---

# 🧩 Core Components

### 🔤 Character-Level Tokenization

The model starts with a simple character vocabulary.

```text
"ROMEO:"
   ↓
['R', 'O', 'M', 'E', 'O', ':']
   ↓
[17, 14, 12, 4, 14, 26]
```

This keeps tokenization transparent and makes the entire language-model pipeline easy to inspect.

---

### ⚡ Grouped Query Attention

Instead of assigning a unique Key/Value head to every Query head:

```text
8 Query Heads
      +
2 KV Heads
      ↓
Grouped Query Attention
```

This introduces a more memory-efficient attention structure inspired by modern LLM architectures.

---

### 🔄 RoPE

**Rotary Positional Embeddings** inject positional information directly into the attention mechanism by rotating query and key representations.

This allows the model to understand:

```text
Token A → Position 1
Token B → Position 2
Token C → Position 3
...
```

without relying on traditional learned positional embeddings.

---

### 📐 RMSNorm

The architecture uses **RMSNorm** instead of LayerNorm to normalize hidden representations before the major sublayers.

```text
Input
  │
  ▼
RMSNorm
  │
  ├── Attention
  │
  ▼
Residual
  │
  ▼
RMSNorm
  │
  ├── SwiGLU
  │
  ▼
Residual
```

---

### 🧩 SwiGLU

The feed-forward block uses **SwiGLU**, a gated activation mechanism widely used in modern Transformer architectures.

Conceptually:

```text
Hidden State
     │
     ├──────────────┐
     ▼              ▼
 Linear          Linear
     │              │
     ▼              ▼
   SiLU          Gate
     │              │
     └──────┬───────┘
            ▼
        Element-wise
        Multiplication
            │
            ▼
        Projection
```

---

### 🔗 Weight Tying

The input token embedding matrix and language-model output projection share weights.

```text
Token Embedding
       │
       │
       └──────────────┐
                      │
                      ▼
                Shared Weights
                      │
                      ▼
                LM Head Output
```

This reduces the total number of trainable parameters.

---

# 📊 Training Results

The model was trained on the **Shakespeare dataset**.

| Metric             |     Result |
| :----------------- | ---------: |
| 📉 Training Loss   | **1.0163** |
| 📈 Validation Loss | **1.5086** |

```text
Training Loss      1.0163
Validation Loss    1.5086
```

The resulting model demonstrates that even a compact Transformer can learn recognizable patterns in Shakespearean text.

---

# ✍️ Sample Generation

### Prompt

```text
ROMEO:
```

### Generated Output

```text
ROMEO:

That she is come to me at once so strength
To shake the contrary of the most dead.
```

The model generates text **autoregressively**, predicting one token at a time and feeding its prediction back into the model.

```text
Token₁
  ↓
Token₂
  ↓
Token₃
  ↓
Token₄
  ↓
   ...
  ↓
Generated Sequence
```

---

# 🧪 Training Pipeline

```text
Shakespeare Dataset
        │
        ▼
Character Vocabulary
        │
        ▼
Token Encoding
        │
        ▼
Train / Validation Split
        │
        ▼
Context Window Sampling
        │
        ▼
Mini-Batch Training
        │
        ▼
Transformer Forward Pass
        │
        ▼
Cross-Entropy Loss
        │
        ▼
Backpropagation
        │
        ▼
AdamW Optimization
        │
        ▼
Validation
        │
        ▼
Autoregressive Generation
```

---

# 🚀 Tech Stack

| Technology              | Purpose                       |
| :---------------------- | :---------------------------- |
| 🐍 **Python**           | Core implementation           |
| 🔥 **PyTorch**          | Neural network & training     |
| 🎮 **CUDA**             | GPU acceleration              |
| ☁️ **Google Colab**     | Training environment          |
| 📓 **Jupyter Notebook** | Experimentation & development |

---

# 📁 Repository Structure

```text
mini-llm-from-scratch/
│
├── 📓 notebook.ipynb
├── 📖 README.md
└──
```

> The current implementation is intentionally kept inside a notebook to make the architecture and training process easy to follow.

---

# 🛣️ Roadmap

### Tokenization

* [ ] Implement Byte Pair Encoding (BPE)
* [ ] Compare character-level vs subword tokenization

### Architecture

* [ ] Increase model depth
* [ ] Increase hidden dimension
* [ ] Experiment with different attention configurations
* [ ] Experiment with context lengths

### Training

* [ ] Train on larger datasets
* [ ] Experiment with learning-rate schedules
* [ ] Add checkpointing
* [ ] Add experiment tracking

### Engineering

* [ ] Convert notebook into a modular Python codebase
* [ ] Add configuration files
* [ ] Add automated evaluation
* [ ] Add unit tests
* [ ] Build a simple inference interface

### Research

* [ ] Fine-tuning experiments
* [ ] Compare RoPE configurations
* [ ] Compare GQA against MHA
* [ ] Analyze scaling behavior

---

# 🎯 Why I Built This

The goal wasn't simply to train a text generator.

The goal was to **understand the mechanics behind modern language models by implementing the important components myself**.

From:

```text
Tokenization
     ↓
Embeddings
     ↓
Attention
     ↓
RoPE
     ↓
Normalization
     ↓
SwiGLU
     ↓
Residual Connections
     ↓
Language Modeling
     ↓
Generation
```

every stage provides a concrete understanding of how a Transformer-based language model processes and generates text.

---

# 👨‍💻 Author

### Sumit Jha

**Computer Science & Engineering • AI/ML Enthusiast • Machine Learning Engineer in the Making**

Building systems, studying machine learning, and exploring how modern AI models work — **one layer at a time.**

---

<p align="center">

### ⭐ If you find this project interesting, consider giving it a star!!

**Built from scratch. Trained from scratch. Understood from the inside out.**

</p>
