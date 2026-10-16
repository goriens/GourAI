<p align="center">
  <h1 align="center">🍛 GourAI</h1>
  <p align="center">
    A GPT-style Transformer Language Model — built from scratch in PyTorch.
    <br />
    <em>Learning by building, one layer at a time.</em>
  </p>
</p>

---

## 🧠 What is GourAI?

GourAI is a **decoder-only Transformer language model** inspired by GPT-2 and [nanoGPT](https://github.com/karpathy/nanoGPT). It's built from the ground up in PyTorch to deeply understand how modern LLMs work — from tokenization to attention to text generation.

**Current Version: `V1` (Character-level, Tiny Shakespeare)**

## 🏗️ Architecture

```
GourAILM (~10.8M parameters)
├── Token Embedding + Positional Embedding
├── 6× Transformer Block
│   ├── Pre-LayerNorm → Causal Multi-Head Self-Attention (6 heads)
│   └── Pre-LayerNorm → Feed-Forward MLP (4x expansion, GELU)
├── Final LayerNorm
└── LM Head (weight-tied with Token Embedding)
```

| Hyperparameter                | Value                |
| ----------------------------- | -------------------- |
| Context Window (`block_size`) | 256                  |
| Layers (`n_layer`)            | 6                    |
| Attention Heads (`n_head`)    | 6                    |
| Embedding Dim (`n_embd`)      | 384                  |
| Vocab Size                    | 65 (character-level) |
| Total Parameters              | ~10.8M               |

### Key Design Choices

- **Pre-LayerNorm** formulation (more stable training than Post-LN)
- **Weight Tying** between token embeddings and LM head ([Press & Wolf, 2016](https://arxiv.org/abs/1608.05859))
- **Scaled Residual Init** — `c_proj` weights scaled by `1/√(2N)` per GPT-2
- **FlashAttention** support (PyTorch 2.0+ `scaled_dot_product_attention`) with manual fallback
- **No bias** in Linear/LayerNorm layers (cleaner, faster)

## 📁 Project Structure

```
GourAI/
├── src/
│   ├── config.py              # Central config (dataclass)
│   ├── device.py              # CUDA / MPS / CPU auto-detection
│   ├── model/
│   │   ├── embeddings.py      # Token + Positional embeddings
│   │   ├── attention.py       # Causal Multi-Head Self-Attention
│   │   ├── mlp.py             # Feed-Forward Network (GELU)
│   │   ├── block.py           # Transformer Block (Pre-LN + Residual)
│   │   └── transformer.py     # Full GourAILM model assembly
│   ├── tokenizer/
│   │   └── char_tokenizer.py  # Character-level tokenizer
│   ├── data/
│   │   └── prepare.py         # Data download, vocab build, train/val split
│   ├── train/                 # (WIP) Training loop
│   └── api/                   # (WIP) Inference API
├── data/                      # Raw text + serialized binary data
├── checkpoints/               # Vocab metadata + saved model weights
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/GourAI.git
cd GourAI
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python -m src.data.prepare
```

Downloads Tiny Shakespeare (~1MB), builds character vocabulary, and creates `train.bin` / `val.bin`.

### 3. Verify Architecture

Each module has a built-in self-verification test:

```bash
python -m src.model.embeddings
python -m src.model.attention
python -m src.model.mlp
python -m src.model.block
python -m src.model.transformer
```

## 📌 V1 Roadmap

- [x] **Phase 1** — Project Setup & Config System
- [x] **Phase 2** — Data Pipeline (Download → Tokenize → Serialize)
- [x] **Phase 3** — Character Tokenizer + Vocabulary
- [x] **Phase 4** — Core Transformer Architecture (Embeddings → Attention → MLP → Block → Model)
- [x] **Phase 5** — Training Loop (AdamW, Cosine LR, Gradient Clipping)
- [x] **Phase 6** — Text Generation (Sampling, Temperature, Top-k)
- [x] **Phase 7** — FastAPI Production Server (REST + SSE Real-Time Streaming)
- [x] **Phase 8** — Modern Web Playground UI & Cloud Deployment

## 🛠️ Tech Stack

- **PyTorch** ≥ 2.0 (FlashAttention support)
- **NumPy** (data serialization)
- **FastAPI + Uvicorn** (upcoming API)
- Python 3.10+

## 📄 License

This project is for educational and personal research purposes.

---
