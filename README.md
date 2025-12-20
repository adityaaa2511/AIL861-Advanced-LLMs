# AIL861-Advanced-LLMs
### Mini Transformer Language Model — Decoder-Only GPT-Style Model (TinyStories)

This repository contains a complete **decoder-only Transformer language model implemented from scratch in PyTorch**, as part of the **AIL861: Advanced LLMs (IIT Delhi, Fall ’25)** coursework.  
The project demonstrates how to build, train, and sample from a GPT-style autoregressive language model on a small dataset.

---

## 🚀 Key Technical Features

### ✔️ **Custom Tokenization & Vocabulary**
- Regex-based word tokenizer using `re.findall`
- Vocabulary of **~30K tokens** built from TinyStories
- Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`
- Manual `encode()` converts text to token IDs with length control

---

### 📚 **Dataset: TinyStories (HuggingFace)**
- Loaded via `load_dataset("roneneldan/TinyStories")`
- Teacher forcing used during training:
  - Input = sequence[:-1]
  - Target = sequence[1:]
- Fixed context length (e.g., **64 tokens**)

---

## 🧩 Transformer Architecture

### 🏗️ **Decoder-Only Transformer (GPT-style)**
- **3 Transformer decoder blocks**
- **300-dimensional embeddings**
- **6 attention heads**
- **FFN dimension = 4× model dimension (1200)**
- **LayerNorm + Residual connections everywhere**

### 💡 **Causal Masking**
- Ensures autoregressive constraint:
  > token i can only attend to positions ≤ i

---

### 🧠 Multi-Head Self Attention (MHA)
- Manual implementation of:
  - Q, K, V projections
  - head splitting (B, H, L, d)
  - `scaled_dot_product_attention`
- Softmax over attention scores
- Output projection + dropout

---

### 🔥 Feed-Forward Network (FFN)
- Position-wise MLP per token: Linear → GELU → Dropout → Linear
- Provides **non-linearity + feature expansion**
- Essential for model expressivity

---

### 🏎️ KV-Cache for Fast Generation
- Stores past keys/values across decoding steps
- Concatenates only new tokens
- Avoids quadratic re-computation
- Enables efficient auto-regressive loops

---

## 🪄 Positional Embeddings
- **Sinusoidal positional embeddings** from Vaswani et al.
- Supports `start_pos` offset for cached decoding

---

## 📝 Training Setup

### ⚙️ Hyperparameters
- Batch size: **16**
- LR: **3e-4**
- Epochs: **3–10**
- Optimizer: **Adam**

### 🚀 Mixed Precision + Gradient Accumulation
- Uses `torch.amp.autocast("cuda")`
- Uses `torch.amp.GradScaler` to avoid underflow
- Accumulation allows **simulated larger batch sizes**

---

### 📉 Metrics
- **Cross-entropy loss**
- **Perplexity = exp(loss)**
- Logged for both train/validation
- Saved plots:
  - `loss.png`
  - `perplexity.png`

---

## 🗣️ Text Generation

### 🎲 Sampling Options
- Temperature scaling
- Top-k filtering
- Multinomial sampling

### 🔭 Beam Search
- Configurable beam size
- Returns sequence with highest log-probability

### 🧑‍💻 Output
Prints:
- Prompt ID sequence → converted back to tokens
- Final generated text

---

## 🛠️ Saving / Loading the Model
- Full model saved via:
```python
torch.save(model.state_dict(), "decoder_tinystories.pt")

### Sample Prompt + Generation
- Prompt: <sos> spot spot saw the
- Generated: <sos> spot spot saw the sun rise in the sky he wanted to go outside and play he ran to his mom and said mom can i go outside and play his mom smiled and said yes spot you can go outside but be careful don t go too far and don t get too close to the sun spot ran outside and saw <eos>
