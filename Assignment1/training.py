import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re, math, time, random
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

SPECIAL = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def build_vocab(dataset, vocab_size=30000):
    counter = Counter()
    for ex in dataset:
        counter.update(tokenize(ex["text"]))
    vocab = dict(SPECIAL)
    for i, (w, _) in enumerate(counter.most_common(vocab_size - len(SPECIAL))):
        vocab[w] = i + len(SPECIAL)
    id2word = {i: w for w, i in vocab.items()}
    return vocab, id2word

def encode(text, vocab, max_len):
    ids = [vocab["<sos>"]]
    for w in tokenize(text):
        ids.append(vocab.get(w, vocab["<unk>"]))
        if len(ids) >= max_len - 1:
            break
    ids.append(vocab["<eos>"])
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

# ============================================================
# 2) Load FastText embeddings
# ============================================================
def load_fasttext(vec_path, vocab, dim):
    emb = np.random.normal(0, 0.02, (len(vocab), dim)).astype(np.float32)
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            if w in vocab and len(parts) == dim + 1:
                emb[vocab[w]] = np.array(parts[1:], dtype=np.float32)
    return torch.tensor(emb)

# ============================================================
# 3) Dataset (Teacher Forcing)
# ============================================================
class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, split, vocab, ctx):
        self.data = split
        self.vocab = vocab
        self.ctx = ctx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = encode(self.data[i]["text"], self.vocab, self.ctx + 1)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

def scaled_dot_product_attention(q,k,v,mask=None):
    B,H,L,D_k = q.shape
    scores = torch.matmul(q,k.transpose(-2,-1)) / np.sqrt(D_k)
    if mask is not None:
        scores = scores + mask

    attn = F.softmax(scores,dim=-1)
    output = torch.matmul(attn,v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads=8,dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self,x):
        B,L,_ = x.shape
        x = x.view(B,L,self.num_heads,self.d_k)
        return x.transpose(1,2)
    
    def _merge_heads(self,x):
        B,H,L,D_k = x.shape
        x = x.transpose(1,2).contiguous().view(B,L,H*D_k)
        return x
    
    def forward(self, x, kv_cache=None, mask=None):
        B,L,_ = x.shape
        qkv_proj = self.qkv(x)

        q,k,v = torch.chunk(qkv_proj,3,dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0],k],dim=2)
            v = torch.cat([kv_cache[1],v],dim=2)

        new_cache = {"k": k.detach(), "v": v.detach()}

        attn_out, attn = scaled_dot_product_attention(q,k,v,mask)
        attn_out = self._merge_heads(attn_out)
        out = self.o_proj(attn_out)
        out = self.dropout(out)
        return out , new_cache
    

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,d_model,d_ff,num_heads,dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads,dropout)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self,x,kv_cache=None,mask=None):
        a, new_cache = self.mha(self.ln1(x),kv_cache,mask)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, new_cache
    
class Decoder(nn.Module):
    def __init__(self,emb_init,vocab_size,d_model,d_ff,num_heads=8,num_layers=3,dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(emb_init, freeze=False)
        self.pe = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model,d_ff,num_heads,dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model,vocab_size, bias=False)

    def _casual_mask(self,L,device):
        mask = torch.triu(torch.ones(L,L,device=device),diagonal=1)
        mask = mask.masked_fill(mask==1,float('-inf'))
        return mask

    def forward(self,x,kv_cache=None,mask=None):
        B,L = x.shape
        if mask is None:
            mask = self._casual_mask(L,DEVICE)
        x = self.pe(self.embedding(x))
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, kv_cache[i] if kv_cache is not None else None, mask)
            new_caches.append(new_cache)
        x = self.ln(x)
        logits = self.output_proj(x)
        return logits, new_caches
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * -(np.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe',pe)

    def forward(self,x):
        B,L, _ = x.shape
        return x + self.pe[:L].unsqueeze(0)
    
def lm_loss(logits,labels):
    return F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=SPECIAL["<pad>"])

def train_with_grad_accum(model,dataloader,optimizer,accum_steps,device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, _ = model(input_ids)
        loss = lm_loss(logits,labels)
        loss.backward()
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * input_ids.size(0) * accum_steps
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model,dataloader,device):
    model.eval()
    total_loss = 0
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, _ = model(input_ids)
        loss = lm_loss(logits,labels)
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset) 

def perplexity(loss):
    return math.exp(loss)

# ============================================================
# 11) Generation (Stochastic)
# ============================================================
@torch.no_grad()
def generate(model, prompt_ids, max_new=50, temp=1.0, topk=20):
    model.eval()
    x = torch.tensor(prompt_ids, device=DEVICE).unsqueeze(0)
    cache = [None] * len(model.blocks)
    for _ in range(max_new):
        logits, cache = model(x, cache)
        logits = logits[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        if topk is not None:
            v, ix = torch.topk(probs, topk)
            probs2 = torch.zeros_like(probs).scatter_(1, ix, v)
            probs = probs2 / probs2.sum()
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
        if nxt.item() == SPECIAL["<eos>"]:
            break
    return x.squeeze(0).tolist()

def main():
    
    CTX = 64
    LAYERS = 3
    HEADS = 6
    D_MODEL = 300 
    D_FF = 4 * D_MODEL
    LR = 3e-4
    BATCH = 16
    ACCUM = 1

    # ---- Load data ----
    ds = load_dataset("roneneldan/TinyStories")
    vocab, id2word = build_vocab(ds["train"])

    # ---- FastText ----
    FT_PATH = "cc.en.300.vec"
    emb = load_fasttext(FT_PATH, vocab, D_MODEL)

    # ---- DataLoaders ----
    train_ds = TinyStoriesDataset(ds["train"], vocab, CTX)
    val_ds   = TinyStoriesDataset(ds["validation"], vocab, CTX)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH)

    model = Decoder(emb,len(vocab),D_MODEL,D_FF,HEADS,LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tr_losses, va_losses = [], []
    tr_ppls, va_ppls = [], []
    for ep in tqdm(range(10)):
        tl = train_with_grad_accum(model, train_dl, opt, ACCUM, DEVICE)
        vl = evaluate(model, val_dl, DEVICE)
        tr_losses.append(tl)
        va_losses.append(vl)
        tr_ppls.append(perplexity(tl))
        va_ppls.append(perplexity(vl))
        print(f"Epoch {ep}: train {tl:.4f} |  tr_ppl {tr_ppls[-1]:.2f} | val {vl:.4f} |  val_ppl {va_ppls[-1]:.2f}")

    # ---- Plots: loss + perplexity ----
    plt.figure()
    plt.plot(tr_losses, label="train loss")
    plt.plot(va_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)

    plt.figure()
    plt.plot(tr_ppls, label="train perplexity")
    plt.plot(va_ppls, label="val perplexity")
    plt.xlabel("epoch")
    plt.ylabel("perplexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("perplexity.png", dpi=150)

    # ---- Sample generation ----
    ex = val_ds[0][0][:5].tolist()
    out = generate(model, ex)
    print("Generated:", " ".join(id2word[i] for i in out if i in id2word))

if __name__ == "__main__":
    main()