import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q,k,v,mask=None):
    B,H,L,D_k = q.shape
    scores = torch.matmul(q.k.tranpose(-2,-1)) / np.sqrt(D_k)
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
        return x.tranpose(1,2)
    
    def _merge_heads(self,x):
        B,H,L,D_k = x.shape
        x = x.transpose(1,2).contiguous().view(B,L,H*D_k)
        return x
    
    def forward(self, x, mask=None):
        B,L,_ = x.shape
        qkv_proj = self.qkv(x)

        q,k,v = torch.chunk(qkv_proj,3,dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_out, attn = scaled_dot_product_attention(q,k,v,mask)
        attn_out = self._merge_heads(attn_out)
        out = self.o_proj(attn_out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
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

    def forward(self,x,mask=None):
        x = x + self.mha(self.ln1(x),mask)
        x = x + self.ffn(self.ln2(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,d_ff,num_heads=8,num_layers=3,dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model,d_ff,num_heads,dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model,vocab_size, bias=False)

    def forward(self,x,mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x,mask)
        x = self.ln(x)
        logits = self.output_proj(x)
        return F.softmax(logits,dim=-1)
    
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
