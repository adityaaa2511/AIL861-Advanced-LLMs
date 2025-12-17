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
    def __init__(self,d_model,num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

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
        return out, attn