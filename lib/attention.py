import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
import torch
# from lib.nn_blocks import softmax, Linear
from torch.nn import Linear
from torch.nn.functional import scaled_dot_product_attention

# def scaled_dot_product_attention(q, k, v, is_causal=False):
#     dk = q.shape[-1]
    
#     if dk != k.shape[-1]:
#         raise ValueError("Dimension mismatch between q and k")
    
#     scores = q @ k.transpose(-2, -1)
    
#     scores = scores / math.sqrt(dk)
    
#     if is_causal:
#         L = scores.size(-1)
#         mask = torch.triu(
#             torch.ones(L, L, device=scores.device, dtype=torch.bool),
#             diagonal=1
#         )
#         scores = scores.masked_fill(mask, float('-inf'))
        
#     attn = softmax(scores, dim=-1)
#     return attn @ v

# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim, num_heads, context_length):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.context_length = context_length
        
#         assert dim % num_heads == 0
        
#         self.q = Linear(dim, dim, bias=False)
#         self.k = Linear(dim, dim, bias=False)
#         self.v = Linear(dim, dim, bias=False)
#         self.out_proj = Linear(dim, dim, bias=False)
        
#     def forward(self, x, is_causal=False):
#         B, T, D = x.shape
        
#         Q = self.q(x)
#         K = self.k(x)
#         V = self.v(x)

#         Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
#         K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

#         out = scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

#         out = out.transpose(1, 2).contiguous().reshape(B, T, D)

#         return self.out_proj(out)        


class MultiHeadAttentionWithRope(nn.Module):
    def __init__(self, dim, num_heads, context_length):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.context_length = context_length
        
        assert dim % num_heads == 0
        
        self.qkv = Linear(dim, 3*dim, bias=False)
        self.out_proj = Linear(dim, dim, bias=False)
        
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=context_length
        )
        
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.cache_pos = 0

    def setup_cache(self, batch_size, device, dtype):
        caches_shape = (batch_size, self.context_length, self.num_heads, self.head_dim)
        self.k_cache = torch.zeros(caches_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(caches_shape, device=device, dtype=dtype)
        self.cache_pos = 0
        
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
        
    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        
        # Context length guard
        if self.cache_pos + T > self.context_length:
            raise ValueError(f"Context length exceeded: cache_pos={self.cache_pos}, T={T}, context_length={self.context_length}")
        
        qkv = self.qkv(x)
        
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Clamp position indices for extra safety
        pos_ids = torch.arange(self.cache_pos, self.cache_pos + T, device=x.device)
        pos_ids = torch.clamp(pos_ids, max=self.context_length - 1)
        q = self.rope(q, input_pos=pos_ids)
        k = self.rope(k, input_pos=pos_ids)
        
        if use_cache:
            if self.k_cache is None or self.k_cache.shape[0] != B:
                self.setup_cache(B, x.device, x.dtype)
            
            # Cache safety: truncate if needed
            end = min(self.cache_pos + T, self.context_length)
            k_to_store = k[:, :end - self.cache_pos]
            v_to_store = v[:, :end - self.cache_pos]
            
            self.k_cache[:, self.cache_pos : end] = k_to_store
            self.v_cache[:, self.cache_pos : end] = v_to_store
            
            k_session = self.k_cache[:, : end]
            v_session = self.v_cache[:, : end]
            self.cache_pos = end
        else:
            k_session, v_session = k, v
        
        attn = scaled_dot_product_attention(
            q.transpose(1, 2),
            k_session.transpose(1, 2),
            v_session.transpose(1, 2),
            is_causal=(T > 1)
        )
        
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(attn)