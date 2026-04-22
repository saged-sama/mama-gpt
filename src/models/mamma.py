import torch.nn as nn
from torch.nn import RMSNorm, functional as F
from lib.attention import MultiHeadAttentionWithRope
import torch

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x))) * self.w3(x)

class Transformer(nn.Module):
    def __init__(self, dim, context_length, num_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        self.attn = MultiHeadAttentionWithRope(dim, num_heads=num_heads, context_length=context_length)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x, use_cache=False):
        x = self.attn_norm(x)
        attn_output = self.attn(x, use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ffn_norm(x))
        return x

class Mamma(nn.Module):
    def __init__(self, vocab_size, dim, context_length, num_layers, num_heads, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            Transformer(dim, context_length, num_heads, hidden_dim)
                for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, use_cache=False):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, use_cache)

        x = self.norm(x)
        return self.output(x)
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        
        for layer in self.layers:
            layer.attn.reset_cache()
            
        logits = self.forward(x, use_cache=True)
        
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits_last, top_k)
                logits_last[logits_last < v[:, [-1]]] = -float("inf")
                
            probs = torch.softmax(logits_last, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat([x, next_token], dim=1)
            
            logits = self.forward(next_token, use_cache=True)
            
        return x