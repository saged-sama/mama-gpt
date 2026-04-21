import torch.nn as nn
from torch.nn import RMSNorm, MultiheadAttention, functional as F

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x))) * self.w3(x)

class Transformer(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        self.attn = MultiheadAttention(dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x):
        x = self.attn_norm(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = x + self.mlp(self.ffn_norm(x))
        return x

class Mamma(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            Transformer(dim, num_heads, hidden_dim)
                for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output(x)