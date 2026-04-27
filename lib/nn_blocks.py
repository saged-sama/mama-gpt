import torch
import torch.nn as nn
import math

def softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - x_max
    exps = torch.exp(x)
    denom = torch.sum(exps, dim=dim, keepdim=True)
    return exps / denom

def xavier_normal_(weight):
    out_dim, in_dim = weight.shape
    std = (2.0/(out_dim + in_dim)) ** 0.5
    
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size=(out_dim, in_dim)))
        xavier_normal_(weight=self.weight)
        
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None
    
    def forward(self, x):
        out = x @ self.weight.T 
        if self.use_bias:
            out = out + self.bias
        return out