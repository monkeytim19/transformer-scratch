import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    
    wavelength: factor to determine the wavelength in the sinusoidal function.
    """
    def __init__(self, wavelength=10000.):
        super().__init__()
        self.wavelength = wavelength

    def forward(self, x):
        """Given a (... x seq_len x embedding_dim) tensor, returns a (seq_len x embedding_dim) tensor."""
        seq_len, embedding_dim = x.shape[-2], x.shape[-1]
        pe = torch.zeros((seq_len, embedding_dim))
        position = torch.arange(seq_len).unsqueeze(1)
        factor = torch.exp(-math.log(self.wavelength) * torch.arange(0, embedding_dim, 2) / embedding_dim)
        pe[:, 0::2] = torch.sin(position * factor)
        pe[:, 1::2] = torch.cos(position * factor)
        return pe

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weights = nn.Parameter(torch.zeros((vocab_size, hidden_dim)))
        nn.init.uniform_(self.weights)

    def forward(self, x):
        return np.sqrt(self.hidden_dim) * self.weights[x]
    
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_kq, d_v):
        super().__init__()
        self.d_kq = d_kq
        self.q_weights = nn.Parameter(torch.rand(d_in, d_kq))
        self.k_weights = nn.Parameter(torch.rand(d_in, d_kq))
        self.v_weights = nn.Parameter(torch.rand(d_in, d_v))

    def forward(self, x, attention_mask=None):
        query = x @ self.q_weights
        key = x @ self.k_weights
        value = x @ self.v_weights

        if attention_mask is None:
            attention_mask = torch.zeros((x.shape[-2], x.shape[-2])).to(x.device)

        attn_scores = query @ torch.transpose(key, -1, -2)
        masked_attn_scores = attn_scores.masked_fill(attention_mask.to(x.device).bool(), -torch.inf)
        attn_weights = torch.softmax(masked_attn_scores / self.d_kq**0.5, dim=-1)
        return attn_weights @ value
    
class CrossAttention(nn.Module):
    def __init__(self, d_in, d_kq, d_v):
        """
        d_in: Dimension of the input
        d_kq: Dimension of the key-query
        d_v: Dimension of the value
        """
        super().__init__()
        self.d_kq = d_kq
        self.q_weights = nn.Parameter(torch.rand(d_in, d_kq))
        self.k_weights = nn.Parameter(torch.rand(d_in, d_kq))
        self.v_weights = nn.Parameter(torch.rand(d_in, d_v))

    def forward(self, x, encoder_x, attention_mask=None):
        """
        
        """
        query = x @ self.q_weights
        key = encoder_x @ self.k_weights
        value = encoder_x @ self.v_weights

        if attention_mask is None:
            attention_mask = torch.zeros((x.shape[-2], encoder_x.shape[-2]))

        attn_scores = query @ torch.transpose(key, -1, -2)
        masked_attn_scores = attn_scores.masked_fill(attention_mask.to(x.device).bool(), -torch.inf)
        attn_weights = torch.softmax(masked_attn_scores / self.d_kq**0.5, dim=-1)
        return attn_weights @ value
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_kq, d_v, n_heads, attn_type):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "self":
            self.heads = nn.ModuleList([SelfAttention(d_in, d_kq, d_v) for _ in range(n_heads)])
        elif attn_type == "cross":
            self.heads = nn.ModuleList([CrossAttention(d_in, d_kq, d_v) for _ in range(n_heads)])
        else:
            raise ValueError("attn_type should be either 'self' or 'cross'.")

    def forward(self, x, encoder_x=None, attention_mask=None):
        if self.attn_type == "self":
            return torch.cat([head(x, attention_mask) for head in self.heads], dim=-1)
        else:
            assert encoder_x is not None
            return torch.cat([head(x, encoder_x, attention_mask) for head in self.heads], dim=-1)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.input_dimension = d_in
        self.output_dimension = d_out
        self.hidden_dimension = d_hidden
        self.linear_1 = nn.Linear(self.input_dimension,self.hidden_dimension)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden_dimension,self.output_dimension)

    def forward(self, x):
        input = x.detach().clone()
        layer_1 = self.relu(self.linear_1(input))
        layer_2 = self.relu(self.linear_2(layer_1))
        return layer_2
    
class DecoderBlock(nn.Module):
    def __init__(self, d_in, d_kq, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_in=d_in, d_kq=d_kq, d_v=d_in//n_heads, n_heads=n_heads, attn_type="self")
        self.fc = FeedForwardNetwork(d_in=d_in, d_out=d_in, d_hidden=d_in*2)
        self.layernorm1 = nn.LayerNorm(d_in)
        self.layernorm2 = nn.LayerNorm(d_in)

    def forward(self, x):
        attention_mask = torch.triu(torch.ones(x.shape[-2], x.shape[-2]), diagonal=1)
        x = self.attn(x, attention_mask=attention_mask) + x
        x = self.layernorm1(x) 
        x = self.fc(x) + x
        x = self.layernorm2(x)
        return x

   
    
if __name__ == "__main__":
    pass