import math
import torch
import torch.nn as nn
from torch import Tensor

# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-math.log(10000.0) / latent_dim))
        pe = torch.zeros(max_len, 1, latent_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sequence_len, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        # Permute back
        return self.dropout(x).permute(1, 0, 2)