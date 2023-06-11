"""A simple transformer model for MNIST classification just to get started. Partial implementation
with PyTorch's nn.TransformerEncoder."""
import math

import torch
import torch.nn as nn

# The two main things in this script are to define the positional encoding and the simple transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class SimpleTransformer(nn.Module):
    def __init__(self):
        # text analogy of vocabulary size and embedding size is 256 and 4 respectively
        # for this image task
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(256, 4) # arbitrary numbers chosen to embed images

        self.pos_encoder = PositionalEncoding(4, 0.5)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=4, 
                                                        nhead=2, 
                                                        dim_feedforward=100,
                                                        batch_first=True,)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(28*28*4, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(-1, 784, 4)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(-1, 28*28*4)
        x = self.fc1(x)
        return x
    

    






