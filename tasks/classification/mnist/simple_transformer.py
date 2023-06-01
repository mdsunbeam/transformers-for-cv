"""A simple transformer model for MNIST classification."""

import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self):
        # text analogy of vocabulary size and embedding size is 256 and 4 respectively
        # for this image task
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(256, 4) # arbitrary numbers chosen to embed images

        self.positional_encoding = PositionalEncoding(4, 0.5)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=4, 
                                                        nhead=2, 
                                                        dim_feedforward=100,
                                                        batch_first=True,)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(28*28*4, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(-1, 784, 4)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.reshape(-1, 28*28*4)
        x = self.fc1(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        





