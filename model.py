from http.client import _DataType
from networkx import directed_configuration_model
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size=int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding *math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        self.seq_len=seq_len
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)

        # create a matrix of shape (seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)

        # create a model of shape
        position=torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
