from http.client import _DataType
from typing import ValuesView
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
        super().__init__()
        self.seq_len=seq_len
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)

        # create a matrix of shape (seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)

        # create a model of shape
        position=torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        # Apply sin to even indices
        pe[:,0::2]=torch.sin(position*div_term)
        # Apply cos to odd indices
        pe[:,1::2]=torch.cos(position*div_term)

        # Add a batch dimension to the positional encoading
        pe=pe.unsqueeze(0)

        # Register the positional Encoading as a buffer
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self,eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.gama=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)

        return self.gama*(x-mean)/(std+self.eps) +self.bias    
    

class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int,dff:int,dropout:float):
        super().__init__()
        self.linear1=nn.Linear(d_model,dff) # w1 and b1
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(dff,d_model)  #w2 and b2


    def forward(self,x):
        
         return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model % h ==0 ,'d_model is not divisible by h'         


        self.dk=d_model//h
        self.w_q=nn.Linear(d_model,d_model)  #wq
        self.w_k=nn.Linear(d_model,d_model)  #wk
        self.w_v=nn.Linear(d_model,d_model)  #wv

        self.w_o=nn.Linear(d_model,d_model)  #wo
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):

        d_k=query.shape[-1]

        attention_score=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score=attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score=dropout(attention_score)   

        return (attention_score @ value) ,attention_score     



    def forward(self,Q,K,V,mask):

        query=self.w_q(Q)  #(Batch,seq_len,d_model)
        key=self.w_k(K)    #(Batch,seq_len,d_model)
        value=self.w_v(V)  #(Batch,seq_len,d_model)


        query=query.view(query.shape[0],query.shape[1],self.h,self.dk).transpose(1,2)

        key=key.view(key.shape[0],key.shape[1],self.h,self.dk).transpose(1,2)

        value=value.view(value.shape[0],value.shape[1],self.h,self.dk).transpose(1,2)

        x,self_attention=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (Batch,h,seq_len,d_k)---->(Batch,seq_len,h,d_k)-------->(Batch,seq_len,d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model % h ==0 ,'d_model is not divisible by h'         


        self.dk=d_model//h
        self.w_q=nn.Linear(d_model,d_model)  #wq
        self.w_k=nn.Linear(d_model,d_model)  #wk
        self.w_v=nn.Linear(d_model,d_model)  #wv

        self.w_o=nn.Linear(d_model,d_model)  #wo
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):

        d_k=query.shape[-1]

        attention_score=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score=attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score=dropout(attention_score)   

        return (attention_score @ value) ,attention_score     



    def forward(self,Q,K,V,mask):

        query=self.w_q(Q)  #(Batch,seq_len,d_model)
        key=self.w_k(K)    #(Batch,seq_len,d_model)
        value=self.w_v(V)  #(Batch,seq_len,d_model)


        query=query.view(query.shape[0],query.shape[1],self.h,self.dk).transpose(1,2)

        key=key.view(key.shape[0],key.shape[1],self.h,self.dk).transpose(1,2)

        value=value.view(value.shape[0],value.shape[1],self.h,self.dk).transpose(1,2)

        x,self_attention=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (Batch,h,seq_len,d_k)---->(Batch,seq_len,h,d_k)-------->(Batch,seq_len,d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)








             




