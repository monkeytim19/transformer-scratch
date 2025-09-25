import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self,query_shape,key_shape,value_shape, model_size=512):
        super().__init__()
        self.d_q= query_shape[-1]
        self.d_k = key_shape[-1]
        self.d_v = value_shape[-1]
        self.model_size = model_size
        self.W_q= nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty((self.model_size, self.d_q))))
        self.W_k= nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty((self.model_size, self.d_k))))
        self.W_v= nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty((self.model_size, self.d_v))))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attention_mask=None):
        query_t = torch.matmul(query, self.W_q)
        key_t = torch.matmul(key, self.W_k)
        value_t = torch.matmul(value, self.W_v)
        
        query_key=torch.matmul(query_t, key_t.transpose(-2,-1))/math.sqrt(self.d_k)
        if attention_mask is not None:
            query_key = query_key.masked_fill(attention_mask.bool(), -torch.inf)
       
        attention = torch.matmul(self.softmax(query_key), value_t)
        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, query_shape,key_shape,value_shape, head_count, model_size=512):
        super(MultiHeadAttention,self).__init__()
        self.head_count = head_count
        self.model_size = model_size
        self.query_shape = query_shape
        self.key_shape = key_shape
        
        self.value_shape = value_shape
        self.W_O = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.head_count*self.value_shape[-1],self.model_size)))

        self.heads = [ Attention(self.query_shape, self.key_shape, self.value_shape, self.model_size) for _ in range(self.head_count)]

    def forward(self, query, key, value):
        mh_p1=torch.cat([head(query, key, value) for head in self.heads],-1)
        mh_p2 = torch.matmul(mh_p1, self.W_O)
        return mh_p2
    

class FeedForwardNetwork(nn.Module):
    def __init__(self,input_dimension,output_dimension,hidden_dimension):
        super(FeedForwardNetwork,self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.linear_1 = nn.Linear(self.input_dimension,self.hidden_dimension)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden_dimension,self.output_dimension)

    def forward(self,x):
        input = x.detach().clone()
        layer_1 = self.relu(self.linear_1(input))
        layer_2 = self.relu(self.linear_2(layer_1))
        return layer_2
