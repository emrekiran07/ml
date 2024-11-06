import torch
import torch.nn as nn
import math
from torch.optim import Adam


class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # usually mean cancels the dimension but we want to keep it
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # relu adds non-linearity that allows the model more complex patterns


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h  = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, x, mask):

        # Project the input to the correct space (d_model size)
        query = self.w_q(x) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(x)   # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(x) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)    


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, input_size, d_model, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        self.input_embedding = nn.Linear(input_size, d_model)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        
        x = self.input_embedding(x)

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class ShotClassifierModel(nn.Module):

    def __init__(self, d_model, encoder: Encoder) -> None:
        super().__init__()

        self.encoder = encoder

        # Classification layer 
        self.classification_layer = nn.Linear(d_model, 1)
    
    def forward(self, x):

        x = encoder(x, mask=None)

        # X has shape of BS, num_players, d_model
        x = torch.mean(x, axis=1)

        # x has shape BS, d_model
        prob = torch.sigmoid(self.classification_layer(x))

        return prob


if __name__ == "__main__":

    multi_head_block = MultiHeadAttentionBlock(256, 8, 0.2)
    ff_block = FeedForwardBlock(256, 1024, 0.2)
    encoder_block = EncoderBlock(input_size=10, d_model=256, self_attention_block=multi_head_block, feed_forward_block=ff_block, dropout=0.2)
    encoder_list = nn.ModuleList([encoder_block])
    encoder = Encoder(encoder_list)

    classification_model = ShotClassifierModel(d_model=256, encoder=encoder)
    x = torch.randn(32, 22, 10)
    y = torch.randint(0, 2, (32, 1)).float()

    optimizer = Adam(classification_model.parameters(), lr=1e-4)
    loss_f = torch.nn.BCELoss()

    # Train the model in one batch
    for e in range(1000):
        optimizer.zero_grad() 
        output = classification_model(x)
        loss = loss_f(output, y)
        loss.backward()
        optimizer.step()

        print(f"Loss at epooch {e}: {loss}")
    
    # Load the real dataset
    # DataLoader pytorch

    # For each epoch:
    #   shuffle the dataset
    #   for each minibatch:
    #       optimizer.zero_grad()
    #       estimated_output = classification_model(b)
    #       loss = loss_f(estimated_output, y)
    #       loss.backward()
    #       optimizer.step()

    






