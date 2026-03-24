import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, D, n_heads, masked=False):
        super().__init__()
        self.D = D
        self.n_heads = n_heads
        assert self.D % self.n_heads == 0
        self.head_dim = self.D // self.n_heads
        self.d_k = self.head_dim
        self.masked = masked

        self.W_Q = nn.Linear(self.D, self.D)
        self.W_K = nn.Linear(self.D, self.D)
        self.W_V = nn.Linear(self.D, self.D)
        self.W_out = nn.Linear(self.D, self.D)

    def forward(self, Q, K, V):
        Q = self.W_Q(Q) #(B, T, D)
        K = self.W_K(K) #(B, S, D)
        V = self.W_V(V) #(B, S, D)

        B, T, D = Q.shape
        B, S, D = K.shape

        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) #(B, T, D) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2) #(B, S, D) -> (B, S, n_heads, head_dim) -> (B, n_heads, S, head_dim)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2) #(B, S, D) -> (B, S, n_heads, head_dim) -> (B, n_heads, S, head_dim)

        scores = Q @ K.transpose(-1, -2) # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, S) -> (B, n_heads, T, S)
        scores = scores / math.sqrt(self.d_k) #(B, n_heads, T, S)
        if self.masked:
            mask = torch.tril(torch.ones(T, S)).unsqueeze(0).unsqueeze(0) #(T, S) -> (1, T, S) -> (1, 1, T, S)
            scores = torch.masked_fill(scores, mask == 0, -1e9) #(B, n_heads, T, S)
        attention_weights = torch.softmax(scores, dim=-1) #(B, n_heads, T, S)
        outputs = attention_weights @ V # (B, n_heads, T, S) @ (B, n_heads, S, head_dim) = (B, n_heads, T, head_dim)

        outputs = outputs.transpose(1, 2).contiguous().view(B, T, D) #(B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, D)
        return self.W_out(outputs)

class FeedForward(nn.Module):
    def __init__(self, D, ffn_dim):
        super().__init__()
        self.expand = nn.Linear(in_features=D, out_features=ffn_dim)
        self.transform = nn.Linear(in_features=ffn_dim, out_features=ffn_dim)
        self.compress = nn.Linear(in_features=ffn_dim, out_features=D)

    def forward(self, x):
        return torch.relu(self.compress(torch.relu(self.transform(torch.relu(self.expand(x))))))

class Encoder(nn.Module):
    def __init__(self, D, ffn_dim, n_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.self_attention = MultiHeadAttention(D, n_heads)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=D)
        self.feed_forward = FeedForward(D, ffn_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=D)

    def forward(self, x):
        for layer in range(self.num_layers):
            skip = x
            x = self.layer_norm1(skip + self.self_attention(x, x, x))
            skip = x
            x = self.layer_norm2(skip + self.feed_forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, D, ffn_dim, n_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.masked_self_attention = MultiHeadAttention(D, n_heads, masked=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=D)
        self.cross_attention = MultiHeadAttention(D, n_heads)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=D)
        self.feed_forward = FeedForward(D, ffn_dim)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=D)

    def forward(self, x, memory):
        for layer in range(self.num_layers):
            skip = x
            x = self.layer_norm1(skip + self.masked_self_attention(x, x, x))
            skip = x
            x = self.layer_norm2(skip + self.cross_attention(x, memory, memory))
            skip = x
            x = self.layer_norm3(skip + self.feed_forward(x))
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, pos, d):
        super().__init__()
        self.pos = pos
        self.d = d
        self.positional_encodings = self.get_positional_encodings()

    def get_positional_encodings(self):
        positional_encodings = torch.zeros(self.pos, self.d) # (pos, D)
        positions = torch.arange(0, self.pos).unsqueeze(1) #(pos, 1)
        index = torch.arange(0, self.d, 2) #(d/2, )
        freq = torch.exp(index * (-math.log(10000.0) / self.d)) #(d/2, )
        angle = positions * freq # (pos, 1) * (d/2, ) -> (pos, 1) * (1, d/2) -> (pos, d/2)
        positional_encodings[:, 0::2] = torch.sin(angle)
        positional_encodings[:, 1::2] = torch.cos(angle)
        return positional_encodings #(pos, D)

    def forward(self, x):
        B, T = x.shape
        pe = self.positional_encodings[:T, :] # (T, D)
        pe = pe.unsqueeze(0).repeat(B, 1, 1) # (T, D) -> (1, T, D) -> (B, T, D)
        return pe

class Transformer(nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, d_model, enc_ffn, dec_ffn, source_seq_len, target_seq_len,
                 n_heads, num_layers):
        super().__init__()
        self.input_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.input_pos_encoder = PositionalEncoder(source_seq_len, EMBEDDING_DIM)
        self.encoder = Encoder(d_model, enc_ffn, n_heads, num_layers)

        self.output_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.output_pos_encoder = PositionalEncoder(target_seq_len, EMBEDDING_DIM)
        self.decoder = Decoder(d_model, dec_ffn, n_heads, num_layers)

        self.output_layer = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, enc_inps, dec_inps):
        # enc_inputs : (B, S) , dec_inps : (B, T)
        enc_token_embeddings = self.input_embeddings(enc_inps) # (B, S, E)
        enc_positional_encodings = self.input_pos_encoder(enc_inps) # (B, S, E)
        enc_embeddings = enc_token_embeddings + enc_positional_encodings # (B, S, E) + (B, S, E) = (B, S, E)
        encoder_outputs = self.encoder(enc_embeddings) # (L, B, S, E)

        dec_token_embeddings = self.output_embeddings(dec_inps) #(B, T, E)
        dec_positional_encodings = self.output_pos_encoder(dec_inps) #(B, T, E)
        dec_embeddings = dec_token_embeddings + dec_positional_encodings # (B, T, E) + (B, T, E) = (B, T, E)
        decoder_outputs = self.decoder(dec_embeddings, encoder_outputs) #(B, T, E)

        logits = self.output_layer(decoder_outputs) #(B, T, V)
        return logits