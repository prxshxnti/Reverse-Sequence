import math
import torch
import torch.nn as nn

# 3 Angles to look for
# 1. Code angle : Tensor shapes
# 2. English angle : Intuition based
# 3. Linear Algebra angle : Since at the end of the day everything is vectors

"""
Self Attention : Queries == Keys [ Queries and keys come from same inputs (Decoder's inps or Enc's inps)
Cross Attention : Queries != Keys [ Queries are decoder inputs and Keys are encoder outputs ] 

In self attention : words attends other words of the same sentence
In cross attention : words attends other words of different sentence 

Self Attention in Encoder : Q = (B, S, d_model), K = (B, S, d_model), V = (B, S, d_model)
Self Attention in Decoder : Q = (B, T, d_model), K = (B, T, d_model), V = (B, T, d_model)
Cross Attention in Decoder : Q = (B, T, d_model), K = (B, S, d_model), V = (B, S, d_model)

Multi head attention : Different heads attending different parts of a many words in a sentence 
(B , S, d_model) => ( B, S, n_heads, head_dim )

Masked Multi head attention : Multi head attention variant where a head when attends other words for a single word, it will not 
checking similarity for words after the single word. If will not look at future words 

Scaled Dot product attention in plain english : 
1. Check the similarity of words between two sentences [ check similarity btw queries and keys ] 
2. Scale the similarity scores because for large similarity scores, softmax probs will be large and attention weights will be not equally distributed. Dot products grow large - softmax will be very sharp and it becomes one hot attention
3. If mask given, in the scores, turn the similarity scores of the given mask to a negative high value. Exp of negative large value is close to zero 
4. get the attention weights by turns scores into probs 
5. Now weight the values/words of a sentences using this attention weights and return the attended result. This is transformation of values using attention weights 

Masking : 
mask = (T, T) or (1, 1, T, T) or (B, 1, T, T) 
for every row, col > row == 0 ------> True then replace with -1e9
-1e9 is because e^-1e9 == 0 [ to kill attention ]

Masked Multi head attention: 
1. Allow Q, K, V to make projections and learn different perspectives
2. Split the attention job to different heads introducing parallelism 
3. Apply attention per head 
4. Merge heads 

Positional Encoding 

We have a sequence of length p, and we want a d-dimensional positional vector for each position.
Instead of treating all d dimensions independently, we group them into d/2 pairs.
Each pair corresponds to a specific frequency, and within each pair: the even index uses sin , the odd index uses cos

For each position pos and pair index i, we compute an angle: 𝜃 = pos / 10000 ^ 2i/d
pos × frequency → angle → sin/cos

Then encode:

even dimension → sin(θ)
odd dimension → cos(θ)

Different pairs use different frequencies, which control how fast the encoding changes across positions.

Combining all frequency bands allows the model to uniquely represent positions and infer relative distances.
"""

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q = (B, S, d_model), K = (B, S, d_model), V = (B, S, d_model)
    similarity_scores = Q @ K.transpose(-1, -2) # (B, S, d_model) @ (B, d_model, S) = (B, S, S)
    d_k = K.shape[2]
    scores = similarity_scores / math.sqrt(d_k) #(B, S, S)
    if mask is not None :
        scores = scores.masked_fill(mask == 0, -1e9) #(B, S, S)
    attention_weights = torch.softmax(scores, dim=-1) #(B, S, S)
    output = attention_weights @ V # (B, S, S) @ (B, S, d_model) = (B, S, d_model)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.n_heads
        self.d_k = self.head_dim

        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_out = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x,  to_mask=False):
        B, T, d_model = x.shape

        Q = self.W_Q(x) #(B, T, d_model)
        K = self.W_K(x) #(B, T, d_model)
        V = self.W_V(x) #(B, T, d_model)

        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) #(B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) #(B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) #(B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)

        scores = Q @ K.transpose(-1, -2) # (B, n_head, T, head_dim ) @ (B, n_head, head_dim, T) = (B, n_head, T, T)
        scores = scores / math.sqrt(self.d_k) # (B, n_heads, T, T)
        if to_mask:
            mask = torch.tril(torch.ones(T,T)).unsqueeze(dim=0).unsqueeze(dim=0) #(T, T) ->( 1, T, T) -> (1, 1, T, T)
            scores = scores.masked_fill(mask == 0, -1e9)  # (B, n_heads, T, T)
        attention_weights  = torch.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        out = attention_weights @ V  # (B, n_heads, T, T)  @ (B, n_heads, T, head_dim) = (B,n_heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, d_model) #( B, T, n_heads, head_dim ) -> (B, T, d_model)
        return self.W_out(out) #(B, T, d_model)

def positional_encoding(pos, d_model):
    assert d_model % 2 == 0
    pe = torch.zeros(pos, d_model) #(pos, d_model)
    positions = torch.arange(0, pos).unsqueeze(1) #(pos, 1)
    index = torch.arange(0, d_model, 2) #(d_model/2, )
    div_term = torch.exp(index * (- math.log(10000.0)/ d_model))  #(d_model/2, )
    pe[:, 0::2] = torch.sin(positions * div_term) #(pos, 1) * (d_model/2,) → (pos, d_model/2)
    pe[:, 1::2] = torch.cos(positions * div_term) #(pos, 1) * (d_model/2,) → (pos, d_model/2)


if __name__ == "__main__":
    positional_encoding(3, 10)