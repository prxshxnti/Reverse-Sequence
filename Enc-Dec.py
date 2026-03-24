import json
import time
import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import logging
import sys
from transformers import Transformer
import os

#Observed Maximum strength of RNN : 2026-03-15_19-23-04 [ Used only small sequences, len 7 for train, val, test ]
#Observed RNN failure for long sequences : 2026-03-15_19-33-22 [ Trained on 10 and evaluated on 15]
#Observed little change in LSTM for long sequences, Loss curves didn't diverge like RNN : 2026-03-15_20-02-34
#Observed very minimal change in accuracy GRU but the thing is converged faster in 10 epochs, so it got over-trained in 20 epochs
#Using bidirectional GRU, there is still some diverging after 8 epochs, train loss increases
#Using bidirectional Gru with bridge the loss didn't increase.

# Small sequences RNN :  Test Loss : 1.314, Test Token Acc : 0.601, Test Seq Acc : 0.004
# Long sequences RNN : Test Loss : 3.350, Test Token Acc : 0.176, Test Seq Acc : 0.000
# LSTM : Test Loss : 2.951, Test Token Acc : 0.144, Test Seq Acc : 0.000
# GRU : Test Loss : 3.853, Test Token Acc : 0.153, Test Seq Acc : 0.000
# Bidirectional GRU without bridge : Test Loss : 3.053, Test Token Acc : 0.152, Test Seq Acc : 0.000
# Bidirectional GRU without bridge(lr 0.1 to 0.01) : Test Loss : 2.968, Test Token Acc : 0.073, Test Seq Acc : 0.000
# Bidirectional GRU with bridge : Test Loss : 2.933, Test Token Acc : 0.129, Test Seq Acc : 0.000
# Bidirectional GRU with bridge(lr 0.1 to 0.01) : Test Loss : 2.973, Test Token Acc : 0.068, Test Seq Acc : 0.000
# Deep Bidirectional GRU : Test Loss : 2.974, Test Token Acc : 0.063, Test Seq Acc : 0.000
#2026-03-17_15-01-33 using bahdanu attention - Test Loss : 0.008, Test Token Acc : 1.000, Test Seq Acc : 1.000

#PARAMETERS
REASON = "Transformer from scratch and fixed some mistakes "

VERSION = time.strftime("%Y-%m-%d_%H-%M-%S")
VOCAB_SIZE = 20
NUM_EPOCHS = 10
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
ATTENTION_SIZE = 128

TRAINING_SAMPLES, TRAINING_SEQ_LEN = 2500, 20
VALIDATION_SAMPLES, VALIDATION_SEQ_LEN = 500, 20
TEST_SAMPLES, TEST_SEQ_LEN = 200, 20

MAX_LEN = max(TRAINING_SEQ_LEN, max(VALIDATION_SEQ_LEN, TEST_SEQ_LEN))

SOS, EOS = VOCAB_SIZE - 2, VOCAB_SIZE - 1

LEARNING_RATE = 0.001
OPTIMIZER_FN = "ADAM"

DIR_PATH = "history"

os.makedirs(f"{DIR_PATH}/{VERSION}", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # console
        # logging.FileHandler(f"{DIR_PATH}/{VERSION}/app.log")  # file
    ]
)

LOGGER = logging.getLogger()

torch.manual_seed(123)
random.seed(123)

def generate_data(num_samples, seq_len, vocab_size):
    X, y_inps, y_targets = [], [], []
    for _ in range(num_samples):
        random_nos = [random.randint(1, vocab_size-2) for _ in range(seq_len)]
        reversed_nos = random_nos[::-1]

        X.append(random_nos)
        #Implement teacher forcing
        y_inps.append([SOS] + reversed_nos) #adding start of sequence token
        y_targets.append(reversed_nos + [EOS]) #adding end of sequence token

    return torch.tensor(X), torch.tensor(y_inps), torch.tensor(y_targets)

def generate_data_new(num_samples, seq_len, vocab_size):
    X, y_inps, y_targets = [], [], []
    for _ in range(num_samples):
        seq_len = random.randint(5, 5 + seq_len)
        random_nos = [random.randint(1, vocab_size-2) for _ in range(seq_len)]
        reversed_nos = random_nos[::-1]

        X.append(random_nos)
        #Implement teacher forcing
        y_inps.append([SOS] + reversed_nos) #adding start of sequence token
        y_targets.append(reversed_nos + [EOS]) #adding end of sequence token

    lengths_X = max(len(x) for x in X)
    lengths_y = max(len(y) for y in y_inps)

    padded_X, padded_inp, padded_target = [], [], []
    for i in range(num_samples):
        X_i, inp_i, target_i = X[i], y_inps[i], y_targets[i]
        padded_X.append(X_i + ([0] * (lengths_X - len(X_i))))
        padded_inp.append(inp_i + ([0] * (lengths_y - len(inp_i))))
        padded_target.append(target_i + ([0] * (lengths_y - len(target_i))))

    return torch.tensor(padded_X), torch.tensor(padded_inp), torch.tensor(padded_target)

X_train, y_train_inp, y_train_targets = generate_data(TRAINING_SAMPLES, TRAINING_SEQ_LEN, VOCAB_SIZE)
X_val, y_val_inp, y_val_targets  = generate_data(VALIDATION_SAMPLES, VALIDATION_SEQ_LEN, VOCAB_SIZE)
X_test, y_test_inp, y_test_targets  = generate_data(TEST_SAMPLES, TEST_SEQ_LEN, VOCAB_SIZE)

LOGGER.info(f"X_train shape : {X_train.shape}, y_train_inp shape : {y_train_inp.shape}, y_train_targets shape : {y_train_targets.shape},"
      f"X_val shape : {X_val.shape}, y_val_inp shape : {y_val_inp.shape}, y_val_targets shape : {y_val_targets.shape},"
      f"X_test shape : {X_test.shape}, y_test_inp shape : {y_test_inp.shape},  y_test_targets shape : {y_test_targets.shape} ")

train_dataset = TensorDataset(X_train, y_train_inp, y_train_targets)
val_dataset = TensorDataset(X_val, y_val_inp, y_val_targets)
test_dataset = TensorDataset(X_test, y_test_inp, y_test_targets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def plot_grad_flow(named_parameters, epoch):
    """
    Interpretation of plot :
    Vanishing gradients : Early layers near zero. Meaning gradients disappear as they propagate backward.
    Exploding gradients : Some layers spike in middle. Meaning gradients blow up.
    """
    layers = []
    avg_grads = []
    max_grads = []

    for name, param in named_parameters:
        if param.grad is not None and "bias" not in name:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    plt.figure(figsize=(10,6))
    plt.bar(range(len(max_grads)), max_grads, alpha=0.3, label="max gradient")
    plt.bar(range(len(avg_grads)), avg_grads, alpha=0.6, label="mean gradient")

    # print(f"Avg grads {list(zip(layers, avg_grads))}") #Debugging how small average gradients are

    plt.xticks(range(len(layers)), layers, rotation=60)
    plt.xlabel("Layers")
    plt.ylabel("Gradient magnitude")
    plt.title(f"Gradient Flow Across Layers for Epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"{DIR_PATH}/{VERSION}/{epoch}_gradient_plot.png")
    plt.show()

class ReverseTaskV0(nn.Module):
    """
    This Architecture involves the usage of a Encoder - Decoder system connected by a context vector
    The encoder/decoder architecture is build using a vanilla RNN
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: (B, S) -> 32, 10
        :param dec_inputs: (B, S) -> 32, 10
        :return: logits -> (B, S, Vocab_size) -> 32, 10, 20
        """
        enc_emb = self.enc_embeddings(enc_inputs) #(B, S, emb_dim) -> 32, 10, 32
        enc_out, enc_hidden = self.encoder(enc_emb) #(B, S, H), (L, B, H) -> (32, 10, 16), (1, 32, 16)

        dec_emb = self.dec_embeddings(dec_inputs)#(B, S, emb_dim) -> 32, 10, 32
        dec_out, dec_hidden = self.decoder(dec_emb, enc_hidden) #(B, S, H), (L, B, H) -> (32, 10, 16), (1, 32, 16)
        logits = self.output_layer(dec_out) #B, S, V -> 32, 10, 20
        return logits

class ReverseTaskV1(nn.Module):
    """
    This Architecture uses LSTM cell
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        enc_emb = self.enc_embeddings(enc_inputs)
        enc_out, (enc_hidden, enc_cell) = self.encoder(enc_emb)
        dec_emb = self.dec_embeddings(dec_inputs)
        dec_out, (dec_hidden, dec_cell) = self.decoder(dec_emb, (enc_hidden, enc_cell))
        logits = self.output_layer(dec_out)
        return logits

class ReverseTaskV2(nn.Module):
    """
    This Architecture uses GRU cell
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        enc_emb = self.enc_embeddings(enc_inputs)
        enc_out, enc_hidden = self.encoder(enc_emb)
        dec_emb = self.dec_embeddings(dec_inputs)
        dec_out, dec_hidden = self.decoder(dec_emb, enc_hidden)
        logits = self.output_layer(dec_out)
        return logits

class ReverseTaskV3(nn.Module):
    """
    This architecture involves the use of bidirectional encoders using GRU cell
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=2*HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(2*HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #(B, S) and (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_outputs, enc_hidden  = self.encoder(enc_emb) #(B, S, 2*H), (2, B, H)

        contex_vector = torch.cat([enc_hidden[0], enc_hidden[1]], dim=1) # (B, H ) + (B, H) along 1 = (B, 2*H)
        contex_vector = contex_vector.unsqueeze(0) #1, B, 2*H

        dec_emb = self.dec_embeddings(dec_inputs) #B, S, E
        dec_outputs, dec_hidden = self.decoder(dec_emb, contex_vector) #(B, S, 2*H), (1, B, 2*H)
        logits = self.output_layer(dec_outputs) #B, S, 2*H -> B, S, Vocabsize
        return logits

class ReverseTaskV4(nn.Module):
    """
    This architecture involves the use of bidirectional encoders using GRU cell and bridge layer
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #(B, S) and (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_outputs, enc_hidden  = self.encoder(enc_emb) #(B, S, 2*H), (2, B, H)

        contex_vector = torch.cat([enc_hidden[0], enc_hidden[1]], dim=1) # (B, H ) + (B, H) along 1 = (B, 2*H)
        contex_vector = self.bridge(contex_vector) #B, H
        contex_vector = contex_vector.unsqueeze(0) #1, B, H

        dec_emb = self.dec_embeddings(dec_inputs) #B, S, E
        dec_outputs, dec_hidden = self.decoder(dec_emb, contex_vector) #(B, S, H), (1, B, H)
        logits = self.output_layer(dec_outputs) #B, S, H -> B, S, Vocab size
        return logits

class ReverseTaskV5(nn.Module):
    """
    This architecture involves the use of bidirectional encoders using GRU cell and bridge layer and increased layers
    DEEP RNN
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True,
                              bidirectional=True, num_layers=3)
        self.bridge = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True, num_layers=3)
        self.output_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #(B, S) and (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        #for bidirectional GRU with one layer : 2 directions, bidirectional GRU with 3 layers - 6 directions ( num layers * num directions)
        enc_outputs, enc_hidden  = self.encoder(enc_emb) #(B, S, 2*H), (6, B, H)

        #Pick only the recent layers forward and backward states ( 4, 5 )
        contex_vector = torch.cat([enc_hidden[-2], enc_hidden[-1]], dim=1) # (B, H ) + (B, H) along 1 = (B, 2*H)
        contex_vector = self.bridge(contex_vector) #B, H
        contex_vector = contex_vector.unsqueeze(0) #1, B, H

        #Decoder architecture is 3 layers so expects contex vector of 3, B, H
        contex_vector = contex_vector.repeat(3, 1, 1) #duplicate a tensor along a dimension - here only repeat 3 times along first dimension

        dec_emb = self.dec_embeddings(dec_inputs) #B, S, E
        dec_outputs, dec_hidden = self.decoder(dec_emb, contex_vector) #(B, S, H), (1, B, H)
        logits = self.output_layer(dec_outputs) #B, S, H -> B, S, Vocab size
        return logits

class ReverseTaskV6(nn.Module):
    """
    This architecture implements Bahdanau Additive attention implementation
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM + HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.W_h = nn.Linear(HIDDEN_SIZE, ATTENTION_SIZE) #for encoder projection where encoder output is h_i
        self.W_s = nn.Linear(HIDDEN_SIZE, ATTENTION_SIZE) #for decoder projection where prev decoder hidden state is s_t-1
        self.v = nn.Linear(ATTENTION_SIZE, 1, bias=False) #turns the vector to a score

        self.output_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #enc_inputs : (B, S), dec_inputs : (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_out, enc_hidden = self.encoder(enc_emb) # (B, S, H), (1, B, H)

        dec_timesteps = dec_inputs.shape[1]
        prev_hidden = enc_hidden  #s_t-1 (1, B, H)
        outputs = []
        for t in range(dec_timesteps):
            #For every timestep : build the context vector ct using enc_outputs
            enc_proj = self.W_h(enc_out) #B, S, A
            dec_proj = self.W_s(prev_hidden)#1, B, A
            dec_proj = dec_proj.squeeze(dim=0).unsqueeze(dim=1) # 1, B, A -> B, A -> B, 1, A

            add_projections = torch.tanh(enc_proj + dec_proj) #B, S, A + B, 1, A => after broadcasting on dim=1 -> B, S, A
            # add_proj is the vector of combined info from prev hidden and each enc output. Each timestep has  attention vector
            scores = self.v(add_projections).squeeze(dim=-1) #B, S, A => B, S,1 => B, S-> This turns attention vector to attention score

            probs = torch.softmax(scores, dim=1) #converts attention score to attention weights #B, S
            context_vector = torch.bmm(probs.unsqueeze(dim=1), enc_out) #bmm of (B, 1, S) @ (B, S, H) -> B, 1, H
            # -> take the attention score of each timestep and turn it a vector of timesteps.
            # Element of this vector is multiplied with hidden vector at each time step(scaling the hidden vector).
            # Scaled hidden vector for all time steps is blended together into a single vector by adding its elements across hidden size dimension

            #Build the embedding vector
            emb_vector = self.dec_embeddings(dec_inputs[:, t]).unsqueeze(dim=1) #B, 1, E

            #Prepare input for decoder step
            s_t = torch.cat([emb_vector, context_vector],dim=2)#(B, 1, E) + (B, 1 , H) = B, 1, E+H
            dec_out_t , dec_hidden = self.decoder(s_t, prev_hidden) #(B, 1, H),( 1, B, H)

            logits = self.output_layer(dec_out_t) #B, 1, V
            outputs.append(logits)
            prev_hidden = dec_hidden

        return torch.cat(outputs, dim=1)

class ReverseTaskV7(nn.Module):
    """
    This architecture implements Luong general attention implementation
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.W = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) #for projecting the vector from encoder space to decoder space

        self.output_layer = nn.Linear(2*HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #enc_inputs : (B, S), dec_inputs : (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_out, enc_hidden = self.encoder(enc_emb) # (B, S, H), (1, B, H)

        dec_timesteps = dec_inputs.shape[1]
        prev_hidden = enc_hidden  #s_t-1 (1, B, H_d)

        enc_proj = self.W(enc_out) # => (B, S, H_e) -> (B, S, H_d)

        outputs = []
        for t in range(dec_timesteps):
            #Build the embedding vector
            emb_vector = self.dec_embeddings(dec_inputs[:, t]).unsqueeze(dim=1) #B, 1, E

            dec_out , dec_hidden = self.decoder(emb_vector, prev_hidden) #(B, 1, H),( 1, B, H)

            #Build the context vector ct using enc_outputs
            s_t = dec_hidden[-1] #B, H

            scores = torch.bmm(enc_proj, s_t.unsqueeze(dim=2)) #B, S, H @ B, H, 1 => B, S, 1

            probs = torch.softmax(scores, dim=1) #converts attention score to attention weights #B, S, 1

            context_vector = torch.bmm(probs.transpose(1, 2), enc_out) #bmm of (B, S, 1) @ (B, S, H) -> B, 1, H
            # -> take the attention score of each timestep and turn it a vector of timesteps.
            # Element of this vector is multiplied with hidden vector at each time step(scaling the hidden vector).
            # Scaled hidden vector for all time steps is blended together into a single vector by adding its elements across hidden size dimension

            logits = self.output_layer(torch.cat([dec_out, context_vector], dim=2)) #B, 1, 2H -> B, 1, V
            outputs.append(logits)
            prev_hidden = dec_hidden

        return torch.cat(outputs, dim=1)

class ReverseTaskV8(nn.Module):
    """
    This architecture implements Luong dot attention implementation
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.output_layer = nn.Linear(2*HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #enc_inputs : (B, S), dec_inputs : (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_out, enc_hidden = self.encoder(enc_emb) # (B, S, H), (1, B, H)

        dec_timesteps = dec_inputs.shape[1]
        prev_hidden = enc_hidden  #s_t-1 (1, B, H_d)

        outputs = []
        for t in range(dec_timesteps):
            #Build the embedding vector
            emb_vector = self.dec_embeddings(dec_inputs[:, t]).unsqueeze(dim=1) #B, 1, E

            dec_out , dec_hidden = self.decoder(emb_vector, prev_hidden) #(B, 1, H),( 1, B, H)

            #Build the context vector ct using enc_outputs
            s_t = dec_hidden[-1] #B, H #to fetch the hidden of last layer

            scores = torch.bmm(enc_out, s_t.unsqueeze(dim=2)) #B, S, H @ B, H, 1 => B, S, 1

            probs = torch.softmax(scores, dim=1) #converts attention score to attention weights #B, S, 1

            context_vector = torch.bmm(probs.transpose(1, 2), enc_out) #bmm of transpose((B, S, 1)) @ (B, S, H) -> B, 1, H
            # -> take the attention score of each timestep and turn it a vector of timesteps.
            # Element of this vector is multiplied with hidden vector at each time step(scaling the hidden vector).
            # Scaled hidden vector for all time steps is blended together into a single vector by adding its elements across hidden size dimension

            logits = self.output_layer(torch.cat([dec_out, context_vector], dim=2)) #B, 1, 2H -> B, 1, V
            outputs.append(logits)
            prev_hidden = dec_hidden

        return torch.cat(outputs, dim=1)

class ReverseTaskV9(nn.Module):
    """
    Transformer architecture from "Attention is all you need" paper using pre-built transformer blocks
    """
    def __init__(self, d_model):
        super().__init__()
        self.input_embeddings = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_input_embeddings = nn.Embedding(MAX_LEN, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.output_embeddings = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_output_embeddings = nn.Embedding(MAX_LEN + 1, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.output_layer = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, enc_inps, dec_inps):
        B, S = enc_inps.shape  # S - source seq len
        enc_position_ids = torch.arange(0, S).unsqueeze(0).repeat(B, 1)
        enc_token_embeddings = self.input_embeddings(enc_inps) # (B, S) -> (B, S, D )
        enc_pos_embeddings = self.pos_input_embeddings(enc_position_ids) # (B, S) -> ( B, S, D)
        enc_embeddings = enc_token_embeddings + enc_pos_embeddings # ( B, S, D ) + ( B, S, D ) = ( B, S, D)
        enc_outputs = self.encoder(enc_embeddings) # (B, S, D)

        B, T = dec_inps.shape #T - Target seq len
        dec_position_ids = torch.arange(0, T).unsqueeze(0).repeat(B, 1)
        dec_token_embeddings = self.output_embeddings(dec_inps) # (B, T) -> (B, T, D)
        dec_pos_embeddings = self.pos_output_embeddings(dec_position_ids) # (B, T) -> (B, T, D)
        dec_embeddings = dec_token_embeddings + dec_pos_embeddings  # ( B, T, D ) + ( B, T, D ) = ( B, T, D)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T)
        dec_outputs = self.decoder(dec_embeddings, enc_outputs, tgt_mask = tgt_mask) # ( B, T, D), (B, S, D) -> ( B, T, D)

        logits = self.output_layer(dec_outputs) # ( B, T, D) -> (B, T, V)
        return logits

model = Transformer(VOCAB_SIZE, EMBEDDING_DIM, EMBEDDING_DIM, 256, 256, MAX_LEN, MAX_LEN+1, 4, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = None
if OPTIMIZER_FN == "SGD":
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER_FN == "ADAM":
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def get_ngrams(values, n):
    values = list(values.detach())
    values = [val.item() for val in values]
    return [values[i : i + n] for i in range(0, len(values) - n + 1)]

def accuracy_fn(predictions, true_values):
    n = 4
    predictions_ngrams, true_values_ngrams = [], []
    for i in range(2, n+1):
        predictions_ngrams.extend(get_ngrams(predictions, i))
        true_values_ngrams.extend(get_ngrams(true_values, i))

    matched = sum(1 if seq in predictions_ngrams else 0 for seq in true_values_ngrams)
    acc = 0.5 * (matched / len(true_values_ngrams)) + 0.5 * (matched/ len(predictions_ngrams))
    return acc

def acc_wrapper(y_preds, dec_target):
    y_preds = torch.argmax(y_preds, dim=-1)
    assert y_preds.shape == dec_target.shape
    acc = []
    for i in range(y_preds.shape[0]):
        acc.append(accuracy_fn(y_preds[i], dec_target[i]))
    return sum(acc) / len(acc)

def sequence_accuracy(y_preds, targets):
    y_preds = torch.argmax(y_preds, dim=-1)
    correct = (y_preds == targets).all(dim=1)
    return correct.float().mean().item()

def token_accuracy(y_preds, targets):
    y_preds = torch.argmax(y_preds, dim=-1)
    correct = (y_preds == targets).float().mean().item()
    return correct

LOGGER.info("="*60)
LOGGER.info("Before Training ")
torch.manual_seed(23)
enc_inp, dec_inp, dec_target = next(iter(train_loader))
LOGGER.info(f"Enc Input : {enc_inp[0:1]}, Dec Input : {dec_inp[0:1]}, Dec Target: {dec_target[0:1]}")
y_preds = model(enc_inp[0:1], dec_inp[0:1])
y_preds = torch.argmax(y_preds, dim=-1)
LOGGER.info(f"Model's Prediction {y_preds}")
LOGGER.info("="*60)

if __name__ == "__main__":
    train_loss, val_loss = [],[]
    train_acc_seq, val_acc_seq, train_acc_token, val_acc_token = [], [], [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss_per_step, val_loss_per_step = [], []
        train_acc_per_step_token, val_acc_per_step_token, train_acc_per_step_seq, val_acc_per_step_seq = [], [], [],[]
        model.train()
        for batch_idx , (enc_inp, dec_inp, dec_target) in enumerate(train_loader):
            y_preds = model(enc_inp, dec_inp)
            B, S, V = y_preds.shape
            loss = loss_fn(y_preds.view(-1, V), dec_target.view(-1))
            optimizer.zero_grad()
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5
            # LOGGER.info(f"Gradient norm : {total_norm:.6f}")
            # 0.01 → 5 ===== Normal gradients and training is stable
            # 10 → 50 → 200 → 1000 ======= Exploding gradients and loss becomes nan [ Solution : clip to 1 ]
            # 1e-4 , 1e-6, 1e-8 ======= Vanishing gradients [ Solution : LayerNorm/Better Initialization/ LSTM/ GRU / ATTENTION / RESIDUAL CONNECTIONS]

            if batch_idx == 0 and epoch % 5 == 0:
                # Healthy training
                # mean gradients ≈ small but visible
                # max gradients ≈ slightly larger

                # Vanishing gradients
                # mean ≈ 1e-7
                # max ≈ 1e-6
                # Bars almost flat near zero.

                # Exploding gradients
                # mean ≈ 0.1
                # max ≈ 50
                # Huge spikes in max gradient.
                plot_grad_flow(model.named_parameters(), epoch)

            optimizer.step()

            train_loss_per_step.append(loss.detach().item())
            train_acc_per_step_token.append(token_accuracy(y_preds, dec_target))
            train_acc_per_step_seq.append(sequence_accuracy(y_preds, dec_target))

        model.eval()
        with torch.inference_mode():
            for enc_inp, dec_inp, dec_target in val_loader:
                y_preds = model(enc_inp, dec_inp)
                B, S, V = y_preds.shape
                loss = loss_fn(y_preds.view(-1, V), dec_target.view(-1))

                val_loss_per_step.append(loss.detach().item())
                val_acc_per_step_token.append(token_accuracy(y_preds, dec_target))
                val_acc_per_step_seq.append(sequence_accuracy(y_preds, dec_target))

        train_loss.append(sum(train_loss_per_step)/ len(train_loss_per_step))
        val_loss.append(sum(val_loss_per_step) / len(val_loss_per_step))
        train_acc_seq.append(sum(train_acc_per_step_seq)/ len(train_acc_per_step_seq))
        val_acc_seq.append(sum(val_acc_per_step_seq)/len(val_acc_per_step_seq))
        train_acc_token.append(sum(train_acc_per_step_token)/ len(train_acc_per_step_token))
        val_acc_token.append(sum(val_acc_per_step_token)/len(val_acc_per_step_token))

        LOGGER.info(f"Epoch : {epoch} / {NUM_EPOCHS} , Train Loss : {train_loss[-1]:.3f}, Validation Loss : {val_loss[-1]:.3f}"
                    f" Train Token Accuracy : {train_acc_token[-1]:.3f}, Val Token Accuracy : {val_acc_token[-1]:.3f}"
                    f" Train Seq Accuracy : {train_acc_seq[-1]:.3f}, Val Seq Accuracy : {val_acc_seq[-1]:.3f}")


    LOGGER.info("="*60)
    LOGGER.info("After Training ")
    torch.manual_seed(23)
    enc_inp, dec_inp, dec_target = next(iter(train_loader))
    LOGGER.info(f"Enc Input : {enc_inp[0:1]}, Dec Input : {dec_inp[0:1]}, Dec Target: {dec_target[0:1]}")
    y_preds = model(enc_inp[0:1], dec_inp[0:1])
    y_preds = torch.argmax(y_preds, dim=-1)
    LOGGER.info(f"Model's Prediction {y_preds}")
    LOGGER.info("="*60)

    test_loss, test_acc_seq, test_acc_token = [], [], []
    model.eval()
    with torch.inference_mode():
        for enc_inp, dec_inp, dec_target in test_loader:
            y_preds = model(enc_inp, dec_inp)
            B, S, V = y_preds.shape
            loss = loss_fn(y_preds.view(-1, V), dec_target.view(-1))
            test_loss.append(loss.detach().item())
            test_acc_seq.append(sequence_accuracy(y_preds, dec_target))
            test_acc_token.append(token_accuracy(y_preds, dec_target))

    LOGGER.info(f"Test Loss : {(sum(test_loss) / len(test_loss)):.3f}, "
                f"Test Token Acc : {(sum(test_acc_token)/len(test_acc_token)):.3f}, "
                f"Test Seq Acc : {(sum(test_acc_seq)/len(test_acc_seq)):.3f}")


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    num_epochs = range(1, NUM_EPOCHS+1)
    axes[0].plot(num_epochs, train_loss, label="Training Loss")
    axes[0].plot(num_epochs, val_loss, label="Validation Loss")
    axes[0].set_xlabel("Num Epochs")
    axes[0].set_label("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()

    axes[1].plot(num_epochs, train_acc_token, label="Training Accuracy")
    axes[1].plot(num_epochs, val_acc_token, label="Validation Accuracy")
    axes[1].set_xlabel("Num Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{DIR_PATH}/{VERSION}/loss_acc_plots.png")
    plt.show()

    #Inference
    def generate(model, e_inp, max_len=20):
        SOS = VOCAB_SIZE - 2
        EOS = VOCAB_SIZE - 1

        d_inp = torch.tensor([[SOS]])
        generated = []

        model.eval()
        with torch.inference_mode():
            for _ in range(max_len):
                logits = model(e_inp, d_inp)

                #Get the prediction at last timestep
                last_timestep = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                next_token = last_timestep.item()
                if next_token == EOS:
                    break
                generated.append(next_token)

                #append next token
                d_inp = torch.cat([d_inp, last_timestep], dim=-1)
        return generated

    enc_inp = torch.tensor([[5, 3, 9, 10, 2, 15, 7, 9, 13, 12, 1, 4, 8, 2, 2, 5, 11, 14, 16, 17]])
    predictions = generate(model, enc_inp)
    LOGGER.info(f"Inference for {enc_inp} :: Model's Predictions {predictions}")

    #Tracking experiments
    with open(f"{DIR_PATH}/{VERSION}/config.json", "w") as f:
        config = {
            "VERSION" : VERSION,
            "VOCAB_SIZE" : VOCAB_SIZE, "NUM_EPOCHS" : NUM_EPOCHS, "BATCH_SIZE" : BATCH_SIZE,
            "EMBEDDING_DIM" : EMBEDDING_DIM , "HIDDEN_SIZE" : HIDDEN_SIZE,
            "TRAINING_SAMPLES" : TRAINING_SAMPLES, "TRAINING_SEQ_LEN" :  TRAINING_SEQ_LEN,
            "VALIDATION_SAMPLES" : VALIDATION_SAMPLES, "VALIDATION_SEQ_LEN" : VALIDATION_SEQ_LEN,
            "TEST_SAMPLES" : TEST_SAMPLES, "TEST_SEQ_LEN" : TEST_SEQ_LEN,
            "SOS" : SOS, "EOS" : EOS, "LEARNING_RATE" : LEARNING_RATE,
            "OPTIMIZER_FN" : OPTIMIZER_FN,
            "MODEL_CLASS" : model.__class__.__name__, "REASON" : REASON,  "ATTENTION_SIZE" : ATTENTION_SIZE
        }
        json.dump(config, f, indent=4)

    torch.save(model.state_dict(), f"{DIR_PATH}/{VERSION}/model.pt")

