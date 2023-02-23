import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .deep_lstm import DeepLSTM

def init_weight(module: nn.Module):
  if type(module) == nn.Linear:
    nn.init.xavier_uniform_(module.weight)

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.lstm = DeepLSTM(input_dim=embedding_size, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)

  def _init_weight(self):
    self.lstm.init_weight(init_weight)
    self.apply(init_weight)

  def forward(self, X, state):
    # X: batch_size, seq_len
    # embs: batch_size, seq_len, embedding_size
    embs = self.embedding(X)
    embs = F.relu(embs)
    # output: batch_size, seq_len, hidden_size (last layer hiddens)
    # hidden: num_layers, batch_size, hidden_size (only takes the last hidden state for each layer)
    # cell: num_layers, batch_size, hidden_size
    outputs, (hidden, cell) = self.lstm(embs.transpose(0, 1), state)
    #outputs: seq_len, batch_size, hidden_size
    outputs = outputs.permute(1, 0, 2)
    #outputs: batch_size, seq_len, hidden_size
    return outputs, (hidden, cell)

class Decoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(Decoder, self).__init__()
    self.vocab_size = vocab_size,
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.num_layers = num_layers
    self.lstm = DeepLSTM(input_dim=embedding_size, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)
    self.fc = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def _init_weight(self):
    self.lstm.init_weight(init_weight)
    self.apply(init_weight)
  
  def init_hidden(self, batch_size):
    return self.lstm.init_hidden(batch_size)
  
  def forward(self, y, H_C):
    # y: batch_size, seq_len
    embs = self.embedding(y)
    embs = F.relu(embs)
    # embs: batch_size, seq_len, embedding_size
    # context: seq_len, batch_size, hidden_size
    outputs, (hidden, cell) = self.lstm(embs.transpose(0, 1), H_C)
    # outputs: seq_len, batch_size, hidden_size
    outputs = self.softmax(self.fc(outputs))
    # outputs: seq_len, batch_size, vocab_size
    outputs = outputs.permute(1, 0, 2)
    # outputs: batch_size, seq_len, vocab_size
    # Hidden: num_layers, batch_size, hidden_size
    return outputs, (hidden, cell)

class AttnDecoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(AttnDecoder, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    
  def _init_weight(self):
    self.lstm.init_weight(init_weight)
    self.apply(init_weight)
  
  def forward(self, y, context, H_C):
    # here the context is a sequence length of hiddens
    # batch_size, seq_len, hidden_size
    
    pass

class EncoderDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.encoder._init_weight()
    self.decoder._init_weight()

  def forward(self, X, y):
    # Teach forcing, pass a whole sentence y to decoder
    encoded_input, _ = self.encoder(X, None) # No need to initalize the hidden first, the lstm will automatically initialize one
    # encoded_input: batch_size, seq_len, hidden_size
    context = encoded_input[:, -1, :] # only take the last hidden state in the sequence
    # context: batch_size, hidden_size
    # y: batch_size, seq_len
    _, C = self.decoder.init_hidden(X.size()[0])
    # context: num_layers, batch_size, hidden_size
    outputs, _ = self.decoder(y, (context.repeat(self.decoder.num_layers, 1, 1), C))
    return outputs # (batch_size, seq_len, vocab_size)
  
  def predict_step(self, X, y, mode='train', max_length=None):
    if mode == 'train':
      # During teaching forcing, the output sentence must have a similar length to the label
      num_steps = y.size()[1]
    else:
      # During evaluate process, decoder generate as many as tokens as it likes, only until it meets the <EOS> token or max_length
      num_steps = max_length
    # y: batch_size, seq_len
    encoded_input, _ = self.encoder(X, None)
    # encoded_input: batch_size, seq_len, hidden_size
    context = encoded_input[:, -1, :] # only take the last layer hidden state
    # context: batch_size, hidden_size
    pred_tokens = y[:, 0].unsqueeze(1) # Take the first token of y <BOS>
    # pred_tokens: batch_size, 1
    outputs = []
    # outputs: num_steps, batch_size, vocab_size (distribution)
    _, C = self.decoder.init_hidden(X.size()[0])
    H_C = (context.repeat(self.decoder.num_layers,1,1), C)
    
    for i in range(num_steps):
      distribution, H_C = self.decoder(pred_tokens[:,-1].unsqueeze(1), H_C) # Take the last step of output
      # distribution: batch_size, 1, vocab_size
      pred_token = torch.argmax(distribution, dim=-1) # batch_size, 1
      pred_tokens = torch.cat((pred_tokens, pred_token), dim=-1)
      outputs.append(distribution.squeeze(1)) # -> batch_size, vocab_soze
    # outputs: seq_len, batch_size, vocab_size
    # pred_tokens: batch_size, seq_len
    return torch.stack(outputs, dim=0).transpose(0, 1), pred_tokens

if __name__ == "__main__":
  encoder = Encoder(vocab_size=5, embedding_size=8, hidden_size=10, num_layers=2, dropout=0.5)
  decoder = Decoder(vocab_size=6, embedding_size=8, hidden_size=10, num_layers=2, dropout=0.5)
  model = EncoderDecoder(encoder, decoder)
  X = torch.randint(0, 4, (4, 8))
  y = torch.randint(0, 5, (4, 20))
  # outputs: batch_size, seq_len, vocab_size
  outputs = model(X, y)
  model.predict_step(X, y, mode='test', max_length=10)
  print(outputs.size())