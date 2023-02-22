import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm import DeepLSTM

def init_weight(moudle: nn.Module):
  if type(module) == nn.Linear:
    nn.init.xavier_uniform_(module.weight)
  else:
    for param in module._flat_weights_names:
      if 'weight' in param:
        nn.init.xavier_uniform_(module._parameters[param])

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.lstm = DeepLSTM(input_dim=embedding_size, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)

  def _init_weight(self):
    self.lstm.init_weight()
    self.apply(init_weight)

  def forward(self, X):
    # X: batch_size, seq_len, vocab_size
    # embs: batch_size, seq_len, embedding_size
    embs = self.embedding(X)
    # output: batch_size, seq_len, hidden_size (last layer hiddens)
    # hidden: num_layers, batch_size, hidden_size (only takes the last hidden state for each layer)
    # cell: num_layers, batch_size, hidden_size
    outputs, (hidden, cell) = self.lstm(self.embedding(X))
    return outputs, (hidden, cell)

class Decoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(Decoder, self).__init__()
    self.vocab_size = vocab_size,
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.lstm = DeepLSTM(input_dim=(embedding_size+hidden_size), hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)
    self.fc = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def _init_weight(self):
    self.lstm.init_weight()
    self.apply(init_weight)
  
  def forward(self, y, context):
    # y: batch_size, seq_len, vocab_size
    embs = self.embedding(y)
    # embs: batch_size, seq_len, embedding_size
    # context: seq_len, batch_size, hidden_size
    outputs, (hidden, cell) = self.lstm(torch.cat((embs.t(), context), dim=-1))
    # outputs: seq_len, batch_size, hidden_size
    outputs = self.softmax(self.fc(outputs))
    # outputs: seq_len, batch_size, vocab_size
    outputs = outputs.permute(1, 0, 2)
    # outputs: batch_size, seq_len, vocab_size
    return outputs

class EncoderDecoder(nn.Moudle):
  def __init__(self, encoder, decoder):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.encoder._init_weight()
    self.decoder._init_weight()

  def forward(self, X, y, *args):
    encoded_input, _ = self.encoder(X)
    # hiddens: num_layers, batch_size, hidden_size
    context = encoded_input[-1] # only take the last layer hidden state
    # context: batch_size, hidden_size
    context = context.repeat(y.size()[1], 1, 1)
    # context: seq_len, batch_size, hidden_size
    # y: batch_size, seq_len, vocab_size
    outputs = self.decoder(y, context)
    return outputs # (batch_size, seq_len, vocab_size)