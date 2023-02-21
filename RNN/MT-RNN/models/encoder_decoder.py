import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # x: (batch_size, seq_len)
    embedded = self.dropout(self.embedding(x))
    # embedded: (batch_size, seq_len, embedding_size)
    output, (hidden, cell) = self.rnn(embedded)
    # output: (batch_size, seq_len, hidden_size)
    # hidden: (num_layers, batch_size, hidden_size)
    # cell: (num_layers, batch_size, hidden_size)
    return hidden, cell