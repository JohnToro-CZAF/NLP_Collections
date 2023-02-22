import torch
import torch.nn as nn
import torch.nn.funtional as F

class LSTMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout):
    super(LSTMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout

class LSTMLayer(nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout):
    super(LSTMLayer, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout
  
  def init_weight(self, ):
    return None

  def init_hidden(self, batch_size):
    # one for cell, one for hidden
    return torch.zeros((2, batch_size, self.hidden_dim))

  def forward(self, X):
    for input in X:
      # input: batch_size, input_dim
      
    return None

class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5)
    super(LSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
    self.input_layer = LSTMLayer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    self.layers = nn.ModuleList([LSTMLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)])

  def init_weight(self):
    self.input_layer.init_weight()
    for layer in self.layers:
      layer.init_weight()

  def forward(self, X):
    # X: seq_len, batch_size, input_dim
    last_hiddens, last_cells = [], []
    H, C = self.input_layer.init_hidden(X.size()[1])
    last_hiddens.append(H), last_cells.append(C)
    outputs, (hidden, cell) = self.input_layer(X, (H, C))
    # outputs are the hiddens in the layer
    # outputs: seq_len, batch_size, hidden_dim
    for layer in self.layers:
      H, C = layer.init_hidden(X.size()[1])
      outputs, (H, C) = layer(outputs, (H, C))
      last_hiddens.append(H), last_cells.append(C)
    
    # last_hiddens: num_layers, batch_size, hidden_dim
    return outputs, (last_hiddens, last_cells)