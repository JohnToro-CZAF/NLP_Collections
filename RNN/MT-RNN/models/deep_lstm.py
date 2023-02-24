import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout):
    super(LSTMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.ih2F = nn.Linear(input_dim+hidden_dim, hidden_dim)
    self.ih2I = nn.Linear(input_dim+hidden_dim, hidden_dim)
    self.ih2C = nn.Linear(input_dim+hidden_dim, hidden_dim)
    self.ih2O = nn.Linear(input_dim+hidden_dim, hidden_dim)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.ic2o = nn.Linear(input_dim+hidden_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.LogSoftmax(dim=1)

  def init_weight(self, f):
    self.apply(f)
  
  def init_hidden(self, batch_size):
    return [torch.zeros(batch_size, self.hidden_dim), torch.zeros(batch_size, self.hidden_dim)]
  
  def forward(self, input, H_C):
    H, C = H_C
    H = H.to(input.device)
    C = C.to(input.device)
    combined = torch.concat((input, H), dim=1)
    F_gate = self.dropout(self.sigmoid(self.ih2F(combined)))
    I_gate = self.dropout(self.sigmoid(self.ih2I(combined)))
    C_gate = self.dropout(self.tanh(self.ih2C(combined)))
    O_gate = self.dropout(self.sigmoid(self.ih2O(combined)))

    C_new = C * F_gate + I_gate * C_gate
    H = self.tanh(C_new) * O_gate
    return H, [H, C_new]

class LSTMLayer(nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout):
    super(LSTMLayer, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.lstm = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)

  def init_hidden(self, batch_size):
    # one for cell, one for hidden
    return torch.zeros((2, batch_size, self.hidden_dim))

  def init_weight(self, f):
    self.lstm.init_weight(f)

  def forward(self, X, H_C):
    #X : seq_len, batch_size, input_dim
    if H_C is None:
      H, C = self.init_hidden(X.size()[1])
    else:
      H, C = H_C
    outputs = []
    for input in X:
      # input: batch_size, input_dim
      output, (H, C) = self.lstm(input, (H, C))
      outputs.append(output)
    # outputs: seq_len, batch_size, hidden_dim
    return torch.stack(outputs, dim=0), [H, C]

class DeepLSTMSequential(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
    super(DeepLSTMSequential, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
    self.input_layer = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    self.layers = nn.ModuleList([LSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers-1)])

  def init_weight(self, f):
    self.input_layer.init_weight(f)
    for layer in self.layers:
      layer.init_weight(f)

  def init_hidden(self, batch_size):
    H, C = [], []
    h, c = self.input_layer.init_hidden(batch_size)
    H.append(h), C.append(c)
    for layer in self.layers:
      h, c = layer.init_hidden(batch_size)
      H.append(h), C.append(c)
    return [torch.stack(H), torch.stack(C)]
    
  def forward(self, X, state):
    # X: seq_len, batch_size, input_dim,
    # state = (H, C): num_layers, batch_size, hidden_dim
    if state is None:
      state = self.init_hidden(X.size()[1])

    last_hiddens, last_cells = [], []
    outputs, (hidden, cell) = self.input_layer(X, (state[0][0], state[1][0]))
    last_hiddens.append(hidden), last_cells.append(cell)
    # outputs are the hiddens in the layer
    # outputs: seq_len, batch_size, hidden_dim
    for ith, layer in enumerate(self.layers):
      outputs, (H, C) = layer(outputs, (state[0][ith+1], state[1][ith+1]))
      last_hiddens.append(H), last_cells.append(C)
    
    # last_hiddens: num_layers, batch_size, hidden_dim
    # outputs: seq_len, batch_size, hidden_dim
    return outputs, [torch.stack(last_hiddens), torch.stack(last_cells)]

class DeepLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
    super(DeepLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
    self.input_layer = LSTMLayer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    self.layers = nn.ModuleList([LSTMLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers-1)])

  def init_weight(self, f):
    self.input_layer.init_weight(f)
    for layer in self.layers:
      layer.init_weight(f)

  def init_hidden(self, batch_size):
    H, C = [], []
    h, c = self.input_layer.init_hidden(batch_size)
    H.append(h), C.append(c)
    for layer in self.layers:
      h, c = layer.init_hidden(batch_size)
      H.append(h), C.append(c)
    return [torch.stack(H), torch.stack(C)]
    
  def forward(self, X, state=None):
    # X: seq_len, batch_size, input_dim,
    # state = (H, C): num_layers, batch_size, hidden_dim
    if state is None:
      state = self.init_hidden(X.size()[1])

    last_hiddens, last_cells = [], []
    outputs, (hidden, cell) = self.input_layer(X, (state[0][0], state[1][0]))
    last_hiddens.append(hidden), last_cells.append(cell)
    # outputs are the hiddens in the layer
    # outputs: seq_len, batch_size, hidden_dim
    for ith, layer in enumerate(self.layers):
      outputs, (H, C) = layer(outputs, (state[0][ith+1], state[1][ith+1]))
      last_hiddens.append(H), last_cells.append(C)
    
    # last_hiddens: num_layers, batch_size, hidden_dim
    # outputs: seq_len, batch_size, hidden_dim
    return outputs, [torch.stack(last_hiddens), torch.stack(last_cells)]
  
if __name__ == "__main__":
  lstm = DeepLSTM(input_dim=10, hidden_dim=20, num_layers=4)
  
  def init_weight(module: nn.Module):
    if type(module) == nn.Linear:
      nn.init.xavier_uniform_(module.weight)
    
  lstm.init_weight(init_weight)
  X = torch.rand((3, 5, 10))
  outputs, (last_hiddens, last_cells) = lstm(X)
  print(outputs.size())
  print(last_hiddens.size())
  print(last_cells.size())