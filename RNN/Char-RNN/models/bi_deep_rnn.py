import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class RNNLayer(nn.Module):
  def __init__(self, dim_input, dim_hidden, direction=1):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.direction = direction

  def forward(self, input, hidden):
    outputs = []
    if self.direction == 1:
      for i in range(input.size()[1]):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    else: 
      for i in range(input.size()[1]-1, -1, -1):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    return torch.stack(outputs, dim=1) # (batch_size, seq_len, dim_hidden)

  def init_hidden(self, batch_size):
    return torch.zeros(batch_size, self.dim_hidden)

class DeepRNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, num_layers, direction=1):
    super(DeepRNN, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.num_layers = num_layers
    self.direction = direction
    
    self.input_layer = RNNLayer(dim_input, dim_hidden, direction=direction)
    self.rnn_layers = nn.ModuleList([RNNLayer(dim_hidden, dim_hidden, direction=direction) for _ in range(num_layers)])
  
  def forward(self, input):
    hidden = self.input_layer.init_hidden(input.size()[0])
    outputs = self.input_layer(input, hidden)
    for i in range(self.num_layers):
      hidden = self.rnn_layers[i].init_hidden(input.size()[0])
      outputs = self.rnn_layers[i](outputs, hidden)
    return outputs

  def init_hidden(self, batch_size):
    return torch.zeros(batch_size, self.dim_hidden)

class BiDeepRNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
    super(BiRNN, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output
    self.num_layers = num_layers

    self.rnn_layers_forward = DeepRNN(dim_input, dim_hidden, num_layers, direction=1)
    self.rnn_layers_backward = DeepRNN(dim_input, dim_hidden, num_layers, direction=-1)
    self.output_layer = nn.Linear(2*dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, input):
    # Since we are dealing with Char-RNN task so we dont need to use all the output
    fs = self.rnn_layers_forward(input)
    bs = self.rnn_layers_backward(input)
    outputs = torch.cat((fs, torch.flip(bs, dims=(1,))), dim=-1)    
    # outputs : (batch_size, seq_len, 2*dim_hidden)
    outputs = self.output_layer(outputs)
    outputs = self.softmax(outputs)
    return outputs