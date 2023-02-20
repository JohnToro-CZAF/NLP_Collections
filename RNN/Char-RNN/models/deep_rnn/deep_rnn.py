import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .input_layer import RNNLayer

class DeepRNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, num_layers, batch_size):
    super(DeepRNN, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output
    self.layers = nn.ModuleList()
    for i in range(num_layers):
      if i == 0:
        self.layers.append(RNNLayer(dim_input, dim_hidden))
      else:
        self.layers.append(RNNLayer(dim_hidden, dim_hidden))
    self.ih2o = nn.Linear(2*dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, input, hidden):
    for i in range(len(self.layers)):
      input, hidden[i] = self.layers[i](input, hidden[i])
    output = self.ih2o(torch.concat((input, hidden[-1]), dim=1))
    output = self.softmax(output)
    return output, hidden
  
  def init_hidden(self, batch_size):
    hidden = []
    for i in range(len(self.layers)):
      hidden.append(Variable(torch.zeros((batch_size, self.dim_hidden))))
    return hidden