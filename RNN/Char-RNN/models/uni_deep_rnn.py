import torch.nn as nn
from .bi_deep_rnn import DeepRNN

class UniDeepRNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
    super(UniDeepRNN, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output
    self.num_layers = num_layers
    
    self.rnn = DeepRNN(dim_input, dim_hidden, num_layers, direction=1)
    self.output_layer = nn.Linear(dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def forward(self, input):
    outputs = self.rnn(input)
    outputs = self.output_layer(outputs)
    outputs = self.softmax(outputs)
    return outputs