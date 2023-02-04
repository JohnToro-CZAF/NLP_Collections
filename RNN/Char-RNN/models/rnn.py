import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output):
    super(RNN, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    output = self.softmax(output)
    return output, hidden

  def init_hidden(self):
    return Variable(torch.zeros((1, self.dim_hidden)))