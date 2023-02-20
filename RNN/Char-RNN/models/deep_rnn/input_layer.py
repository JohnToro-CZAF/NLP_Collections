import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNNLayer(nn.Module):
  def __init__(self, dim_input, dim_hidden):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_hidden)

  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), dim=1)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    
    return output, hidden