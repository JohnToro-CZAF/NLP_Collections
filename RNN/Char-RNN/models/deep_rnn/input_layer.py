import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNNLayer(nn.Moudle):
  def __init__(self, dim_input, dim_hidden):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden

  def forward(self, input):
    pass

  def init_hidden(self, batch_size):
    return Variable(torch.zeros())