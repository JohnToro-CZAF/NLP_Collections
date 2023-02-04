import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, batch_size):
    super().__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output
    self.batch_size = batch_size

    self.ih2F = nn.Linear(dim_input+dim_hidden, dim_hidden)
    self.ih2I = nn.Linear(dim_input+dim_hidden, dim_hidden)
    self.ih2C = nn.Linear(dim_input+dim_hidden, dim_hidden)
    self.ih2O = nn.Linear(dim_input+dim_hidden, dim_hidden)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.ic2o = nn.Linear(dim_input+dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, input, H_C):
    if H_C is None:
      H = torch.zeros((input.shape[0], dim_hidden)),
      C = torch.zeros((input.shape[0], dim_hidden)) 
    else:
      H, C = H_C
    # input is in the size of (batch_size, dim_input)
    # dim_hidden is in the size of (batch_size, dim_hidden)
    # memory_cell is in the size of (batch_size, dim_hidden)
    # print(input.size(), H.size())
    combined = torch.concat((input, H), dim=1)
    F_gate = self.sigmoid(self.ih2F(combined))
    I_gate = self.sigmoid(self.ih2I(combined))
    C_gate = self.tanh(self.ih2C(combined))
    O_gate = self.sigmoid(self.ih2O(combined))

    C_new = C * F_gate + I_gate * C_gate
    H = self.tanh(C_new) * O_gate
    output = self.ic2o(torch.concat((input, H), dim=1))
    output = self.softmax(output)

    return output, (H, C_new)

  def init_hidden(self, batch_size):
    return (torch.zeros((batch_size, self.dim_hidden)), torch.zeros((batch_size, self.dim_hidden)))