# Must convert the data to ASCII form first
import glob
import os
import string
from typing import List, Dict
import torch.nn as nn

data_pth = "./data/names/*.txt"
# Get all the the countries category
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def find_paths(file_name: str) -> str:
  return glob.glob(file_name)

def read_file(file_name: str) -> List:
  with open(file_name) as f:
    return f.read().strip().split("\n")

def read_data(data_pth: str):
  for fn in find_paths(data_pth):
    category = os.path.splitext(os.path.basename(fn))[0]
    # Read each file
    lines = read_file(fn)
    return [unicodeToAscii(line) for line in lines]
class RNN(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output):
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=1)
  
  def foward(self, input, hidden):
    combined = torch.cat((input, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    output = self.softmax(output)
    return output, hidden

  def init_hidden(self):
    return torch.zeros((1, self.dim_hidden))

import random
class DataCollator():
  def __init__(self, data, batch_size):
    self.train = data['train']
    self.test = data['test']
