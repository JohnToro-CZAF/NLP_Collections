# Must convert the data to ASCII form first
import glob
import os
import string
from typing import List, Dict
import torch as torch
import torch.nn as nn

data_pth = "./data/names/*.txt"
# Get all the the countries category
all_letters = string.ascii_letters + ",.;'"
n_letters = len(all_letters)

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

all_categories = []
def read_data(data_pth: str):
  names = []
  tags = []
  for fn in find_paths(data_pth):
    category = os.path.splitext(os.path.basename(fn))[0]
    all_categories.append(category)
    # Read each file
    lines = read_file(fn)
    names.extend([unicodeToAscii(line) for line in lines])
    tags.extend([category for line in lines])
  return names, tags

from sklearn.model_selection import train_test_split
def split_data(names: List, tags: List, random_seed: int):
  X_train, y_train, X_test, y_test = train_test_split(names, tags, test_size=0.1,random_state=random_seed)
  X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)
  data = {
    'train': {
      'names': X_train,
      'tags': y_train
    }, 
    'val': {
      'names': X_val,
      'tags': y_val
    },
    'test': {
      'names': X_test,
      'tags': y_test
    }
  }
  return data

from torch.nn import functional as F
class Dataloader:
  def __init__(self, names, tags):
    self.names = names
    self.tags = tags

  def name_2_tensor(self, name: str) -> torch.Tensor:
    tmp_one_hot = F.one_hot(torch.tensor([all_letters.index(char) for char in name]), num_classes=n_letters)
    return torch.unsqueeze(tmp_one_hot, -2)

  def tag_2_tensor(self, tag: str) -> torch.Tensor:
    tmp_one_hot = F.one_hot(torch.tensor([all_categories.index(tag)]), num_classes=len(all_categories))
    return torch.unsqueeze(tmp_one_hot, 0)

  def __len__(self):
    return len(names)

  def __getitem__(self, index):
    name = self.names[index]
    tag = self.tags[index]
    return self.name_2_tensor(name), self.tag_2_tensor(tag)

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


class TrainingArgs:
  def __init__(self, learning_rate: float, num_batch: int):
    self.learning_rate = learning_rate
    self.num_batch = num_batch


class Trainer:
  def __init__(self, model=None, training_args=None, data_loader=None):
    self.args = training_args
    self.model = model
    self.data_loader = data_loader
    self.loss = nn.NLLLoss()
  
  def train_step(self, input, label):
    hidden = self.model.init_hidden()
    self.model.zero_grad()
    
    for i in range(input.size()[0]):
      output, hidden = self.model(input[i], hidden)
    
    loss = self.loss(output, label)
    loss.backward()

    for p in self.model.parameters():
      p.data.add_(p.grad.data, alpha=-self.args.learning_rate)
    
    return output, loss.item()

  def train(self):
    for batch in range(self.args.num_batch):
      current_loss = 0
      for input, label in self.data_loader:
        output, loss = self.train_step(input, label)
        current_loss += loss
      print(current_loss)
      

      