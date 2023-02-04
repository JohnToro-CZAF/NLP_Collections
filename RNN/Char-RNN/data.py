from typing import Dict, List
import torch
import torch.nn.functional as F
import string
import glob
import os
import unicodedata

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters + ".,;:'"
    )

def find_paths(file_name: str) -> str:
  return glob.glob(file_name)

def read_file(file_name: str) -> List:
  with open(file_name) as f:
    return f.read().strip().lower().split("\n")

def read_data(data_pth: str):
  all_letters = string.ascii_letters + ".,':;"
  all_categories = []
  names = []
  tags = []
  for fn in find_paths(data_pth):
    category = os.path.splitext(os.path.basename(fn))[0]
    all_categories.append(category)
    # Read each file
    lines = read_file(fn)
    names.extend([unicodeToAscii(line) for line in lines])
    tags.extend([category for line in lines])
  return names, tags, all_letters, all_categories

from sklearn.model_selection import train_test_split
def split_data(names: List, tags: List, random_seed: int):
  X_train, X_test, y_train, y_test = train_test_split(names, tags, test_size=0.1,random_state=random_seed, shuffle=True)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed, shuffle=True)
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
from torch.utils.data import Dataset
class NameDataset(Dataset):
  def __init__(self, names: List[str], tags: List[str], all_letters: List[str], all_categories: List[str]):
    super(NameDataset, self).__init__()
    self.names = names
    self.tags = tags
    self.all_letters = all_letters
    self.all_categories = all_categories
    self.max_length = max_length

  def name_2_tensor(self, name: str) -> torch.Tensor:
    # Padding '.' to name tensor
    padded_name = [self.all_letters.index(char) for char in name] + (self.max_length - len(name)) * [len(self.all_categories) - 1]
    return F.one_hot(torch.tensor(padded_name), num_classes=len(self.all_letters))

  def tag_2_tensor(self, tag: str) -> torch.Tensor:
    return torch.tensor([self.all_categories.index(tag)]).view(1, ) 

  def __len__(self):
    return len(self.names)

  def __getitem__(self, index):
    name = self.names[index]
    tag = self.tags[index]
    return (self.name_2_tensor(name), self.tag_2_tensor(tag))