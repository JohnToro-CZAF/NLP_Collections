import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rnn import RNN
from models.lstm import LSTM
from data import read_data, split_data, NameDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad, torch.tensor(yy)
class TrainingArgs:
  def __init__(self, learning_rate: float, num_epochs: int):
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs

class Trainer:
  def __init__(self, model=None, training_args=None, train_loader=None, val_loader=None):
    self.args = training_args
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.loss = nn.NLLLoss()
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.learning_rate)

  def eval_step(self, input, label):
    with torch.no_grad():
      hidden = self.model.init_hidden(input.size()[0])
      if not isinstance(hidden, tuple):
        hidden = hidden.to(input.device)
      else:
        hidden = [h.to(input.device) for h in hidden]
      for i in range(input.size()[1]):
        output, hidden = self.model(input[:,i,:], hidden)
    
    loss = self.loss(output, label)
    return output, loss

  def acc_in_batch(self, pred, label):
    return torch.sum(torch.argmax(pred, dim=1) == label).item()

  def eval(self, current_epoch):
    val_acc = []
    val_loss = []
    for input, label in self.val_loader:
      output, loss = self.eval_step(input, label)
      batch_acc = self.acc_in_batch(output, label)

      val_acc.append(batch_acc/input.size()[0])
      val_loss.append(loss.item()/input.size()[0])
    
    print("Curren acc on val is: ", np.mean(val_acc))
    print("Current loss on val is: ", np.mean(val_loss))
  
  def train_step(self, input, label):
    hidden = self.model.init_hidden(batch_size=input.size()[0])
    if not isinstance(hidden, tuple):
      hidden = hidden.to(input.device)
    else:
      hidden = [h.to(input.device) for h in hidden]
    # print(input)
    self.optimizer.zero_grad()
    # print(input[:, 1, :])
    for i in range(input.size()[1]):
      output, hidden = self.model(input[:, i, :], hidden)
    loss = self.loss(output, label)
    loss.backward()
    self.optimizer.step()
    return output, loss.item()

  def train(self):
    self.model.train()
    for epoch in range(self.args.num_epochs):
      train_acc = []
      train_loss_arr = []
      train_loss = 0
      # self.optimizer.zero_grad()

      for input, label in self.train_loader:
        # print(input.size(), label.size())
        output, loss = self.train_step(input, label)
        batch_acc = self.acc_in_batch(output, label)
        # print(output.size(), label.size())
        # print(batch_acc)
        # train_loss += loss
        train_loss_arr.append(loss/input.size()[0])
        train_acc.append(batch_acc/input.size()[0])
        train_loss += loss
      
      # train_loss.backward()
      # self.optimizer.step()
      print("-"*10 + str(epoch) + "-"*10)    
      print("Current loss on train is: ", np.mean(train_loss_arr))
      print("Curren acc on train is: ", np.mean(train_acc))
      self.eval(epoch)

def main():
  data_pth = "./data/names/*.txt"
  names, tags, all_letters, all_categories = read_data(data_pth)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  data = split_data(names, tags, random_seed=1)
  training_args = TrainingArgs(
    learning_rate=0.001,
    num_epochs=300
  )
  
  data_train = NameDataset(
    names=data['train']['names'], 
    tags=data['train']['tags'], 
    all_letters=all_letters, 
    all_categories=all_categories,
    device=device
  )

  data_val = NameDataset(
    names=data['val']['names'], 
    tags=data['val']['tags'], 
    all_letters=all_letters, 
    all_categories=all_categories,
    device=device
  )

  train_loader = DataLoader(data_train, batch_size=8, shuffle=True, collate_fn=pad_collate)
  val_loader = DataLoader(data_val, batch_size=8, shuffle=True, collate_fn=pad_collate)

  # model = RNN(len(all_letters), 32, len(all_categories), batch_size=32)
  model = LSTM(len(all_letters), 128, len(all_categories), batch_size=8)
  model.to(device)
  trainer = Trainer(
    model=model, 
    training_args=training_args, 
    train_loader=train_loader,
    val_loader=val_loader
  )

  trainer.train()

if __name__ == "__main__":
  main()