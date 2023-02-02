import torch
import torch.nn as nn
from model import RNN
from data import read_data, split_data, Dataloader

class TrainingArgs:
  def __init__(self, learning_rate: float, num_epochs: int):
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs

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
    # print(output, label)
    loss = self.loss(output[0], label[0][0]) # Since the dimension of label default is with (len_seq, batch_size, all_categories), here batchsize is 0 and we also only take final output layer -> 0
    loss.backward()

    for p in self.model.parameters():
      p.data.add_(p.grad.data, alpha=-self.args.learning_rate)
    
    return output, loss.item()

  def train(self):
    for batch in range(self.args.num_epochs):
      current_loss = 0
      for input, label in self.data_loader:
        output, loss = self.train_step(input, label)
        current_loss += loss
      print(current_loss)

def main():
  data_pth = "./data/names/*.txt"
  names, tags, all_letters, all_categories = read_data(data_pth)
  data = split_data(names, tags, random_seed=1)
  training_args = TrainingArgs(
    learning_rate=0.001, 
    num_epochs=300
  )
  data_loader = Dataloader(names=data['train']['names'], tags=data['train']['tags'], all_letters=all_letters, all_categories=all_categories)
  model = RNN(len(all_letters), 128, len(all_categories))
  trainer = Trainer(model=model, training_args=training_args, data_loader=data_loader) 
  trainer.train()

if __name__ == "__main__":
  main()