import torch
import torch.nn as nn
from models import RNN, LSTM
from data import read_data, split_data, Dataloader

class TrainingArgs:
  def __init__(self, learning_rate: float, num_epochs: int):
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs

class Trainer:
  def __init__(self, model=None, training_args=None, data_loader_train=None, data_loader_val=None):
    self.args = training_args
    self.model = model
    self.data_loader_train = data_loader_train
    self.data_loader_val = data_loader_val
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

  def eval_step(self, input, label):
    hidden = self.model.init_hidden()
    for i in range(input.size())[0]:
      output, hidden = self.model(input[i], hidden)
    
    loss = self.loss(output[0], label[0][0])
    return output, loss

  def eval(self, current_epoch):
    total_loss = 0
    correct_ans = 0
    for input, label in self.data_loader_val:
      output, loss = self.eval_step(input, label)
      total_loss += loss
      pred = F.one_hot([self.all_categories.index(torch.argmax(output, dim=1))], len(self.all_categories))
      if label == pred:
        correct_ans += 1
    print("Current epoch has loss: ", total_loss)
    print("Curren acc on val is: ", correct_ans/len(self.data_loader_val))
  
  def train(self):
    for epoch in range(self.args.num_epochs):
      current_loss = 0
      for input, label in self.data_loader_train:
        output, loss = self.train_step(input, label)
        current_loss += loss
      self.eval(epoch)

def main():
  data_pth = "./data/names/*.txt"
  names, tags, all_letters, all_categories = read_data(data_pth)
  data = split_data(names, tags, random_seed=1)
  training_args = TrainingArgs(
    learning_rate=0.005,
    num_epochs=300
  )
  data_loader_train = Dataloader(names=data['train']['names'], tags=data['train']['tags'], all_letters=all_letters, all_categories=all_categories)
  data_loader_val = Dataloader(names=data['val']['names'], tags=data['val']['tags'], all_letters=all_letters, all_categories=all_categories)
  model = RNN(len(all_letters), 128, len(all_categories))
  
  trainer = Trainer(
    model=model, 
    training_args=training_args, 
    data_loader_train=data_loader_train, 
    data_loader_val=data_loader_val
  )

  trainer.train()

if __name__ == "__main__":
  main()