import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rnn import RNN
from models.lstm import LSTM
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
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.learning_rate)
  
  def train_step(self, input, label):
    hidden = self.model.init_hidden()
    self.optimizer.zero_grad()
    
    for i in range(input.size()[0]):
      output, hidden = self.model(input[i], hidden)
    # print(output, label)
    loss = self.loss(output, label) # Since the dimension of label default is with (len_seq, batch_size, all_categories), here batchsize is 0 and we also only take final output layer -> 0
    loss.backward()
    self.optimizer.step()
    
    return output, loss.item()

  def eval_step(self, input, label):
    with torch.no_grad():
      hidden = self.model.init_hidden()
      for i in range(input.size()[0]):
        output, hidden = self.model(input[i], hidden)
    
    loss = self.loss(output, label)
    return output, loss.item()

  def eval(self, current_epoch):
    total_loss = 0
    correct_ans = 0
    for (input, label), (input_raw, label_raw) in self.data_loader_val:
      output, loss = self.eval_step(input, label)
      total_loss += loss
      # pred_category = torch.argmax(output, dim=1)
      top_n, top_i = output.topk(1)
      category_i = top_i[0].item()
      if category_i == self.data_loader_val.all_categories.index(label_raw):
        correct_ans += 1
    
    print("Current loss on val is: ", total_loss/len(self.data_loader_val))
    print("Curren acc on val is: ", correct_ans/len(self.data_loader_val))
  
  def train(self):
    for epoch in range(self.args.num_epochs):
      total_loss = 0
      correct_ans = 0
      for (input, label), (raw_input, raw_label) in self.data_loader_train:
        # print(input, label)
        # print(raw_input, raw_label)
        # print(self.data_loader_train.all_categories)
        output, loss = self.train_step(input, label)
        total_loss += loss
        # pred_category = torch.argmax(output, dim=1)
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()

        if category_i == self.data_loader_train.all_categories.index(raw_label):
          correct_ans += 1  
      print(correct_ans, len(self.data_loader_train))
      print("-"*10 + str(epoch) + "-"*10)    
      print("Current loss on train is: ", total_loss/len(self.data_loader_train))
      print("Curren acc on train is: ", correct_ans/len(self.data_loader_train))
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
  # model = RNN(len(all_letters), 128, len(all_categories))
  model = LSTM(len(all_letters), 128, len(all_categories))
  trainer = Trainer(
    model=model, 
    training_args=training_args, 
    data_loader_train=data_loader_train, 
    data_loader_val=data_loader_val
  )

  trainer.train()

if __name__ == "__main__":
  main()