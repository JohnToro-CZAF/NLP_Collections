import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import get_data_loader
from models.encoder_decoder import Encoder, Decoder, EncoderDecoder
import matplotlib.pyplot as plt
import numpy as np
import argparse

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
x_epoch = []
y_loss = {'train': [], 'val': [], 'test': []}

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax0.plot(x_epoch, y_loss['test'], 'go-', label='test')
    if current_epoch == 0:
        ax0.legend()
    fig.savefig(os.path.join('./lossGraphs', 'train.jpg'))

class TrainingArgs(object):
  def __init__(self, epochs, lr, weight_decay):
    self.epochs = epochs
    self.lr = lr
    self.weight_decay = weight_decay

class Trainer(object):
  def __init__(self, model, train, val, test, tokenizer, device, args=None):
    self.model = model
    self.train_data = train
    self.val_data = val
    self.test_data = test
    self.tokenizer = tokenizer
    self.device = device
    self.args = args
    self.criterion = nn.NLLLoss(reduction='sum')
    self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
  def train_epoch(self, epoch):
    self.model.train()
    running_loss = 0.0
    for i, (X, y, y_hat) in enumerate(self.train_data):
      X = X.to(self.device)
      y = y.to(self.device)
      y_hat = y_hat.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.model(X, y)
      mask = (y != self.tokenizer.pad_token_id).float()
      loss = self.criterion(outputs.reshape(-1, outputs.size()[-1]), torch.flatten(y_hat))
      loss = (loss * torch.flatten(mask)).sum() / mask.sum()
      running_loss += loss.item()
      loss.backward()
      self.optimizer.step()
      print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, i, loss.item()))
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, running_loss / len(self.train_data)))
    y_loss['train'].append(running_loss / len(self.train_data))
  
  def val_epoch(self, epoch):
    self.model.eval()
    with torch.no_grad():
      running_loss = 0.0
      for i, (X, y, y_hat) in enumerate(self.val_data):
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(X, y)
        mask = (y != self.tokenizer.pad_token_id).float()
        loss = self.criterion(outputs.reshape(-1, outputs.size()[-1]), torch.flatten(y_hat))
        loss = (loss * torch.flatten(mask)).sum() / mask.sum()
        running_loss += loss.item()
      y_loss['val'].append(running_loss / len(self.val_data))
  
  def test(self):
    self.model.eval()
    with torch.no_grad():
      running_loss = 0.0
      for i, (X, y, h_hat) in enumerate(self.test_data):
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)
        outputs = self.model(X, y)
        mask = (y != self.tokenizer.pad_token_id).float()
        loss = self.criterion(outputs.reshape(-1, outputs.size()[-1]), torch.flatten(y_hat))
        loss = (loss * torch.flatten(mask)).sum() / mask.sum()
        running_loss += loss.item()
      y_loss['test'].append(running_loss / len(self.test_data))
  
  def train(self):
    for epoch in range(args.epochs):
      self.train_epoch(epoch)
      self.val_epoch(epoch)
      draw_curve(epoch)
    self.test()
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='data/eng-fra.txt')
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--weight_decay', type=float, default=0.0001)
  parser.add_argument('--dropout', type=float, default=0.5)

  parser.add_argument('--save_path', type=str, default='model.pt')
  parser.add_argument('--load_path', type=str, default='model.pt')
  args = parser.parse_args()
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train, val, test, tokenizer = \
    get_data_loader(batch_size=args.batch_size, 
                    num_workers=args.num_workers, 
                    filename=args.data_path)
  
  encoder = Encoder(vocab_size=train.dataset.vocab_size['eng'], \
                    embedding_size=256, \
                    hidden_size=256, \
                    num_layers=2,
                    dropout=args.dropout)
  decoder = Decoder(vocab_size=train.dataset.vocab_size['fra'],\
                    embedding_size=256, \
                    hidden_size=256, \
                    num_layers=2,
                    dropout=args.dropout)

  training_args = TrainingArgs(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay) 
  model = EncoderDecoder(encoder, decoder).to(device)
  trainer = Trainer(model=model, 
                   train=train, 
                   val=val, 
                   test=test, 
                   tokenizer=tokenizer,
                   device=device,
                   args=training_args)
  trainer.train()
