import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import get_data_loader
from models.encoder_decoder import Encoder, Decoder, EncoderDecoder
import numpy as np
import argparse

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
    self.criterion = nn.NLLLoss()
    self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
  def train_epoch(self, epoch):
    self.model.train()
    for i, (X, y, y_hat) in enumerate(self.train_data):
      X = X.to(self.device)
      y = y.to(self.device)
      y_hat = y_hat.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.model(X, y)
      mask = (y != self.tokenizer.pad_token_id).float()
      print(outputs.view(-1, outputs.size()[-1]).size(), y_hat.size(), mask.size())
      loss = self.criterion(outputs.view(-1, outputs.size()[-1]), y_hat, reduction='sum')
      loss = (loss * mask.view(-1)).sum() / mask.sum()
      loss.backward()
      self.optimizer.step()
      if i % 100 == 0:
        print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
  
  def val_epoch(self, epoch):
    self.model.eval()
    with torch.no_grad():
      for i, (X, y, y_hat) in enumerate(self.val_data):
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(X, y)
        mask = (y != self.tokenizer.pad_token_id).float()
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), y_hat.view(-1))
        loss = (loss * mask.view(-1)).sum() / mask.sum()
        if i % 100 == 0:
          print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
  
  def test(self):
    self.model.eval()
    with torch.no_grad():
      for i, (X, y, h_hat) in enumerate(self.test_data):
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)
        outputs = self.model(X, y)
        mask = (y != self.tokenizer.pad_token_id).float()
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), y_hat.view(-1))
        loss = (loss * mask.view(-1)).sum() / mask.sum()
        if i % 100 == 0:
          print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
  
  def train(self):
    for epoch in range(args.epochs):
      self.train_epoch(epoch)
      self.val_epoch(epoch)
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
