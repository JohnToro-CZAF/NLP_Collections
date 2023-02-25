import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from data import get_data_loader
from models.encoder_decoder import Encoder, Decoder, EncoderDecoder
from models.attn_model import AttnEncoder, AttnDecoder, EncoderAttnDecoder
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
from utils import bleu_eval
import random

tb = SummaryWriter('./runs/run4')
attention = "attn"
# attention = None

class TrainingArgs(object):
  def __init__(self, epochs, lr, weight_decay, teaching_forcing, clip):
    self.epochs = epochs
    self.lr = lr
    self.weight_decay = weight_decay
    self.teaching_forcing = teaching_forcing
    self.clip = clip

class HyperParams(object):
  def __init__(self, epochs, lr, teaching_forcing, clip, weight_decay, batch_size, dropout):
    self.epochs = epochs
    self.lr = lr
    self.teaching_forcing = teaching_forcing
    self.clip = clip
    self.weight_decay = weight_decay
    self.batch_size = batch_size
    self.dropout = dropout

class Trainer(object):
  def __init__(self, model, train, val, test, tokenizer, device, encoder_optimizer, decoder_optimizer, args=None):
    self.model = model
    self.train_data = train
    self.val_data = val
    self.test_data = test
    self.tokenizer = tokenizer
    self.device = device
    self.args = args
    self.criterion = nn.NLLLoss(reduction='sum')
    self.encoder_optimizer = encoder_optimizer
    self.decoder_optimizer = decoder_optimizer
  
  def train_epoch(self, epoch):
    self.model.train()
    running_loss = 0.0
    with tqdm(self.train_data, unit="batch") as tepoch:
      for i, (X, y, y_hat) in enumerate(tepoch):
        tepoch.set_description(f"Epoch {epoch}")
        
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = y_hat.to(self.device)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        if random.random() < self.args.teaching_forcing:
          if attention == "attn":
            outputs, attn_scores, pred_tokens = self.model.predict_step(X, y)
          else:
            outputs, pred_tokens = self.model.predict_step(X, y)
        else:
          outputs = self.model(X, y)

        mask = (y != self.tokenizer.pad_token_id).float()
        loss = self.criterion(outputs.reshape(-1, outputs.size()[-1]), torch.flatten(y_hat))
        loss = (loss * torch.flatten(mask)).sum() / mask.sum()
        running_loss += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), args.clip)
        
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        tepoch.set_postfix(loss=loss.item()/X.size()[0])
        if i % 100 == 99:
          tb.add_scalar("Train Loss", running_loss/100, epoch * len(self.train_data) + i)
          running_loss = 0.0
    
  def val_epoch(self, epoch):
    self.model.eval()
    with torch.no_grad():
      running_loss = 0.0
      with tqdm(self.val_data, unit="batch") as tepoch:
        for i, (X, y, y_hat) in enumerate(tepoch):
          tepoch.set_description(f"Epoch {epoch}")
          
          X = X.to(self.device)
          y = y.to(self.device)
          y_hat = y_hat.to(self.device)
          self.encoder_optimizer.zero_grad()
          self.decoder_optimizer.zero_grad()

          if random.random() < self.args.teaching_forcing:
            if attention == "attn":
              outputs, attn_scores, pred_tokens = self.model.predict_step(X, y)
            else:
              outputs, pred_tokens = self.model.predict_step(X, y)
          else:
            outputs = self.model(X, y)

          mask = (y != self.tokenizer.pad_token_id).float()
          loss = self.criterion(outputs.reshape(-1, outputs.size()[-1]), torch.flatten(y_hat))
          loss = (loss * torch.flatten(mask)).sum() / mask.sum()
          running_loss += loss.item()
          tepoch.set_postfix(loss=loss.item()/X.size()[0])
      
      tb.add_scalar("Val Loss", running_loss/len(self.val_data), epoch)
  
  def test(self, epoch):
    self.model.eval()
    with torch.no_grad():
      with tqdm(self.test_data, unit="batch") as tepoch:
        bleu = 0.0
        for i, (X, y, y_hat) in enumerate(self.test_data):
          tepoch.set_description(f"Epoch {tepoch}")
          score_in_batch = 0.0
          X = X.to(self.device)
          y = y.to(self.device)
          y_hat = y_hat.to(self.device)
          self.encoder_optimizer.zero_grad()
          self.decoder_optimizer.zero_grad()

          if attention == "attn":
            outputs, attn_scores, pred_tokens = self.model.predict_step(X, y, 'test', decode_max_length=25)
          else:
            outputs, pred_tokens = self.model.predict_step(X, y, 'test', decode_max_length=25)
          
          # outputs: batch_size, seq_len, vocab_size
          # pred_tokens: batch_size, seq_len
          for i, (b, label) in enumerate(zip(pred_tokens, y)):
            label_tokens = self.tokenizer.ids_2_tokens(label, lang='fra')
            source_tokens = self.tokenizer.ids_2_tokens(X[i], lang='eng')
            pred_tokens = self.tokenizer.ids_2_tokens(b, lang='fra')
            score_in_batch += bleu_eval(label_tokens[1:], pred_tokens[1:], k=5) # not consider <BOS> token in the begining
          
          score_in_batch /= y.size()[0]
          bleu += score_in_batch
        
        bleu /= len(self.test_data)
        print("BLEU score is:", bleu)
        tb.add_scalar("BLEU Test", bleu, epoch)   
  
  def train(self):
    for epoch in range(args.epochs):
      self.train_epoch(epoch)
      self.val_epoch(epoch)
      # if epoch % 2 == 1:
      self.test(epoch)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='data/eng-fra.txt')
  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--batch_size', type=int, default=300)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--lr', type=float, default=0.0001)
  parser.add_argument('--weight_decay', type=float, default=0.0001)
  parser.add_argument('--dropout', type=float, default=0.2)
  parser.add_argument('--teaching_forcing', type=float, default=0.5)
  parser.add_argument('--clip', type=float, default=1)

  parser.add_argument('--save_path', type=str, default='./saved_models/')
  # parser.add_argument('--load_path', type=str, default='./saved_models/23:01:25:08')
  parser.add_argument('--load_path', type=str, default=None)
  args = parser.parse_args()

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print("Training on device: {}".format(device))
  print("Getting data loaders from dataset")
  
  # train, val, test, tokenizer = \
  #   get_data_loader(batch_size=args.batch_size, 
  #                   num_workers=args.num_workers, 
  #                   filename=args.data_path)

  train, val, test, tokenizer = \
    get_data_loader(batch_size=args.batch_size, 
                    num_workers=args.num_workers, 
                    filename=args.data_path,
                    global_max_len=60)
  
  # encoder = Encoder(vocab_size=train.dataset.vocab_size['eng'],
  #                   embedding_size=256,
  #                   hidden_size=256,
  #                   num_layers=4,
  #                   dropout=args.dropout).to(device)
  encoder = AttnEncoder(vocab_size=train.dataset.vocab_size['eng'],
                    embedding_size=256,
                    hidden_size=256,
                    num_layers=3,
                    dropout=args.dropout).to(device)
  # decoder = Decoder(vocab_size=train.dataset.vocab_size['fra'],
  #                   embedding_size=256,
  #                   hidden_size=256,
  #                   num_layers=2,
  #                   dropout=args.dropout).to(device)
  decoder = AttnDecoder(vocab_size=train.dataset.vocab_size['fra'],
                    embedding_size=256,
                    hidden_size=256,
                    num_layers=1,
                    dropout=args.dropout,
                    max_length=60).to(device)
  
  training_args = TrainingArgs(
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    teaching_forcing=args.teaching_forcing,
    clip=args.clip)
  
  encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  model = EncoderAttnDecoder(encoder, decoder).to(device)
  # model = EncoderDecoder(encoder, decoder).to(device)
  
  if args.load_path is not None:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
  trainer = Trainer(model=model, 
                   train=train, 
                   val=val, 
                   test=test,
                   tokenizer=tokenizer,
                   device=device,
                   encoder_optimizer=encoder_optimizer,
                   decoder_optimizer=decoder_optimizer,
                   args=training_args)
  trainer.train()
  torch.save({
    'model_state_dict': model.state_dict()
  }, args.save_path + datetime.now().strftime("%D:%H:%M:%S")[-11:])