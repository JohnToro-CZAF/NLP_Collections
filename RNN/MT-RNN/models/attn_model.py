import torch
import torch.nn
import torch.nn.functional as F
from .deep_lstm import DeepLSTM, DeepLSTMSequential

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.lstm = DeepLSTM(input_dim=embedding_size, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)

  def _init_weight(self):
    self.lstm.init_weight(init_weight)
    self.apply(init_weight)

  def forward(self, X):
    """
    inputs:
      # X: batch_size, seq_len
    returns:
      # output: batch_size, seq_len, hidden_size (last layer hiddens)
      # hidden: num_layers, batch_size, hidden_size (only takes the last hidden state for each layer)
      # cell: num_layers, batch_size, hidden_size
    """
    embs = self.embedding(X) # batch_size, seq_len, embedding_size
    embs = F.relu(embs)
    outputs, (hidden, cell) = self.lstm(embs.transpose(0, 1), state)
    #outputs: seq_len, batch_size, hidden_size
    outputs = outputs.permute(1, 0, 2)
    return outputs, (hidden, cell)

class AttnDecoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, max_length):
    super(AttnDecoder, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.max_length = max_length
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.lstm = DeepLSTMSequential(input_dim=embedding_size, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)
    self.attn = nn.Linear(hidden_size * 2, max_length) # Fuse input, hidden -> attention score over span of context
    self.dropout_nn = nn.Dropout(dropout)
    
  def _init_weight(self):
    self.lstm.init_weight(init_weight)
    self.apply(init_weight)
  
  def forward(self, y, context, H_C):
    # here the context is a sequence length of hiddens (of last layers) from encoder
    # context: batch_size, seq_len_encoder, hidden_size
    # y: batch_size, seq_len_decoder
    # H_C: [num_layers, batch_size, hidden_size] * 2
    embs = self.embedding(y) # batch_size, seq_len_decoder, embedding_size
    embs = self.dropout_nn(embs)
    pass

class EncoderAttnDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    self.encoder = encoder
    self.decoder = decoder 
    self.encoder._init_weight()
    self.decoder._init_weight()
  
  def forward(self, X, y):
    
    pass