import torch
import torch.nn as nn
import torch.nn.functional as F
from .deep_lstm import DeepLSTM, DeepLSTMSequential

def init_weight(module: nn.Module):
  if type(module) == nn.Linear:
    nn.init.xavier_uniform_(module.weight)
class AttnEncoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
    super(AttnEncoder, self).__init__()
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
    outputs, (hidden, cell) = self.lstm(embs.transpose(0, 1))
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
    self.max_length = max_length # max_length to attent on the context (encoded input)
    
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.lstm = DeepLSTMSequential(input_dim=embedding_size, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)
    self.attn = nn.Linear(hidden_size + embedding_size, max_length) # Fuse input, hidden -> attention score over span of context
    self.attn_combined = nn.Linear(embedding_size+hidden_size, embedding_size)
    self.dropout_nn = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax(dim=-1)
    
  def _init_weight(self):
    self.lstm.init_weight(init_weight)
    self.apply(init_weight)
  
  def init_hidden(self, batch_size):
    return self.lstm.init_hidden(batch_size)
  
  def forward(self, y, context, H_C):
    """
    summary:
      For teaching forcing, input entire the target sequence to decoder lstm
    inputs:
      # y: batch_size, decode_seq_len
      # context: batch_size, input_seq_len, hidden_size (last layer hidden, not really use, if we dont use attention)
      # H_C: [num_layers, batch_size, hidden_size]*2 (each layer is initialized with last hidden in context)
    returns:
      # attn_scores: batch_size, max_length (input_seq_len)
      # outputs: batch_size, decode_seq_len, vocab_size
      # hidden: num_layers, batch_size, hidden_size
      # cell: num_layers, batch_size, hidden_size
    """
    embs = self.embedding(y) # batch_size, seq_len_decoder, embedding_size
    embs = self.dropout_nn(embs)
    H, C = H_C
    attn_history = []
    outputs = []
    for input in embs.transpose(0, 1): # batch_size, embedding_size
      # TODO: Currenlty the attention is only applied to one layer only: how to scale to multiple layers?
      attn_scores = self.softmax(self.attn(torch.concat((H[0], input), -1)))
      # batch_size, max_len -> (batch_size, 1, max_len) * (batch_size, max_len, hidden_size)
      attn_vector = torch.bmm(attn_scores.unsqueeze(1), context)
      # new_input is fused version of input and attened vector
      attn_vector = attn_vector.squeeze(1) # batch_size, hidden_size
      new_input = self.attn_combined(torch.cat((attn_vector, input), -1))
      output, (H, C) = self.lstm(new_input, (H, C))
      output = self.softmax(self.fc(output))
      outputs.append(output)
      attn_history.append(attn_scores)
    return torch.stack(outputs, dim=0).transpose(0, 1), torch.stack(attn_history), (H, C)

class EncoderAttnDecoder(nn.Module):
  """
  # Note: with this type of model, we have to make sure that the input to the encoder, has to be equal to encoder_max_length,
  # since we fixed the length of the input, for self.attention
  """
  def __init__(self, encoder, decoder):
    super(EncoderAttnDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder 
    self.encoder._init_weight()
    self.decoder._init_weight()
  
  def forward(self, X, y):
    """
    summary:
      For teaching forcing, input entire the target sequence to decoder lstm
    inputs:
      # X: batch_size, input_seq_len
      # y: batch_size, decode_seq_len
    returns:
      # outputs: batch_size, decode_seq_len, vocab_size (output distribution)
    """
    encoded_input, _ = self.encoder(X) # batch_size, input_seq_len, hidden_size
    context = encoded_input[:, -1, :] # batch_size, hidden_size (take the last hidden in input sequence)
    _, C = self.decoder.init_hidden(X.size()[0])
    H = context.repeat(self.decoder.num_layers, 1, 1) # num_layers, batch_size, hidden_size (initialize decoder with context in all layers)
    outputs, _, _ = self.decoder(y, encoded_input, (H, C))
    return outputs # (batch_size, seq_len, vocab_size)
  
  def predict_step(self, X, y, mode='train', decode_max_length=None):
    """
    summary:
      Training and testing by input only the <BOS> token, and then put output of last timestep as the next input
    inputs:
      # X: batch_size, input_seq_len
      # y: batch_size, decode_seq_len
      # max_length (during evaluate process, decoder generate as many as tokens as it likes, only until it meets the <EOS> token or max_length)
    returns:
      # outputs: batch_size, decode_seq_len|decode_max_length, vocab_size (output distribution)
      # attn_scores: batch_size, decode_seq_len|decode_max_length, input_max_length
      # pred_tokens: batch_size, decode_deq_len|decode_max_length
    """

    if mode == 'train':
      # During teaching forcing, the output sentence must have a similar length to the label
      num_steps = y.size()[1]
    else:
      num_steps = decode_max_length
    
    encoded_input, _ = self.encoder(X) # batch_size, seq_len, hidden_size
    context = encoded_input[:, -1, :] # batch_size, hidden_size (take the last hidden in input sequence)
    pred_tokens = y[:, 0].unsqueeze(1) # batch_size, 1 (Take the first token of y <BOS>)
    outputs = []
    _, C = self.decoder.init_hidden(X.size()[0])
    H = context.repeat(self.decoder.num_layers,1,1)
    attn_scores = []
    for i in range(num_steps):
      distribution, attn_history, (H, C) = self.decoder(pred_tokens[:,-1].unsqueeze(1), encoded_input, (H, C)) # Take the last step of output
      # distribution: batch_size, 1, vocab_size
      pred_token = torch.argmax(distribution, dim=-1) # batch_size, 1
      pred_tokens = torch.cat((pred_tokens, pred_token), dim=-1)
      outputs.append(distribution.squeeze(1)) # -> batch_size, vocab_soze
      attn_scores.append(attn_history)
    # outputs: num_steps, batch_size, vocab_size
    return torch.stack(outputs, dim=0).transpose(0,1), torch.cat(attn_scores, dim=0).transpose(0,1), pred_tokens
  
if __name__  == "__main__":
  encoder = AttnEncoder(vocab_size=5, embedding_size=8, hidden_size=10, num_layers=2, dropout=0.5)
  decoder = AttnDecoder(vocab_size=6, embedding_size=8, hidden_size=10, num_layers=1, dropout=0.5, max_length=7)
  model = EncoderAttnDecoder(encoder, decoder)
  X = torch.randint(0, 4, (4, 7))
  y = torch.randint(0, 5, (4, 20))
  # outputs: batch_size, seq_len, vocab_size
  outputs = model(X, y)
  model.predict_step(X, y, mode='test', decode_max_length=15)
  print(outputs.size())