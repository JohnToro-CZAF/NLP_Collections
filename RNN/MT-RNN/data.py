import argparse 
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter, namedtuple
import torch.nn.functional as F
from typing import List, Dict, Tuple

class EngFranRawDataset(Dataset):
  def __init__(self, filename):
    super(EngFranRawDataset, self).__init__()
    self.filename = filename
    self.corpus = []
    self.vocab = {
      'eng': defaultdict(str, {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}),
      'fra': defaultdict(str, {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3})
    }
    self.freq = {
      'eng': Counter(),
      'fra': Counter()
    }
    self.vocab_size = {
      'eng': 4,
      'fra': 4
    }

  def read_data(self):
    longest_len = 0
    with open(self.filename, 'r', encoding='utf-8') as f:
      for i, _ in enumerate(tqdm(f)):
        EngFra = defaultdict(str)
        pairs = _.split("\t")[:-1]
        for item, lang in zip(pairs, ['eng', 'fra']):
          item = item.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u2009', ' ').lower().strip()
          check = lambda char, prev: char in '.,?/!' and prev != ' '
          sentence = [char if idx > 0 and not check(char, item[idx-1]) else (" " + char) for idx, char in enumerate(item)]
          sentence = [word for word in "".join(sentence).split() if word]
          if lang == "fra":
            sentence = ["<BOS>"] + sentence
          else:
            sentence = sentence + ["<EOS>"]
          EngFra[lang] = sentence
          longest_len = max(longest_len, len(sentence))
        EngFra['fra_label'] = EngFra['fra'][1:] + ["<EOS>"]
        self.corpus.append(EngFra)
      print("Longest sentence is: ", longest_len)

  def build_vocab(self):
    self.vocab_eng = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    self.vocab_fra = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    for PairEngFra in self.corpus:
      for lang in ['eng', 'fra']:
        for word in PairEngFra[lang]:
          self.freq[lang][word] += 1
    
    for lang in ['eng', 'fra']:
      for word in self.freq[lang].keys():
        if self.freq[lang][word] < 2:
          self.vocab[lang][word] = 3 # <UNK>
        else:
          if word in self.vocab[lang]:
            continue
          self.vocab[lang][word] = self.vocab_size[lang]
          self.vocab_size[lang] += 1

  def split_train_val_test(self, ratio=0.9):
    train_size = int(ratio * len(self.corpus))
    val_size = int((len(self.corpus) - train_size) / 2)
    test_size = len(self.corpus) - train_size - val_size
    train, val, test = torch.utils.data.random_split(self.corpus, [train_size, val_size, test_size])
    
    TrainDataset = TranslationDataset(train, self.vocab, self.vocab_size)
    ValDataset = TranslationDataset(val, self.vocab, self.vocab_size)
    TestDataset = TranslationDataset(test, self.vocab, self.vocab_size)
    return TrainDataset, ValDataset, TestDataset

  def get_tokenizer(self):
    return Tokenizer(self.vocab, self.vocab_size)

class TranslationDataset(Dataset):
  def __init__(self, dataset, vocab, vocab_size):
    super(TranslationDataset, self).__init__()
    self.dataset = dataset
    self.vocab = vocab
    self.vocab_size = vocab_size
  
  def __len__(self):
    return len(self.dataset)
  
  # def word_to_tensor(self, word:str, lang='eng') -> torch.Tensor:
  #   return F.one_hot(torch.tensor(self.vocab[lang][word]), num_classes=self.vocab_size[lang])

  def sent_to_tensor(self, sentence: List[str], lang='eng') -> torch.Tensor:
    # return seq_len, vocab_size
    return torch.tensor([self.vocab[lang][word] for word in sentence])

  def pair_to_tensor(self, pair: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    return [self.sent_to_tensor(pair[lang], lang) for lang in ['eng', 'fra']] + [self.sent_to_tensor(pair['fra_label'], 'fra')]

  def __getitem__(self, idx):
    return self.pair_to_tensor(self.dataset[idx])

class Tokenizer():
  def __init__(self, vocab, vocab_size):
    self.token_2_id = vocab
    self.id_2_token = {
      'eng': defaultdict(),
      'fra': defaultdict()
    }
    self.vocab_size = vocab_size
    self.id_2_token['eng'] = {v: k for k, v in self.token_2_id['eng'].items()}
    self.id_2_token['fra'] = {v: k for k, v in self.token_2_id['fra'].items()}
    self.pad_token_id = self.token_2_id['eng']['<PAD>']
    # print(self.token_2_id['fra']['<BOS>'])
  
  def tokens_2_ids(self, sentence: List[str], lang='eng') -> torch.Tensor:
    # return seq_len, vocab_size
    return torch.tensor([self.token_2_id[lang][word] for word in sentence])
  
  def ids_2_tokens(self, tensor: torch.Tensor, lang='eng') -> List[str]:
    tensor = tensor.detach().cpu().numpy()
    return [self.id_2_token[lang][id] for id in tensor]

def collate_fn_max_batch(data):
  eng, fra, fra_label = zip(*data)
  eng = torch.nn.utils.rnn.pad_sequence(eng, batch_first=True, padding_value=0)
  fra = torch.nn.utils.rnn.pad_sequence(fra, batch_first=True, padding_value=0)
  fra_label = torch.nn.utils.rnn.pad_sequence(fra_label, batch_first=True, padding_value=0)
  return eng, fra, fra_label

def collate_fn_max_length(data, max_length=100):
  # pad first seq to desired length
  eng, fra, fra_label = zip(*data)
  eng = list(eng)
  eng[0] = nn.ConstantPad1d((0, max_length - eng[0].shape[0]), 0)(eng[0])
  eng = torch.nn.utils.rnn.pad_sequence(eng, batch_first=True, padding_value=0)
  fra = torch.nn.utils.rnn.pad_sequence(fra, batch_first=True, padding_value=0)
  fra_label = torch.nn.utils.rnn.pad_sequence(fra_label, batch_first=True, padding_value=0)
  return eng, fra, fra_label 

def get_data_loader(filename, num_workers=4, batch_size=4, shuffle=True, global_max_len=None):
  dataset = EngFranRawDataset(filename)
  dataset.read_data()
  dataset.build_vocab()
  train_data, val_data, test_data = dataset.split_train_val_test()
  tokenizer = dataset.get_tokenizer()
  if global_max_len is not None:
    collate_fn = lambda x: collate_fn_max_length(x, global_max_len)
  else:
    collate_fn = collate_fn_max_batch
  
  return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True), \
         DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True), \
         DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True), \
         tokenizer

if __name__ == '__main__':
  a, b, c, t = get_data_loader()
  print(t.tokens_2_ids(['hello', 'world'], 'eng'))
  print(t.ids_2_tokens(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'fra'))

  # testing purpose
  # for i in a:
  #   print(i[0].size(), i[1].shape)
  #   break