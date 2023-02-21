import argparse 
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter, namedtuple
import torch.nn.functional as F
from typing import List, Dict, Tuple

class EngFranDataset(Dataset):
  def __init__(self, filename, device='cuda:0'):
    super(EngFranDataset, self).__init__()
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
    EngFra = namedtuple('EngFra', 'eng fra')
    with open(self.filename, 'r') as f:
      for i, _ in enumerate(tqdm(f)):
        pairs = _.split("\t")[:-1]
        for item, lang in zip(pairs, ['eng', 'fra']):
          check = lambda char, prev: char in '.,?/!' and prev != ' '
          sentence = [char if idx > 0 and not check(char, item[idx-1]) else (" " + char) for idx, char in enumerate(item)]
          sentence = "".join(sentence)
          sentence = sentence.lower().strip()
          if lang == "fra":
            sentence = "<BOS> " + sentence 
          sentence = sentence + " <EOS>"
          pairs[pairs.index(item)] = sentence
        
        PairEngFra = EngFra(*pairs)
        self.corpus.append(PairEngFra)

  def build_vocab(self):
    self.vocab_eng = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    self.vocab_fra = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    for PairEngFra in self.corpus:
      for lang in ['eng', 'fra']:
        for word in PairEngFra._asdict()[lang].split():
          if word not in self.vocab[lang]:
            self.vocab[lang][word] = self.vocab_size[lang]
            self.vocab_size[lang] += 1
          self.freq[lang][word] += 1
    
    for lang in ['eng', 'fra']:
      for word in self.vocab[lang]:
        if self.freq[lang][word] < 2:
          self.vocab[lang][word] = 3 # <UNK>
          self.vocab_size[lang] -= 1
    
    print(self.vocab)
    print(self.vocab_size)


  def word_to_tensor(self, word:str, lang='eng') -> torch.Tensor:
    return F.one_hot(torch.tensor(self.vocab[lang][word]), num_classes=self.vocab_size[lang])

  def sent_to_tensor(self, sentence: List[str], lang='eng') -> torch.Tensor:
    # return seq_len, vocab_size
    return torch.stack([self.word_to_tensor(word, lang) for word in sentence.split(" ")], dim=0)

  def __len__(self):
    return len(self.corpus)
  
  def __getitem__(self, idx):
    return [self.sent_to_tensor(sent, lang).to(self.device) for lang, sent in self.corpus[idx]._asdict().items()]

if __name__ == '__main__':
  dataset = EngFranDataset(filename='data/eng-fra.txt')
  dataset.read_data()
  dataset.build_vocab()

