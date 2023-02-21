import argparse 
from tqdm import tqdm
import torch
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

    self.word_2_idx = defaultdict(str)
    self.idx_2_word = defaultdict(int)

  def read_data(self):
    with open(self.filename, 'r', encoding='utf-8') as f:
      for i, _ in enumerate(tqdm(f)):
        PairEngFra = defaultdict(str)
        pairs = _.split("\t")[:-1]
        for item, lang in zip(pairs, ['eng', 'fra']):
          item = item.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u2009', ' ').lower().strip()
          check = lambda char, prev: char in '.,?/!' and prev != ' '
          sentence = [char if idx > 0 and not check(char, item[idx-1]) else (" " + char) for idx, char in enumerate(item)]
          sentence = [word for word in "".join(sentence).split() if word]
          if lang == "fra":
            sentence = ["<BOS>"] + sentence 
          sentence = sentence + ["<EOS>"]
          PairEngFra[lang] = sentence
        
        self.corpus.append(PairEngFra)

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
          self.vocab[lang][word] = self.vocab_size[lang]
          self.vocab_size[lang] += 1

  def word_to_tensor(self, word:str, lang='eng') -> torch.Tensor:
    tmp = torch.tensor([self.vocab[lang][word]])
    return F.one_hot(tmp, num_classes=self.vocab_size[lang])

  def sent_to_tensor(self, sentence: List[str], lang='eng') -> torch.Tensor:
    # return seq_len, vocab_size
    return torch.stack([self.word_to_tensor(word, lang) for word in sentence], dim=0)

  def pair_to_tensor(self, pair: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    return [self.sent_to_tensor(pair[lang], lang) for lang in ['eng', 'fra']]

  def split_train_val_test(self, ratio=0.9):
    train_size = int(ratio * len(self.corpus))
    val_size = int((len(self.corpus) - train_size) / 2)
    test_size = len(self.corpus) - train_size - val_size
    # tqdm for progress bar converting sentence to tensor
    corpus_tensor = []
    with tqdm(total=len(self.corpus), desc="Converting sentence to tensor") as pbar:
      for i, pair in enumerate(self.corpus):
        try:
          corpus_tensor.append(self.pair_to_tensor(pair))
        except:
          print(pair)
        pbar.update(1)
    return torch.utils.data.random_split(corpus_tensor, [train_size, val_size, test_size])

def get_data_loader(num_workers=4, batch_size=4, shuffle=True):
  dataset = EngFranRawDataset(filename='data/eng-fra.txt')
  dataset.read_data()
  dataset.build_vocab()  
  train_data, val_data, test_data = dataset.split_train_val_test()

  return DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers), \
         DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers), \
         DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

if __name__ == '__main__':
  get_data_loader()

