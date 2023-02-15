import argparse
from typing import List, Dict
from collections import defaultdict, Counter
import copy
from tqdm.notebook import trange, tqdm
import string
from time import sleep

class Vocab():
  def __init__(self, max_num_vocab: int):
    self.max_num_vocab = max_num_vocab
    self.word_2_edit = defaultdict()
    self.word_freq = Counter()
    self.vocab = set(['-'])
    self.BPE = Counter()

  def n_gram(self, tokens: List[str], n: int) -> List[str]:
    return [x for x in zip(tokens[:-(n-1)], tokens[(n-1):])]

  def clean_line(self, text: str) -> str:
    """
     input: text
     output: a cleans version, without multiple spaces
    """
    return text.lower().strip().translate(str.maketrans('','', string.punctuation)).replace('\n', '')

  def merge_edit(self, edit: List[str], target: str) -> List[str]:
    result = []
    last_merged = False
    idx = 0
    while idx < len(edit):
      if idx+1 < len(edit):
        if (edit[idx] + edit[idx+1]) == target:
          result.append(target)
          if idx+1 == len(edit)-1:
            last_merged = True
          idx += 2
        else:
          result.append(edit[idx])
          idx += 1
      elif not last_merged:
        result.append(edit[idx])
        idx += 1
    return result

  def add_line(self, line: str):
    line = self.clean_line(line)
    for word in line.split(" "):
      if not word in self.word_2_edit.keys():
        self.word_2_edit[word] = list(word) + list('_')
      self.word_freq[word] += 1
    self.vocab.update(set(list(line))) # list of characters

  def build_vocab(self):
    # Initialize with character first
    while len(self.vocab) <= self.max_num_vocab:
      for word, edit in self.word_2_edit.items():
        grams = self.n_gram(tokens=edit, n=2)
        for gram in grams:
          self.BPE[gram[0] + gram[1]] += self.word_freq[word]
          self.BPE[gram[0]] -= self.word_freq[word]
          self.BPE[gram[1]] -= self.word_freq[word]
      next_merge = self.BPE.most_common(1)[0][0]

      # print(next_merge)
      for word, edit in self.word_2_edit.items():
        if next_merge in word + '_':
          self.word_2_edit[word] = self.merge_edit(edit, target=next_merge)
      self.vocab.add(next_merge)

  def tokenize(self, text: str) -> List[str]:
    result = []
    # print(self.word_2_edit)
    for word in text.split(" "):
      result.extend(self.word_2_edit[word])
    return result
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus", type=str, required=True)
  parser.add_argument("--input", type=str, required=True)
  parser.add_argument("--max-num-vocab", type=int, required=True)
  args = parser.parse_args()

  sh_vocab = Vocab(max_num_vocab=args.max_num_vocab)

  with open(args.corpus, "r") as f:
    lines = f.readlines()
  for line in lines:
    sh_vocab.add_line(line)
  
  sh_vocab.build_vocab()
  print(sh_vocab.tokenize(args.input.lower()))