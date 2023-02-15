import argparse
from typing import List, Dict
from collection import defaultdict, Counter
import copy

class Vocab():
  def __init__(self, max_num_vocab: int):
    self.max_num_vocab = max_num_vocab
    self.word_2_edit = defaultdict()
    self.word_freq = Counter()
    self.vocab = set()
    self.BPE = Counter()

  def n_gram(self, tokens: List[str], n: int) -> List[str]:
    return [a+b for [a, b] in zip(tokens[:-(n-1)], tokens[(n-1):])]

  def clean_line(self, text: str) -> str:
    """
     input: text
     output: a cleans version, without multiple spaces
    """
    pass

  def merge_edit(self, edit: List[str], target: str) -> List[str]:
    result = []
    last_merged = False
    for idx in range(len(edit)):
      if idx+1 < len(edit):
        if (edit[idx] + edit[idx+1]) == target:
          result.append(target)
          if idx+1 == len(edit)-1:
            last_merged = True
        else:
          result.append(edit[idx])
      elif not last_merged:
        result.append(edit[idx])
    return result

  def add_line(self, line: str):
    for word in line.split(" "):
      if not word in self.word_2_edit.keys():
        self.word_2_edit[word] = list(word)
      self.word_freq[word] += 1
    self.vocab.update(set(list(line))) # list of characters
  
  def build_vocab(self):
    # Initialize with character first
    while len(self.vocab) <= max_num_vocab:
      for word, edit in word_2_edit.items():
        grams = n_gram(tokens=edit, n=2)
        for gram in grams:
          self.BPE[gram] += self.word_freq[word]
      next_merge = self.BPE.most_common(1)
      for word, edit in word_2_edit.items():
        if next_merge in word:
          self.word_2_edit[word] = self.merge_edit(edit, target=next_merge)
      self.vocab.add(next_merge)

  def tokenize(self, text: str) -> List(str):
    result = []
    for word in text.split(" "):
      result.extend(word_2_edit[word])
  

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
  
  print(sh_vocab.tokenize(args.input))