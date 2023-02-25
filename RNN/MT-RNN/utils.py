from torchtext.data.utils import ngrams_iterator
import numpy as np
from typing import Union, List

def bleu_eval(label_tokens: Union[str, List], pred_tokens: Union[str, List], k:int) -> float:
  """ Calculate bleu score between target sentence and translated sentence

  Args:
      tgt_sent (str): language sentence
      pred_sent (str): translated sentence
      k (int): only consider k-gram

  Returns:
      float: bleu scores
  """
  if isinstance(label_tokens, str):
    label_tokens = label_tokens.split()
  if isinstance(pred_tokens, str):
    pred_tokens = pred_tokens.split()
  
  while label_tokens[-1] in ['<PAD>', '<EOS>']:
    label_tokens.pop()
  
  # pred_tokens normally like this: A B C D '.' <EOS> '.' <EOS> since in the predict_steps, I dont force to stop early when decoder generate the first <EOS>. This is not a problem since, during training, a mask (with label length) is applied on the loss ( generate redundent tokens is not a problem), while during testing, I can remove all the tokens after the first appearance of <EOS>
  # idx = None
  # for i in range(len(pred_tokens)):
  #   if pred_tokens[i] == '<EOS>':
  #     idx = i
  #     break      
  # pred_tokens = pred_tokens[:idx]
  
  pred_tokens = list(filter(lambda a: a not in ['<PAD>', '<EOS>'], pred_tokens))
    
  print(pred_tokens)
  print(label_tokens)
  len_label, len_pred = len(label_tokens), len(pred_tokens)
  if (len_pred == 0):
    len_pred = 1 
  scores = np.exp(min(0,1-len_label/len_pred))
  for n in range(1, k+1):
    matched = 0
    ngrams_pred = list(ngrams_iterator(pred_tokens, n))
    ngrams_label = list(ngrams_iterator(label_tokens, n))
    for gram in ngrams_pred:
      if gram in ngrams_label:
        matched +=1
    if ngrams_pred == 0:
      continue
    scores *= np.power(matched/len(ngrams_pred), np.power(0.5, n))
  return scores

if __name__ == "__main__":
  tgt_sent = "today is not a real day, even tho i liked it a lot"
  pred_sent = "today i liked her but today is not a good day, you know"
  pred_sent1 = "i liked machine learning today but not in a row a month you know"
  print(bleu_eval(tgt_sent, pred_sent, 5))
  print(bleu_eval(tgt_sent, pred_sent1, 5))
  