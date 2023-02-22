from torchtext.data.utils import ngrams_iterator
import numpy as np


def bleu_eval(tgt_sent: str, pred_sent: str, k:int) -> float:
  """ Calculate bleu score between target sentence and translated sentence

  Args:
      tgt_sent (str): language sentence
      pred_sent (str): translated sentence
      k (int): only consider k-gram

  Returns:
      float: bleu scores
  """
  label_tokens, pred_tokens = tgt_sent.split(" "), pred_sent.split(" ")
  len_label, len_pred = len(label_tokens), len(pred_tokens)
  scores = np.exp(min(0,1-len_label/len_pred))
  for n in range(1, k+1):
    ngrams_pred = list(ngrams_iterator(pred_tokens, n))
    ngrams_label = list(ngrams_iterator(label_tokens, n))
    for gram in ngrams_pred:
      if gram in ngrams_label:
        matched +=1
    scores *= np.pow(matched/len(ngrams_pred, np.pow(0.5, n)))
  return scores

if __name__ == "__main__":
  tgt_sent = "today is not a real day, even tho i liked it a lot"
  pred_sent = "today i liked her but today is not a good day, you know"
  pred_sent1 = "i liked machine learning today but not in a row a month you know"
  print(bleu_eval(tgt_sent, pred_sent, 5))
  print(bleu_eval(tgt_sent, pred_sent1, 5))
  