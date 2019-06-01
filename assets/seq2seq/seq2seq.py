import numpy as np

from copy                   import copy
from scipy.stats            import norm
from scipy.spatial.distance import euclidean
from pyts.approximation     import SymbolicAggregateApproximation
from seq2seq_lstm           import Seq2SeqLSTM

ECO_NUM_OF_CLASSES = 3

def normalise(_ts):

  # obtains a mapping to allow for converting SAX label to numeric representation
  mu = np.mean(_ts)
  sd = np.std(_ts, ddof=1)
  binmap = {'a': mu + norm.ppf(1/3) * sd, 'b': mu, 'c': mu + norm.ppf(2/3) * sd}

  # applies differentiation and z-normalisation
  ts = np.array([0] + [_ts[i] - _ts[i-1] for i in range(1,len(_ts))])
  mu = np.mean(ts)
  sd = np.std(ts, ddof=1)

  return ((ts - mu)/sd, binmap)

def ts3sax(_ts):
  # obtains a SAX representation of the time series
  ts, binmap = normalise(_ts)
  model = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
  sax_ts = model.fit_transform(ts.reshape(1, -1)) # data comprises a single sample
  return(''.join(sax_ts[0]), binmap)

# generates all the subsequences of length [w] from a time series [ts]
def genss(ts, w):
  for i in range(len(ts) - w + 1):
    yield (i, ts[i:i+w])

def aggregate(L):
  return(np.mean(L))


def seq2seq(_ts, w):
  """
  Parameters:
  _ts - a univariate time series (as a list of values chronologically ordered)
  w   - length of the sliding window
  """

  # differentiates and normalises the time series, if required
  ts, binmap = ts3sax(np.array(_ts))

  # creates the training pairs
  L = [' '.join(ss) + ' .' for (pos, ss) in genss(ts[:-w], w)]
  input_texts_for_training  = []
  target_texts_for_training = []
  for i in range(len(L) - 1):
    input_texts_for_training.append(L[i])
    target_texts_for_training.append(L[i+1])

  # fits the model to the training pairs
  seq2seq = Seq2SeqLSTM(latent_dim=5, validation_split=0.4, epochs=5, lr=1e-3, verbose=True, lowercase=False)
  seq2seq.fit(input_texts_for_training, target_texts_for_training)

  # obtains mapping from query sequence predicted one-day-ahead shifted sequence
  Q = [' '.join(ts[-w:]) + ' .']
  predicted_texts = seq2seq.predict(input_texts_for_training)
  for v in predicted_texts: print(v)

  # converts the result to numerical representation
  lbl = 'c' #lbl = predicted_texts[0][-1]
  res = binmap[lbl]

  return res


def prot01():

  ts = [35.5, 17, 21.5, 32, 41.5, 37, 25.5, 35, 46.5, 48, 44.5, 54, 56.5, 65, 42.5, 53, 72.5, 69, 70.5, 72]
  w  = 3

  for ss in genss(ts, w): print(ss)
  Q = ts[-w:]
  print(Q)

def prot02():

  ts = [35.5, 17, 21.5, 32, 41.5, 37, 25.5, 35, 46.5, 48, 44.5]
  w  = 3

  ref = ts[-1]
  val = seq2seq(ts[:-1], w)

  print('Actual value: {0:4.1f}, Forecast: {1:4.1f}'.format(ref, val))

def prot03():

  ts = [35.5, 17, 21.5, 32, 41.5, 37, 25.5, 35, 46.5, 48, 44.5]
  w  = 3
  k  = 2

  ref = ts[-1]
  val = knn_tspi(ts[:-1], k, w, fn=normalise)

  print('Actual value: {0:4.1f}, Forecast: {1:4.1f}'.format(ref, val))

def prot04():

  ts = [35.5, 17, 21.5, 32, 41.5, 37, 25.5, 35, 46.5, 48, 44.5, 54, 56.5, 65, 42.5, 53, 72.5, 69, 70.5, 72]
  w  = 3
  k  = 3

  ref = ts[-1]
  val = knn_tspi(ts[:-1], k, w, fn=normalise)

  print('Actual value: {0:4.1f}, Forecast: {1:4.1f}'.format(ref, val))

def prot05():

  ts = [35.5, 17, 21.5, 32, 41.5, 37, 53, 72.5, 69, 70.5, 72, 54, 56.5, 65, 42.5, 53, 72.5, 69, 70.5, 72]
  w  = 3
  k  = 1

  ref = ts[-1]
  val = knn_tspi(ts[:-1], k, w, fn=normalise, fd=euclidean)

  print('Actual value: {0:4.1f}, Forecast: {1:4.1f}'.format(ref, val))

def prot06():

  ts = [35.5, 17, 21.5, 32, 41.5, 37, 53, 72.5, 69, 70.5, 72, 54, 56.5, 65, 42.5, 53, 72.5, 69, 70.5, 72]
  w  = 3
  k  = 1

  ref = ts[-1]
  val = knn_tspi(ts[:-1], k, w, fn=ts3sax, fd=levenshtein)

  print('Actual value: {0:4.1f}, Forecast: {1:4.1f}'.format(ref, val))

def main():
  prot02()

if(__name__ == '__main__'):
  main()

