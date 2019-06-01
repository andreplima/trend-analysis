import numpy as np

from copy                   import copy
from scipy.stats            import norm
from scipy.spatial.distance import euclidean
from pyts.approximation     import SymbolicAggregateApproximation

ECO_NUM_OF_CLASSES = 3

def normalise(_ts):
	# applies differentiation and z-normalisation
  ts = np.array([0] + [_ts[i+1] - _ts[i] for i in range(len(_ts)-1)])
  mu = np.mean(ts)
  sd = np.std(ts, ddof=1)
  return ((ts - mu)/sd, (mu, sd))

def ts3sax(_ts):
	# obtains a SAX representation of the time series
  ts, (mu, sd) = normalise(_ts)
  binmap = {'a': mu - sd * norm.ppf(1/3), 'b': mu, 'c': mu - sd * norm.ppf(2/3)}
  model = SymbolicAggregateApproximation(n_bins=3, strategy='normal')
  sax_ts = model.fit_transform(ts.reshape(1, -1)) # data comprises a single sample
  return(''.join(sax_ts[0]), binmap)

# generates all the subsequences of length [w] from a time series [ts]
def genss(ts, w):
  for i in range(len(ts) - w + 1):
    yield (i, ts[i:i+w])

def CID(a, b):

  def CE(a):
    return sum([(a[i] - a[i+1])**2 for i in range(len(a)-1)]) ** .5

  cf = max(CE(a), CE(b))/min(CE(a), CE(b))
  return cf * euclidean(a,b)

def levenshtein(s, t):

  # degenerate cases
  if s == t:      return 0
  if len(s) == 0: return len(t)
  if len(t) == 0: return len(s)

  # initialize v0 (the previous row of distances)
  # this row is A[0][i]: edit distance for an empty s
  # the distance is just the number of characters to delete from t
  v0 = []
  v1 = []
  for i in range(len(t)+1):
    v0.append(i)
    v1.append(0)

  for i in range(len(s)):
    # calculate v1 (current row distances) from the previous row v0
    # first element of v1 is A[i+1][0]
    # edit distance is delete (i+1) chars from s to match empty t
    v1[0] = i + 1

    # use formula to fill in the rest of the row
    for j in range(len(t)):
      cost = 0 if s[i] == t[j] else 1
      v1[j + 1] = min(v1[j]+1, v0[j+1]+1, v0[j]+cost)

    # copy v1 (current row) to v0 (previous row) for next iteration
    for j in range(len(t)+1):
      v0[j] = v1[j]

  return v1[len(t)]

def aggregate(L):
  return(np.mean(L))


def knn_tspi(_ts, k, w, fn=normalise, fd=CID, fa=aggregate):
  """
  Parameters:
  _ts - a univariate time series (as a list of values chronologically ordered)
  w   - length of the sliding window
  k   - number of nearest neighbours that will be used to predict the target value
  fn  - a normalising function that will be applied to remove
  fd  - a distance function that will be used to compare two arbitrary subsequences of length [w] extracted from [ts]
  fa  - aggregating function

  Please, consider having a look at this article:
  Parmezan, A. R. S., & Batista, G. E. (2015, December). A study of the use of complexity measures in the similarity search process adopted
  by knn algorithm for time series prediction. In 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA)
  (pp. 45-51). IEEE.
  [https://bdpi.usp.br/bitstream/handle/BDPI/50010/2749829.pdf;jsessionid=4B273341218463337CD653EF2B283F25?sequence=1]

  """

  # differentiates and normalises the time series, if required
  ts = np.array(_ts)
  if(fn != None): ts, _ = fn(ts)

  # defines the query Q subsequence
  Q = ts[-w:]

  # creates a subsequence generator
  S = genss(ts[:-w], w)

  # computes the distance between the query and each subsequence
  D = [(pos, fd(Q,ss)) for (pos,ss) in S]

  # identifies the k subsequences in S that are the nearest to Q
  P = [pos for (pos, _) in sorted(D, key=lambda e:e[1])][:k]

  # recovers the next value of each subsequence in P to forecast the next value for query Q
  res = fa([_ts[pos+w] for pos in P])

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
  k  = 2

  ref = ts[-1]
  val = knn_tspi(ts[:-1], k, w, fn=None)

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
  prot05()

if(__name__ == '__main__'):
  main()

