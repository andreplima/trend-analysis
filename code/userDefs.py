import sys
import pickle
import codecs
import os
import os.path
import re
import itertools
import multiprocessing
import time
import timeit

import numpy as np
import scipy as sp
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from math            import log, log10
from random          import seed, sample, shuffle, randint, random
from itertools       import permutations, chain, combinations, cycle

from copy            import copy
from datetime        import datetime
from collections     import OrderedDict, namedtuple, defaultdict
from configparser    import RawConfigParser

if(sys.argv[0] == 'forecast.py'):
  from scipy.stats                 import norm
  from pmdarima                    import auto_arima
  from statsmodels.tsa.holtwinters import ExponentialSmoothing
  from scipy.spatial.distance      import euclidean
  from pyts.approximation          import SymbolicAggregateApproximation
  from seq2seq_lstm                import Seq2SeqLSTM

if(sys.argv[0] == 'optimise.py'):
  from scipy.stats                 import norm
  from sklearn.metrics             import confusion_matrix
  from scipy.optimize              import differential_evolution
  if('PARAM_ADJINFLAT' in os.environ and os.environ['PARAM_ADJINFLAT'] == 'True'):
    from cpi                       import inflate
  else:
    def inflate(v, past, to=None): return -1 #raise ValueError

if(sys.argv[0] == 'measure.py'):
  from scipy                       import interp
  from sklearn.metrics             import confusion_matrix, roc_curve, auc
  from sklearn.preprocessing       import label_binarize
  from sklearn.utils.multiclass    import unique_labels

ECO_SEED = 23
ECO_PRECISION = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'

# constants regarding the dataset encoding
ECO_FIELDSEP   = ','
ECO_DATEFORMAT = '%Y-%m-%d'

# constants regarding the specific problem
ECO_TICKER_ENSEMBLE   = 'Ensemble'
ECO_PRICE_UNAVAILABLE = -1
ECO_PRED_UNAVAILABLE  = -1
ECO_NUM_OF_CLASSES = 3
ECO_CLASS_DOWN     = 'Down'
ECO_CLASS_STABLE   = 'Stable'
ECO_CLASS_UP       = 'Up'
ECO_ALL_CLASSES    = [ECO_CLASS_DOWN, ECO_CLASS_STABLE, ECO_CLASS_UP]

TypeConstituent = namedtuple('TypeConstituent', 'name sector first last')
TypeResult      = namedtuple('TypeResult', 'ss accuracy smape')
#--------------------------------------------------------------------------------------------------
# General purpose definitions
#--------------------------------------------------------------------------------------------------

# buffer where all tsprint messages are stored
LogBuffer = []

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def stimediff(finishTs, startTs):
  return str(datetime.strptime(finishTs, ECO_DATETIME_FMT) - datetime.strptime(startTs, ECO_DATETIME_FMT))

def datestr2ts(_date):
  return int(time.mktime(datetime.strptime(_date, ECO_DATEFORMAT).timetuple()))

def ts2datestr(_timestamp):
  return datetime.fromtimestamp(_timestamp).strftime(ECO_DATEFORMAT)

def tsprint(msg, verbose=True):
  buffer = '[{0}] {1}'.format(stimestamp(), msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

def resetLog():
  LogBuffer = []

def saveLog(filename):
  saveAsText('\n'.join(LogBuffer), filename)

def serialise(obj, name):
  f = open(name + '.pkl', 'wb')
  p = pickle.Pickler(f)
  p.fast = True
  p.dump(obj)
  f.close()
  p.clear_memo()

def deserialise(name):
  f = open(name + '.pkl', 'rb')
  p = pickle.Unpickler(f)
  obj = p.load()
  f.close()
  return obj

def file2List(filename, separator = ',', erase = '"', _encoding = 'iso-8859-1'):

  contents = []
  f = codecs.open(filename, 'r', encoding=_encoding)
  if(len(erase) > 0):
    for buffer in f:
      contents.append(buffer.replace(erase, '').rstrip().split(separator))
  else:
    for buffer in f:
      contents.append(buffer.rstrip().split(separator))
  f.close()

  return(contents)

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

def minmax(v, vmin, vmax, limit_condition=0.0):
  if(vmin <= v):
    if(v <= vmax):
      if(vmin < vmax):
        res = (v - vmin)/(vmax - vmin)
      else:

        # this is a little bit tricky; not sure I got it right...
        # the point here was:
        # -- given vmin <= v =< vmax, all real values
        # -- then what is the limit of (v - vmin) / (vmax - vmin) when vmax tends to vmin?
        # --                                                   or when vmin tends to vmax?
        # -- candidate solutions: using matlab, we obtain:
        #    solution 1: piecewise([v == vmin, 0], [v ~= vmin, Inf*(v - vmin)]), when vmax tends to vmin
        #    solution 2: piecewise([v == vmax, 0], [v ~= vmax, Inf]),            when vmin tends to vmax
        # -- as v == vmin whenever this code region is reached, then we convention the result to be zero
        # -- but it is possible to default to another value defined by the limit_condition parameter
        #
        # -- IMPORTANT: in the case of Ssn evaluation, it seems to make more sense to default to 1
        # --            because the surprise of an unknown items should be higher than zero, right?
        #
        # Reproducibility:
        # -- check with the following matlab script below
        # (requires Symbolic Math Toolbox; check its availability with the command 'ver')
        # clear
        # clc
        # format rat
        # syms v vmin vmax
        # assume(v, 'real')
        # assume(vmin, 'real')
        # assume(vmax, 'real')
        # assume(vmin <= vmax)
        # assume(vmin <= v)
        # assume(v <= vmax)
        # limit((v - vmin)/(vmax - vmin), vmax, vmin, 'right')
        # limit((v - vmin)/(vmax - vmin), vmin, vmax, 'left')

        res = limit_condition

    else:
      # v > vmax, clips the result to 1
      res = 1.0
  else:
    # v < vmin, clips the result to 0
    res = 0.0

  return res

def unminmax(v, vmin, vmax, limit_condition = 0.0):
  return vmin + v * (vmax - vmin)

#-------------------------------------------------------------------------------------------------------------------------------------------
# definitions related to parameter files interface
#-------------------------------------------------------------------------------------------------------------------------------------------

# Essay Parameters hashtable
EssayParameters = {}

def setupEssayConfig(configFile = ''):

  # initialises the random number generator
  seed(ECO_SEED)

  # defines default values for some configuration parameters
  setEssayParameter('ESSAY_ESSAYID',  'None')
  setEssayParameter('ESSAY_CONFIGID', 'None')
  setEssayParameter('ESSAY_SCENARIO', 'None')
  setEssayParameter('ESSAY_RUNS',     '1')

  # overrides default values with user-defined configuration
  loadEssayConfig(configFile)

  return listEssayConfig()

def setEssayParameter(param, value):
  """
  Purpose: sets the value of a specific parameter
  Arguments:
  - param: string that identifies the parameter
  - value: its new value
    Premises:
    1) When using inside python code, declare value as string, independently of its true type.
       Example: 'True', '0.32', 'Rastrigin, normalised'
    2) When using parameters in Config files, declare value as if it was a string, but without the enclosing ''.
       Example: True, 0.32, Rastrigin, only Reproduction
  Returns: None
  """

  so_param = param.upper()

  # boolean-valued parameters
  if(so_param in ['PARAM_SAVEIMAGES', 'PARAM_ADJINFLAT', 'PARAM_OPTIMODE']):

    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in ['ESSAY_RUNS', 'PARAM_MAXCORES', 'PARAM_MODELINIT', 'PARAM_TESTPOINTS', 'PARAM_NEW',
                    'PARAM_NEW', 'PARAM_NEW', 'PARAM_NEW']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_NEW', 'PARAM_NEW', 'PARAM_NEW']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH', 'PARAM_TARGETPATH', 'PARAM_LOADPRICES',
                    'PARAM_STOCKLIST', 'PARAM_STOCKEXCP', 'PARAM_PRICES', 'PARAM_MODELS',
                    'PARAM_MODELINIT', 'PARAM_TIMELINE']):

    so_value = value

  # parameters that represent text
  else:

    so_value = value[0]

  EssayParameters[so_param] = so_value

def getEssayParameter(param):
  return EssayParameters[param.upper()]

class OrderedMultisetDict(OrderedDict):

  def __setitem__(self, key, value):

    try:
      item = self.__getitem__(key)
    except KeyError:
      super(OrderedMultisetDict, self).__setitem__(key, value)
      return

    if isinstance(value, list):
      item.extend(value)
    else:
      item.append(value)

    super(OrderedMultisetDict, self).__setitem__(key, item)

def loadEssayConfig(configFile):

  """
  Purpose: loads essay configuration coded in a essay parameters file
  Arguments:
  - configFile: name and path of the configuration file
  Returns: None, but EssayParameters dictionary is updated
  """

  if(len(configFile) > 0):

    if(os.path.exists(configFile)):

      # initialises the config parser and set a custom dictionary in order to allow multiple entries
      # of a same key (example: several instances of GA_ESSAY_ALLELE
      config = RawConfigParser(dict_type = OrderedMultisetDict)
      config.read(configFile)

      # loads parameters codified in the ESSAY section
      for param in config.options('ESSAY'):
        setEssayParameter(param, config.get('ESSAY', param))

      # loads parameters codified in the PROBLEM section
      for param in config.options('PROBLEM'):
        setEssayParameter(param, config.get('PROBLEM', param))

      # expands parameter values that requires evaluation
      # parameters that may occur once, and hold lists or tuples
      if('PARAM_SOURCEPATH' in EssayParameters):
        EssayParameters['PARAM_SOURCEPATH']  = eval(EssayParameters['PARAM_SOURCEPATH'][0])

      if('PARAM_TARGETPATH' in EssayParameters):
        EssayParameters['PARAM_TARGETPATH']  = eval(EssayParameters['PARAM_TARGETPATH'][0])

      if('PARAM_LOADPRICES' in EssayParameters):
        EssayParameters['PARAM_LOADPRICES']  = eval(EssayParameters['PARAM_LOADPRICES'][0])

      if('PARAM_STOCKLIST' in EssayParameters):
        EssayParameters['PARAM_STOCKLIST']  = eval(EssayParameters['PARAM_STOCKLIST'][0])

      if('PARAM_STOCKEXCP' in EssayParameters):
        EssayParameters['PARAM_STOCKEXCP']  = eval(EssayParameters['PARAM_STOCKEXCP'][0])

      if('PARAM_PRICES' in EssayParameters):
        EssayParameters['PARAM_PRICES']  = eval(EssayParameters['PARAM_PRICES'][0])

      if('PARAM_MODELS' in EssayParameters):
        EssayParameters['PARAM_MODELS']  = eval(EssayParameters['PARAM_MODELS'][0])

      if('PARAM_TIMELINE' in EssayParameters):
        EssayParameters['PARAM_TIMELINE']  = eval(EssayParameters['PARAM_TIMELINE'][0])

      # checks if configuration is ok
      (check, errors) = checkEssayConfig(configFile)
      if(not check):
        print(errors)
        exit(1)

    else:

      print('*** Warning: Configuration file [{1}] was not found'.format(configFile))

def checkEssayConfig(configFile):

  check = True
  errors = []
  errorMsg = ""

  # insert criteria below
  if(EssayParameters['ESSAY_ESSAYID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_ESSAYID'].lower() not in configFile.lower()):
    check = False
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the config filename'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'].lower() not in configFile.lower()):
    check = False
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the config filename'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_LOADPRICES' in EssayParameters):
    if(len(EssayParameters['PARAM_LOADPRICES']) == 0):
      check = False
      param_name = 'PARAM_LOADPRICES'
      restriction = 'must contain at least one item'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_LOADPRICES' in EssayParameters):
    if(len(set(EssayParameters['PARAM_LOADPRICES']).difference(['open', 'close', 'high', 'low'])) > 0):
      check = False
      param_name = 'PARAM_LOADPRICES'
      restriction = 'must be composed of open, close, high or low'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_PRICES' in EssayParameters):
    if(len(EssayParameters['PARAM_PRICES']) == 0):
      check = False
      param_name = 'PARAM_PRICES'
      restriction = 'must contain at least one item'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_MODELS' in EssayParameters):
    if(len(EssayParameters['PARAM_MODELS']) == 0):
      check = False
      param_name = 'PARAM_MODELS'
      restriction = 'must contain at least one item'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_MODELS' in EssayParameters):
    ref = ['MI', 'GA', 'MA', 'ARIMA', 'EWMA', 'KNN', 'SAX', 'LSTM']
    val = [ modelType for (modelType, _) in EssayParameters['PARAM_MODELS']]
    if(len(set(val).difference(ref)) > 0):
      check = False
      param_name = 'PARAM_MODELS'
      restriction = 'must be composed of {0}'.format(', '.join(ref))
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_HORIZON' in EssayParameters):
    if(EssayParameters['PARAM_HORIZON'] == 0):
      check = False
      param_name = 'PARAM_HORIZON'
      restriction = 'must be larger than zero'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_TESTPOINTS' in EssayParameters):
    if(EssayParameters['PARAM_TESTPOINTS'] < 3):
      check = False
      param_name = 'PARAM_TESTPOINTS'
      restriction = 'must be larger than 3'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if('PARAM_SAMPLING' in EssayParameters):
    if(EssayParameters['PARAM_SAMPLING'] not in ['linear', 'random', 'heuristic']):
      check = False
      param_name = 'PARAM_SAMPLING'
      restriction = 'must be set to linear, heuristic or random'
      errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  # summarises errors found
  if(len(errors) > 0):
    separator = "=============================================================================================================================\n"
    errorMsg = separator
    for i in range(0, len(errors)):
      errorMsg = errorMsg + errors[i]
    errorMsg = errorMsg + separator

  return(check, errorMsg)

# recovers the current essay configuration
def listEssayConfig():

  res = ''
  for e in sorted(EssayParameters.items()):
    res = res + "{0} : {1} (as {2})\n".format(e[0], e[1], type(e[1]))

  return res

def getPrototypeID():

  if('PARAM_PROTOTYPE' in os.environ):
    res = os.environ['PARAM_PROTOTYPE']
  else:
    res = os.getcwd().split(os.sep)[-2]

  return res

def getMountedOn():

  if('PARAM_MOUNTEDON' in os.environ):
    res = os.environ['PARAM_MOUNTEDON'] + os.sep
  else:
    res = os.getcwd().split(os.sep)[-0] + os.sep

  return res

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: dataset preprocessing (level 0)
#--------------------------------------------------------------------------------------------------
def loadConstituents(sourcepath, filename, _encoding, separator = ECO_FIELDSEP):

  constituents = {}
  f = codecs.open(os.path.join(*sourcepath, filename), 'r', encoding=_encoding)
  for buffer in f:
    try:
      (ticker, _name, _sector) = buffer.rstrip().split(separator)
      constituents[ticker] = TypeConstituent(_name, _sector, None, None)
    except:
      tsprint('-- cannot parse row: {0}'.format(buffer.rstrip()))
  f.close()

  constituents.pop('Symbol')
  tsprint('-- {0} constituent profiles were loaded.'.format(len(constituents)))

  return constituents

def loadStocks(sourcepath, filename, _encoding, _loadprices, separator = ECO_FIELDSEP):

  laststock = None
  attr2pos = {'open': 0, 'high': 1, 'low': 2, 'close': 3}
  stocks = {}
  f = codecs.open(os.path.join(*sourcepath, filename), 'r', encoding=_encoding)
  for buffer in f:
    try:
      (_date, _open, _high, _low, _close, _volume, ticker) = buffer.rstrip().split(separator)
      _timestamp = datestr2ts(_date)
      temp = {}
      for price in _loadprices:
        temp[price] = float([_open, _high, _low, _close][attr2pos[price]])
      stocks[(ticker,_timestamp)] = temp

    except:
      tsprint('-- cannot parse row: {0}'.format(buffer.rstrip()))

  f.close()
  tsprint('-- {0} daily records were loaded.'.format(len(stocks)))

  return stocks

def buildTimeline(stocks, constituents):

  timeline  = set()
  firstTs   = defaultdict(int)
  lastTs    = defaultdict(int)

  minPrice  = defaultdict(float)
  maxPrice  = defaultdict(float)

  for (ticker, _timestamp) in stocks:
    timeline.add(_timestamp)
    if(firstTs[ticker] == 0 or firstTs[ticker] > _timestamp): firstTs[ticker] = _timestamp
    if( lastTs[ticker] == 0 or  lastTs[ticker] < _timestamp):  lastTs[ticker] = _timestamp

  timeline = sorted(timeline)
  timedict = {timeline[timepos]: timepos for timepos in range(len(timeline))}

  newConstituents = {}
  for ticker in firstTs:

    try:
      newConstituents[ticker] = TypeConstituent(constituents[ticker].name,
                                                 constituents[ticker].sector,
                                                 timedict[firstTs[ticker]],
                                                 timedict[lastTs[ticker]])
    except:
      tsprint('-- cannot process data from: {0}'.format(ticker))

  tsprint('-- {0} business days found in the master timeline.'.format(len(timeline)))
  tsprint('-- {0} constituents have daily records.'.format(len(newConstituents)))

  noQuotes = sorted(set(constituents).difference(list(newConstituents)))
  tsprint('-- {0} constituents have no daily records:'.format(len(noQuotes)))

  for ticker in noQuotes:
    tsprint('   {0}: {1}'.format(ticker, constituents[ticker].name))

  return (timeline, newConstituents)

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions - prediction (Level 1)
#--------------------------------------------------------------------------------------------------
def applyModels(param_stocklist, param_stockexcp, param_prices, param_models, param_modelinit, testPoints, timeline, constituents, stocks):
  """
    Produces an in-sample, 1-day-ahead prediction for each combination of assets, test points, series, and models
  """
  # recovers the length of the minimum timeline window required (rp - recent past) to train a model
  rp = param_modelinit['MT'] # xxx questionable choice, is it not?

  failures = 0
  forecasts = []
  for ticker in param_stocklist:
    if(ticker not in param_stockexcp):
      for timepos in testPoints:
        timestamp = timeline[timepos]
        for priceType in param_prices:
          for modelType, modelOverrideParams in param_models:
            tsprint('-- obtaining estimates for {0} in {1} ({2}) using a(n) {3} model'.format(ticker, ts2datestr(timestamp), timepos, modelType))
            model, modelParams = modelHub(modelType, param_modelinit, modelOverrideParams)

            # this test assures that the current stock has daily records prior to the target prediction date
            # e.g.: it is not possible to forecast the close price for the HPQ asset in Aug 8, 2013
            # because the price series are reported for the interval between Oct 19, 2015 and Feb 7, 2018.
            if(timepos - rp >= constituents[ticker].first and timepos <= constituents[ticker].last):

              realVal = stocks[(ticker, timestamp)][priceType]
              predVal = model(ticker, timepos, modelParams, priceType, timeline, constituents, stocks)
              forecasts.append((ticker, timepos, priceType, modelType, realVal, predVal))

            else:
              failures += 1
              forecasts.append((ticker, timepos, priceType, modelType, ECO_PRICE_UNAVAILABLE, 0))
              tsprint('   cannot forecast price for asset {0} in {1} because the series runs from {2} thru {3} and length of window is {4}.'.format(
                          ticker,
                          ts2datestr(timestamp),
                          ts2datestr(timeline[constituents[ticker].first]),
                          ts2datestr(timeline[constituents[ticker].last]),
                          rp,
                          ))

  return forecasts, failures

def selectTestPoints(timeline, param_testpoints, param_timeline, param_sampling):
# returns timepos, not timestamp

  if(param_sampling == 'linear'):
    delta = len(timeline) / (param_testpoints + 1)
    res = [int(delta * (i + 1)) for i in range(param_testpoints)]

  elif(param_sampling == 'random'):
    res = sorted(sample(range(len(timeline)), param_testpoints))

  elif(param_sampling == 'heuristic'):
    res = [timeline.index(datestr2ts(strdate)) for strdate in param_timeline]

  else:
    raise ValueError

  return res

def getFolderForecast(param_sampling, param_models):

#  res = {'linear': 'L', 'heuristic': 'H', 'random': 'R'}[param_sampling]
#  if(len(param_models) == 1):
#    res = res + param_models[0]
#  else:
#    res = res + 'EN'
#
#  return res

#  return '{0}_{1}'.format(param_sampling.lower(), '+'.join(sorted([modelType for (modelType, _) in param_models])))
  return '{0}'.format(param_sampling.lower())

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: prediction models (level 1)
#--------------------------------------------------------------------------------------------------

def modelHub(modelType, param_modelinit, modelOverrideParams):
  """
  returns a function that implements a particular (meta-)model
  """
  if(modelType == 'MA'):
    model = model_ma
  elif(modelType == 'ARIMA'):
    model = model_arima
  elif(modelType == 'EWMA'):
    model = model_hw
  elif(modelType == 'KNN'):
    model = model_knn
  elif(modelType == 'SAX'):
    model = model_sax
  elif(modelType == 'LSTM'):
    model = model_lstm
  elif(modelType == 'MI'):
    model = model_mirror
  elif(modelType == 'GA'):
    model = model_ga
  else:
    raise ValueError

  # overrides the default model parameters (param_modelinit) with specific values set in the config file
  modelParams = param_modelinit
  if(modelOverrideParams != None):
    for param in modelOverrideParams:
      modelParams[param] = modelOverrideParams[param]

  return(model, modelParams)

def model_ma(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
  """
  produces a simple moving average estimate (1-day-ahead, no lag)
  all models in this section MUST allow for these parameters:
  - ticker: the stock symbol
  - timepos: index of the timeline referring to the business day to which a prediction must be produced
  - param_modelinit: a dictionary with the initial values of the model parameters (from config)
  - priceType: the series to which the prediction belongs (open, close, high, or low)
  - timeline: a list that abstracts the period covered by the dataset; created in the preprocessing step
  - constituents: a dictionary with the assets covered by the dataset; created in the preprocessing step
  - stocks: a dictionary with the prices of the assets in the dataset; created in the preprocessing step

  parameters recovered from model initialisation dictionary:
  st - the length of the "short term" range
  """
	
  st = param_modelinit['ST']
  segment = [stocks[(ticker, timeline[timepos - j - 1])][priceType] for j in range(st)]
  return np.mean(segment)

def model_arima(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
  """
  produces an estimate using an optimised ARIMA model (1-day-ahead, no lag)
  please consider having a look at [https://www.alkaline-ml.com/pmdarima/index.html]

  parameters recovered from model initialisation dictionary:
  _p0, _p1 - xxx
  _q0, _q1 - xxx
  _d0      - xxx
  _m0      - xxx
  _P0      - xxx
  _D       - xxx
  _d       - xxx
  _season  - xxx
  """

  # recovers model parameters
  _p0 = param_modelinit['p0']
  _q0 = param_modelinit['q0']
  _p1 = param_modelinit['p1']
  _q1 = param_modelinit['q1']
  _d0 = param_modelinit['d0']
  _m0 = param_modelinit['m0']

  _P0 = param_modelinit['P0']
  _d  = param_modelinit['d']
  _D  = param_modelinit['D']
  _season = param_modelinit['season']

  segment = [stocks[(ticker, timeline[_timepos])][priceType] for _timepos in range(max(constituents[ticker].first, timepos-int(1*_m0)), timepos)]

  # fits the model with historical data (prior to the 'timepos') and produces an estimate (for 'timepos')
  try:
    modelobj = auto_arima(segment, start_p=_p0, start_q=_q0, max_p=_p1, max_q=_q1, m=_m0,
                          start_P=_P0, seasonal=_season, d=_d, D=_D,
                          solver='nm',
                          error_action='ignore',  # don't want to know if an order does not work
                          stepwise=True,          # set to stepwise
                          suppress_warnings=False, trace=False, disp=False)

    estimate = modelobj.predict()

  except Exception as e:
    estimate = [ECO_PRED_UNAVAILABLE]
    tsprint('   cannot perform prediction for asset {0} in {1} because the model failed. {2}'.format(
                ticker,
                ts2datestr(timeline[timepos]),
                str(e),
                ))

  return estimate[0]

def model_hw(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
  """
  produces an estimate using an optimised Holt-Winters model (1-day-ahead, no lag)
  please consider having a look at [https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1]

  parameters recovered from model initialisation dictionary:
  _seasonType - xxx
  _m0         - xxx
  """

  # recovers model parameters
  _seasonType = param_modelinit['seasonType']
  _m0 = param_modelinit['Em0']

  segment = [stocks[(ticker, timeline[_timepos])][priceType] for _timepos in range(constituents[ticker].first, timepos)]
  segment = np.array(segment)

  # fits the model with historical data (prior to the 'timepos') and produces an estimate (for 'timepos')
  modelobj = ExponentialSmoothing(segment, seasonal=_seasonType, seasonal_periods=_m0).fit()
  try:
    estimate = modelobj.predict()

  except Exception as e:
    estimate = [ECO_PRED_UNAVAILABLE]
    tsprint('   cannot perform prediction for asset {0} in {1} because the model failed. {2}'.format(
                ticker,
                ts2datestr(timeline[timepos]),
                str(e),
                ))

  return estimate[0]

def model_knn(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
  """
  produces an estimate using the knn-TSPi algorithm (1-day-ahead, no lag)
  please, consider having a look at this article:
  Parmezan, A. R. S., & Batista, G. E. (2015, December). A study of the use of complexity measures in the similarity search process adopted
  by knn algorithm for time series prediction. In 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA)
  (pp. 45-51). IEEE.
  [https://bdpi.usp.br/bitstream/handle/BDPI/50010/2749829.pdf;jsessionid=4B273341218463337CD653EF2B283F25?sequence=1]

  parameters recovered from model initialisation dictionary:
  w  - length of the sliding window
  k  - number of nearest neighbours that will be used to predict the target value
  rp - the length of the "relevant past" range
  """

  # recovers model parameters
  w  = param_modelinit['w']
  k  = param_modelinit['k']
  rp = param_modelinit['MT']

  #segment = [stocks[(ticker, timeline[_timepos])][priceType] for _timepos in range(constituents[ticker].first, timepos)]
  segment = [stocks[(ticker, timeline[timepos - j - 1])][priceType] for j in range(rp)]
  segment = np.array(segment)

  # fits the model with historical data (prior to the 'timepos') and produces an estimate (for 'timepos')
  (fn, fd, fa) = (normalise, CID, aggregate)

  try:

    ts, _ = fn(segment)   # differentiates and normalises the time series
    Q = ts[-w:]           # defines the query Q subsequence
    S = genss(ts[:-w], w) # creates a subsequence generator as [(position <as int>, subsequence <as np.array>), ...]

    # computes the distance between the query and each of the subsequence
    D = [(pos, fd(Q,ss)) for (pos,ss) in S]

    # identifies the k subsequences in S that are the nearest to Q
    P = [pos for (pos, _) in sorted(D, key=lambda e:e[1])][:k]

    # recovers the next value of each subsequence in P and use them to forecast the next value for query Q
    res = fa([segment[pos+w] for pos in P])

  except Exception as e:
    res = ECO_PRED_UNAVAILABLE
    tsprint('   cannot perform prediction for asset {0} in {1} because the model failed. {2}'.format(
                ticker,
                ts2datestr(timeline[timepos]),
                str(e),
                ))

  return res

def model_sax(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
  """
  produces an estimate using a knn-TSPi algorithm modified to work with SAX representations (1-day-ahead, no lag)

  parameters recovered from model initialisation dictionary:
  w - length of the sliding window
  k - number of nearest neighbours that will be used to predict the target value
  """

  # recovers model parameters
  w = param_modelinit['wsax']
  k = param_modelinit['k']
  rp = param_modelinit['MT']

  #segment = [stocks[(ticker, timeline[_timepos])][priceType] for _timepos in range(constituents[ticker].first, timepos)]
  segment = [stocks[(ticker, timeline[timepos - j - 1])][priceType] for j in range(rp)]
  segment = np.array(segment)

  # fits the model with historical data (prior to the 'timepos') and produces an estimate (for 'timepos')
  (fn, fd, fa) = (ts2sax, levenshtein, aggregate)

  try:

    ts, _ = fn(segment)   # differentiates and normalises the time series
    Q = ts[-w:]           # defines the query Q subsequence
    S = genss(ts[:-w], w) # creates a subsequence generator as [(position <as int>, subsequence <as np.array>), ...]

    # computes the distance between the query and each of the subsequence
    D = [(pos, fd(Q,ss)) for (pos,ss) in S]

    # identifies the k subsequences in S that are the nearest to Q
    P = [pos for (pos, _) in sorted(D, key=lambda e:e[1])][:k]

    # recovers the next value of each subsequence in P and use them to forecast the next value for query Q
    res = fa([segment[pos+w] for pos in P])

  except Exception as e:
    res = ECO_PRED_UNAVAILABLE
    tsprint('   cannot perform prediction for asset {0} in {1} because the model failed. {2}'.format(
                ticker,
                ts2datestr(timeline[timepos]),
                str(e),
                ))

  return res

def model_lstm(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
  """
  produces an estimate using a knn-TSPi algorithm modified to work with SAX representations (1-day-ahead, no lag)

  parameters recovered from model initialisation dictionary:
  w - length of the sliding window
  k - number of nearest neighbours that will be used to predict the target value
  """

  # recovers model parameters
  w = param_modelinit['wlstm']

  segment = [stocks[(ticker, timeline[_timepos])][priceType] for _timepos in range(constituents[ticker].first, timepos)]
  segment = np.array(segment)

  try:

    # differentiates and normalises the time series, if required
    ts, binmap = ts2sax(segment, applyDiff=False, n_bins=5)

    # creates the training pairs
    input_texts_for_training  = []
    target_texts_for_training = []
    L = [' '.join(ss) + ' .' for (pos, ss) in genss(ts[:-w], w)]
    for i in range(len(L) - 1):
      input_texts_for_training.append(L[i])
      target_texts_for_training.append(L[i+1])

    # fits the model to the training pairs
    seq2seq = Seq2SeqLSTM(latent_dim=15, validation_split=0.1, epochs=30, lr=1e-3, verbose=False, lowercase=False, batch_size=64)
    seq2seq.fit(input_texts_for_training, target_texts_for_training)

    # obtains mapping from query sequence predicted one-day-ahead shifted sequence
    Q = ' '.join(ts[-w:]) + ' .'
    predicted_texts = seq2seq.predict([Q])

    # converts the result to numerical representation
    lbl = predicted_texts[0].strip().replace('.','').replace(' ', '')[-1]
    res = binmap[lbl]

  except Exception as e:
    res = ECO_PRED_UNAVAILABLE
    tsprint('   cannot perform prediction for asset {0} in {1} because the model failed. {2}'.format(
                ticker,
                ts2datestr(timeline[timepos]),
                str(e),
                ))

  return res

def model_mirror(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
# mirror - returns the real value
  return stocks[(ticker, timeline[timepos])][priceType]

def model_ga(ticker, timepos, param_modelinit, priceType, timeline, constituents, stocks):
# general average obtained from prices in the period [first, timepos-1]
  segment = [stocks[(ticker, timeline[_timepos])][priceType] for _timepos in range(constituents[ticker].first, timepos)]
  return np.mean(segment)

def normalise(_ts, applyDiff=True, n_bins=ECO_NUM_OF_CLASSES):

  # obtains a mapping to allow for converting SAX label to numeric representation
  mu = np.mean(_ts)
  sd = np.std(_ts, ddof=1)
  #binmap = {'a': mu + norm.ppf(1/n_bins) * sd, 'b': mu, 'c': mu + norm.ppf((n_bins - 1)/n_bins) * sd}
  binref = np.linspace(0, 1, n_bins + 2)[1:-1]
  binppf = [norm.ppf(binref[i]) for i in range(len(binref))]
  binmap = {chr(97 + i): mu + binppf[i] * sd for i in range(len(binppf))}

  # applies differentiation and z-normalisation
  if(applyDiff):
    ts = np.array([0] + [_ts[i] - _ts[i-1] for i in range(1,len(_ts))])
  else:
    ts = np.array(_ts)
  mu = np.mean(ts)
  sd = np.std(ts, ddof=1)

  return ((ts - mu)/sd, binmap)

def ts2sax(_ts, applyDiff=True, n_bins=ECO_NUM_OF_CLASSES):
  # obtains a SAX representation of the time series
  ts, binmap = normalise(_ts, applyDiff=applyDiff, n_bins=n_bins)
  model = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
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

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: optimise weights (level 2)
#--------------------------------------------------------------------------------------------------

def applyWeights(param_stocklist, param_prices, param_models, param_adjinflat, param_optimode, timeline, forecasts, testPoints):

  # the first price in the list is the target of the classification task
  targetPrice = param_prices[0]

  # reorganises the forecasts list
  realSeries = defaultdict(list)
  predSeries = defaultdict(list)
  all_models = [_modelType for (_modelType, _) in param_models]
  for (ticker, timepos, priceType, modelType, realVal, predVal) in forecasts:
    if(ticker in param_stocklist and modelType in all_models and priceType in param_prices):
      if(priceType == targetPrice):
        realSeries[ticker].append((timepos, realVal)) # will store duplicated entries
      predSeries[ticker].append((timepos, modelType, priceType, predVal))

  # removes duplicated entries in realSeries
  for ticker in realSeries:
    realSeries[ticker] = list(set(realSeries[ticker]))

  # computes the the limits and initial thresholds for each stock series based on the real values observed at the test points
  thresholds = {ticker: findThresholds(realSeries[ticker], timeline, param_adjinflat) for ticker in realSeries}
  limits     = {ticker: findLimits(    realSeries[ticker], timeline, param_adjinflat) for ticker in realSeries}

  # creates the initial weights for each model and priceType
  weights = {(ticker, modelType, priceType): 1.0 for ticker in realSeries for modelType in all_models for priceType in param_prices}
  ws = sum(weights.values())
  nt = len(thresholds.keys())
  weights = {key: weights[key] / ws * nt for key in weights}

  # computes the aggregated forecasts, classify real and predicted series and organise them in sklearn format
  pairs = classifySeries(thresholds, weights, timeline, testPoints, realSeries, predSeries)

  if(param_optimode == True):
    # uses DE to obtain optimised weights and thresholds
    tsprint('-- optimising thresholds and weights using differential evolution')

    # defines the structural bounds of the chromosome
    bounds, de2twMap, v0 = tw2debv(thresholds, weights, limits, param_models, param_prices)

    # optimises weights and thresholds
    #params = (pairs, v0)
    params = (de2twMap, limits, timeline, testPoints, realSeries, predSeries, v0)
    result = differential_evolution(de_objfn, bounds, args=params, popsize=30, maxiter=100, polish=False, seed=ECO_SEED)
    tsprint('-- residual energy of the solution: {0:7.5f}'.format(result.fun))

    # converts the final solution to its corresponding thresholds and weights
    thresholds, weights = dev2tw(result.x, de2twMap, limits)

    # computes the aggregated forecasts, classify real and predicted series and organise them in sklearn format
    pairs = classifySeries(thresholds, weights, timeline, testPoints, realSeries, predSeries)

  return weights, thresholds, limits, pairs

def tw2debv(thresholds, weights, limits, param_models, param_prices):

  list_tickers = sorted(thresholds.keys())
  list_models  = sorted([modelType for (modelType, _) in param_models])
  list_prices  = sorted([priceType for priceType in param_prices])

  bounds = []
  de2twMap = {}
  x0 = []
  pos = 0
  for ticker in list_tickers:

    (theta_l, theta_h, param_adjinflat, present) = thresholds[ticker]
    (limit_l, limit_h, _, _) = limits[ticker]

    bounds.append((0, 1))
    de2twMap[pos] = ('threshold', (ticker, None, None))
    x0.append(minmax(theta_l, limit_l, limit_h))
    pos += 1

    bounds.append((0, 1))
    de2twMap[pos] = ('threshold', (ticker, param_adjinflat, present))
    x0.append(minmax(theta_h, limit_l, limit_h))
    pos += 1

  for ticker in list_tickers:
    for modelType in list_models:
      for priceType in list_prices:

        bounds.append((0, 1))
        de2twMap[pos] = ('weights', (ticker, modelType, priceType))
        x0.append(weights[(ticker, modelType, priceType)])
        pos += 1

  x0 = np.array(x0)

  return bounds, de2twMap, x0

def dev2tw(v, de2twMap, limits):

  thresholds = {}
  weights    = {}

  state = 'start'
  for pos in range(v.shape[0]):
    (source, params) = de2twMap[pos]

    if(source == 'threshold'):
      (ticker, param_adjinflat, present) = params
      (limit_l, limit_h, _, _) = limits[ticker]

      if(state == 'start'):
        theta_l = unminmax(v[pos], limit_l, limit_h)
        theta_h = None
        state   = 'waiting'

      elif(state == 'waiting'):
        theta_h = unminmax(v[pos], limit_l, limit_h)
        if(theta_h < theta_l): # repairs the solution
          (theta_h, theta_l) = (theta_l, theta_h)
        thresholds[ticker] = (theta_l, theta_h, param_adjinflat, present)
        theta_l = None
        theta_h = None
        state   = 'start'

      else:
        raise ValueError

    elif(source == 'weights'):
      (ticker, modelType, priceType) = params
      weights[(ticker, modelType, priceType)] = v[pos]

    else:
      raise ValueError

  # normalises the weights per ticker
  ws = defaultdict(float)
  for (ticker, modelType, priceType) in weights:
    ws[ticker] += weights[(ticker, modelType, priceType)]
  for (ticker, modelType, priceType) in weights:
    weights[(ticker, modelType, priceType)] /= ws[ticker]

  return thresholds, weights

# dummy version -- pursues a solution that corresponds to v_debug
#def de_objfn(v, *args):
#  (pairs, v_debug) = args
#  return np.linalg.norm(v - v_debug)

# 1st try: global (ensemble level) optimisation destroys local (stock level) performance
#def de_objfn(v, *args):
#
#  # unpacks arguments
#  (de2twMap, limits, timeline, testPoints, realSeries, predSeries, v0) = args
#
#  # converts the current solution to its corresponding thresholds and weights
#  thresholds, weights = dev2tw(v, de2twMap, limits)
#
#  # computes the aggregated forecasts, classify real and predicted series and have them organised in sklearn format
#  pairs = classifySeries(thresholds, weights, timeline, testPoints, realSeries, predSeries)
#  _, all_true, all_pred = computeMetrics(pairs)
#
#  # computes the confusion matrix
#  cm = confusion_matrix(all_true, all_pred)
#
#  # computes the energy of the current solution
#  C = cm * cm.shape[0] / cm.sum()
#  I = np.eye(*C.shape)
#  res = np.linalg.norm(C - I)
#
#  return res
##  return np.linalg.norm(v - v0)

def de_objfn(v, *args):

  # unpacks arguments
  (de2twMap, limits, timeline, testPoints, realSeries, predSeries, v0) = args

  # converts the current solution to its corresponding thresholds and weights
  thresholds, weights = dev2tw(v, de2twMap, limits)

  # computes the aggregated forecasts, classify real and predicted series and have them organised in sklearn format
  pairs = classifySeries(thresholds, weights, timeline, testPoints, realSeries, predSeries)

  res = 0.0
  for ticker in pairs:

    # pairs up real and forecast values for current ticker
    y_true, y_pred = zip(*[(realClass, predClass) for (_, _, _, realClass, predClass) in pairs[ticker]])

    # computes the confusion matrix
    cm = confusion_matrix(y_true, y_true, ECO_ALL_CLASSES)

    # computes the energy of the current solution
    C = cm * cm.shape[0] / cm.sum()
    I = np.eye(*C.shape)
    res += np.linalg.norm(C - I)

  return res
#  return np.linalg.norm(v - v0)

def classifySeries(thresholds, weights, timeline, testPoints, realSeries, predSeries):

  # computes the weighted predictions in predSeries
  newPredSeries = defaultdict(list)
  for ticker in predSeries:
    acc = defaultdict(float)
    for (timepos, modelType, priceType, predVal) in predSeries[ticker]:
      acc[timepos] += predVal * weights[(ticker, modelType, priceType)]
    for timepos in acc:
      newPredSeries[ticker].append((timepos, acc[timepos]))

  # classifies the values obtained in realSeries
  realClasses = {ticker: classify(realSeries[ticker], timeline, thresholds[ticker]) for ticker in realSeries}

  # classifies the values obtained in newPredSeries
  newPredClasses = {ticker: classify(newPredSeries[ticker], timeline, thresholds[ticker]) for ticker in newPredSeries}

  # reorganises data into pairs to reduce impedance to sklearn routines
  f = lambda e: e[0] # order entries by timepos
  for ticker in realSeries:     realSeries[ticker].sort(key=f)
  for ticker in newPredSeries:  newPredSeries[ticker].sort(key=f)
  for ticker in realClasses:    realClasses[ticker].sort(key=f)
  for ticker in newPredClasses: newPredClasses[ticker].sort(key=f)

  pairs = defaultdict(list)
  for ticker in realSeries:
    for i in range(len(testPoints)):
      (timepos_1, realVal)   = realSeries[ticker][i]
      if(realVal != ECO_PRICE_UNAVAILABLE):
        timepos = testPoints[i]
        (timepos_2, predVal)   = newPredSeries[ticker][i]
        if(predVal != ECO_PRED_UNAVAILABLE):
          (timepos_3, realClass) = realClasses[ticker][i]
          (timepos_4, predClass) = newPredClasses[ticker][i]

          if(timepos == timepos_1 == timepos_2 == timepos_3 == timepos_4):
            pairs[ticker].append((timepos, realVal, predVal, realClass, predClass))
          else:
            raise ValueError

  return pairs

def findThresholds(series, timeline, param_adjinflat):

  if(param_adjinflat):
    # adjust values for inflation up to the latest test point in the series
    present = datetime.fromtimestamp(timeline[max([timepos for (timepos, val) in series])])
    segment = sorted([round(inflate(val, datetime.fromtimestamp(timeline[timepos]), to=present),2) for (timepos, val) in series])
  else:
    present = None
    segment = sorted([val for (timepos, val) in series])

  nb = len(segment) // ECO_NUM_OF_CLASSES
  theta_l = max(segment[:nb])
  theta_h = min(segment[-nb:])

  return(theta_l, theta_h, param_adjinflat, present)


def findLimits(series, timeline, param_adjinflat):

  if(param_adjinflat):
    # adjust values for inflation up to the latest test point in the series
    present = datetime.fromtimestamp(timeline[max([timepos for (timepos, val) in series])])
    segment = sorted([round(inflate(val, datetime.fromtimestamp(timeline[timepos]), to=present),2) for (timepos, val) in series])
  else:
    present = None
    segment = sorted([val for (timepos, val) in series])

  limit_l = min(segment)
  limit_h = max(segment)

  return(limit_l, limit_h, param_adjinflat, present)


def classify(series, timeline, thresholds):

  # unpacks data from thresholds
  (theta_l, theta_h, param_adjinflat, present) = thresholds

  # classifies the values obtained in series
  L = []
  for (timepos, val) in series:

    if(param_adjinflat):
      v = inflate(val, datetime.fromtimestamp(timeline[timepos]), to=present)
    else:
      v = val

    if(v <= theta_l):
      res = ECO_CLASS_DOWN
    elif(v >= theta_h):
      res = ECO_CLASS_UP
    else:
      res = ECO_CLASS_STABLE

    L.append((timepos, res))

  return L

def getFolderOptimise(param_sampling, param_models, param_adjinflat, param_optimode):

  predictlbl = getFolderForecast(param_sampling, param_models)
  adjinflat  = {False: 'unadjusted', True: 'adjusted'}[param_adjinflat]
  optmode    = {False: 'averaged',   True: 'optimised'}[param_optimode]
  #modelstr = '+'.join([modelType for (modelType, _) in param_models])
  modelstr = param_models[0][0] if len(param_models) == 1 else 'ensemble'

  return '{0}_{1}_{2}_{3}'.format(predictlbl, adjinflat, optmode, modelstr)

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: assess ensemble (level 3)
#--------------------------------------------------------------------------------------------------

def computeMetrics(pairs):
  all_true = []
  all_pred = []
  results  = {}
  for ticker in pairs:

    # computes the accuracy for a specific ticker
    y_true, y_pred = zip(*[(realClass, predClass) for (timepos, realVal, predVal, realClass, predClass) in pairs[ticker]])
    v_accuracy = computeAccuracy(y_true, y_pred)
    all_true += y_true
    all_pred += y_pred

    # computes the error (SMAPE measure) for a specific ticker
    y_true, y_pred = zip(*[(realVal, predVal) for (timepos, realVal, predVal, realClass, predClass) in pairs[ticker]])
    v_smape = computeSmape(y_true, y_pred)

    results[ticker] = TypeResult(len(y_true), v_accuracy, v_smape)

  ens_ss  = 0
  ens_acc = 0.0
  ens_err = 0.0
  for ticker in results:
    (ss, accuracy, error) = results[ticker]
    ens_ss  += ss
    ens_acc += ss * accuracy
    ens_err += ss * error

  results[ECO_TICKER_ENSEMBLE] = TypeResult(ens_ss, ens_acc/ens_ss, ens_err/ens_ss)

  return results, all_true, all_pred

def computeAccuracy(y_true, y_pred):
  #return accuracy_score(y_true, y_pred)
  return sum([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])/len(y_true) # metric from Vo, Luo and Vuo

def computeSmape(y_true, y_pred):
  return sum([abs(y_true[i] - y_pred[i])/(.5 * (y_true[i] + y_pred[i])) for i in range(len(y_pred))]) / len(y_true)

def plot_confusion_matrix(y_true, y_pred, title, saveit, param_targetpath, filename, cmap=plt.cm.Reds):
  """
    Code adapted from: [https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py]
  """

  # computes the confusion matrix
  cm = confusion_matrix(y_true, y_pred, ECO_ALL_CLASSES)

  # sets up the plotter
  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         xticklabels=ECO_ALL_CLASSES, yticklabels=ECO_ALL_CLASSES,
         title=title,
         ylabel='True',
         xlabel='Predicted')

  # rotates the tick labels and set their alignment
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # loops over data dimensions and creates text annotations
  fmt = 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")

  # plots in screen or saves the image
  if(saveit):
    print('-- saving the figures.')
    plt.savefig(os.path.join(*param_targetpath, filename), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the figure.')
    fig.tight_layout()
    plt.show()
    print('-- figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))

  return cm.shape != (1, 1)

def plot_ROC_curve(y_true, y_pred, title, saveit, param_targetpath, filename, cmap=plt.cm.Reds):
  """
    Code adapted from: [https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html]
    Please consider having a look at [http://gim.unmc.edu/dxtests/ROC1.htm]
  """

  # reorganises the data to represent a multiclass output
  y_test  = label_binarize(y_true, ECO_ALL_CLASSES)
  y_score = label_binarize(y_pred, ECO_ALL_CLASSES)

  # computes the ROC curve and ROC area for each class
  n_classes = len(ECO_ALL_CLASSES)
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # computes the micro-average ROC curve and its area
  #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
  #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # computes the macro-average ROC curve and its area
  # 1. aggregates all false positive rates
  #all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # 2. interpolates all ROC curves at these points
  #mean_tpr = np.zeros_like(all_fpr)
  #for i in range(n_classes):
  #  mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # 3. averages interpolated points and computes the area under the ROC macro-average curve
  #mean_tpr /= n_classes
  #fpr["macro"] = all_fpr
  #tpr["macro"] = mean_tpr
  #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # plots the ROC curve for each class and aggregations (micro and macro)
  lw = 2
  #plt.figure()
  fig, ax = plt.subplots()
  plt.grid(True, color='w', linestyle='solid', linewidth=lw/2)
  plt.gca().patch.set_facecolor('0.95')

  #plt.plot(fpr["micro"], tpr["micro"],
  #         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
  #         color='deeppink', linestyle='-.', linewidth=lw)
  #
  #plt.plot(fpr["macro"], tpr["macro"],
  #         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
  #         color='navy', linestyle='-.', linewidth=lw)

  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
  for i, color in zip(range(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
               label='ROC curve of class {0} (area = {1:0.2f})'.format(ECO_ALL_CLASSES[i], roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(title)
  plt.legend(loc="lower right")

  # plots in screen or saves the image
  if(saveit):
    print('-- saving the figures.')
    plt.savefig(os.path.join(*param_targetpath, filename), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the figure.')
    fig.tight_layout()
    plt.show()
    print('-- figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))

def getPlotDesc(configid, param_sampling, param_models, param_adjinflat, param_optimode):

  group = {'C0':'Test set', 'C1': 'Baseline', 'C2': 'Winter', 'C3': 'Summer'}[configid]
  modelstr = param_models[0][0] if len(param_models) == 1 else 'ensemble'
  adjinflat  = {False: 'unadjusted', True: 'adjusted'}[param_adjinflat]
  optmode    = {False: 'unit',       True: 'optimised'}[param_optimode]

  return '{0} stocks, {1} sampling\n{2}, {3} prices, {4} weights'.format(group, param_sampling, modelstr, adjinflat, optmode)
