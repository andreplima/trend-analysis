import unittest
import random
import userDefs as ud
import numpy as np

from operator import __and__
from functools import reduce
from math import log, log10
from collections import namedtuple
from datetime import datetime

from userDefs import ECO_SEED, ECO_PRECISION, ECO_DATETIME_FMT, ECO_FIELDSEP, ECO_DATEFORMAT, ECO_PRICE_UNAVAILABLE
from userDefs import ECO_PRED_UNAVAILABLE, ECO_NUM_OF_CLASSES, ECO_CLASS_DOWN, ECO_CLASS_STABLE, ECO_CLASS_UP
from userDefs import TypeConstituent, TypeResult

#-------------------------------------------------------------------------------------------------------------------------------------------
# compares expected and actual responses cases
#-------------------------------------------------------------------------------------------------------------------------------------------

class Test_findThresholds(unittest.TestCase):

  def test_condition01(self):

    # test data
    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series   = [(0, 10.0), (1, 11.0), (2, 12.0)]
    param_adjinflat = False
    present  = None

    # expected response
    ref = (10.0, 12.0, param_adjinflat, present)

    # actual response
    val = ud.findThresholds(series, timeline, param_adjinflat)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

  def test_condition02(self):

    # test data
    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series   = [(0, 10.0), (1, 11.0), (2, 12.0)]
    param_adjinflat = True
    present  = datetime.fromtimestamp(timeline[2])

    # expected response
    ref = (10.28, 12.0, param_adjinflat, present)

    # actual response
    val = ud.findThresholds(series, timeline, param_adjinflat)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

class Test_findLimits(unittest.TestCase):

  def test_condition01(self):

    # test data
    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series   = [(0, 10.0), (1, 11.0), (2, 12.0)]
    param_adjinflat = False
    present  = None

    # expected response
    ref = (10.0, 12.0, param_adjinflat, present)

    # actual response
    val = ud.findLimits(series, timeline, param_adjinflat)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

  def test_condition02(self):

    # test data
    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series   = [(0, 10.0), (1, 11.0), (2, 12.0)]
    param_adjinflat = True
    present  = datetime.fromtimestamp(timeline[2])

    # expected response
    ref = (10.28, 12.0, param_adjinflat, present)

    # actual response
    val = ud.findLimits(series, timeline, param_adjinflat)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

class Test_classify(unittest.TestCase):

  def test_condition01(self):

    # test data
    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series   = [(0, 10.0), (1, 11.0), (2, 12.0)]
    param_adjinflat = False
    thresholds = ud.findThresholds(series, timeline, param_adjinflat)

    # expected response
    ref = [(0, 'Down'), (1, 'Stable'), (2, 'Up')]

    # actual response
    val = ud.classify(series, timeline, thresholds)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

  def test_condition02(self):

    # test data
    timeline   = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series     = [(0, 10.0), (1, 11.0), (2, 12.0)]
    predseries = [(0, 9.25), (1, 10.25), (2, 11.25)]
    param_adjinflat = False
    thresholds = ud.findThresholds(series, timeline, param_adjinflat)

    # expected response
    ref = [(0, 'Down'), (1, 'Stable'), (2, 'Stable')]

    # actual response
    val = ud.classify(predseries, timeline, thresholds)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

  def test_condition03(self):

    # test data
    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series   = [(0, 10.0),  (1, 11.0),  (2, 12.0)]
    param_adjinflat = True
    thresholds = ud.findThresholds(series, timeline, param_adjinflat)

    # expected response
    ref = [(0, 'Down'), (1, 'Stable'), (2, 'Up')]

    # actual response
    val = ud.classify(series, timeline, thresholds)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)

  def test_condition04(self):

    # test data
    timeline   = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    series     = [(0, 10.0),  (1, 11.0),  (2, 12.0)]
    predseries = [(0, 9.25), (1, 10.25), (2, 11.25)]
    param_adjinflat = True
    thresholds = ud.findThresholds(series, timeline, param_adjinflat)

    # expected response
    ref = [(0, 'Down'), (1, 'Stable'), (2, 'Stable')]

    # actual response
    val = ud.classify(predseries, timeline, thresholds)

    # compares expected and actual responses
    res = [ref[i] == val[i] for i in range(len(ref))]
    success = reduce(__and__, res)
    self.assertTrue(success)


class Test_tw2debv(unittest.TestCase):

  def test_condition01(self):

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = False
    present = None

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    # expected response
    ref_de2twMap = {0: ('threshold', ('stock1', None, None)),
                    1: ('threshold', ('stock1', param_adjinflat, present)),
                    2: ('threshold', ('stock2', None, None)),
                    3: ('threshold', ('stock2', param_adjinflat, present)),
                    4: ('weights', ('stock1', 'MA', 'close')),
                    5: ('weights', ('stock2', 'MA', 'close'))
                      }
    ref_bounds   = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    ref_x0 = np.array([0, 1, 0, 1, 1, 1])

    # actual response
    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # compares expected and actual responses
    success = ref_bounds == bounds and ref_de2twMap == de2twMap and np.allclose(ref_x0, x0)
    self.assertTrue(success)

  def test_condition02(self):

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = False
    present = None

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    # expected response
    ref_de2twMap = {0: ('threshold', ('stock1', None, None)),
                    1: ('threshold', ('stock1', param_adjinflat, present)),
                    2: ('threshold', ('stock2', None, None)),
                    3: ('threshold', ('stock2', param_adjinflat, present)),
                    4: ('weights', ('stock1', 'ARIMA', 'close')),
                    5: ('weights', ('stock1', 'MA', 'close')),
                    6: ('weights', ('stock2', 'ARIMA', 'close')),
                    7: ('weights', ('stock2', 'MA', 'close')),
                      }
    ref_bounds   = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    ref_x0 = np.array([0, 1, 0, 1, .5, .5, .5, .5])

    # actual response
    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # compares expected and actual responses
    success = ref_bounds == bounds and ref_de2twMap == de2twMap and np.allclose(ref_x0, x0)
    self.assertTrue(success)

  def test_condition03(self):

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = False
    present = None

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close', 'open']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    # expected response
    ref_de2twMap = {0: ('threshold', ('stock1', None, None)),
                    1: ('threshold', ('stock1', param_adjinflat, present)),
                    2: ('threshold', ('stock2', None, None)),
                    3: ('threshold', ('stock2', param_adjinflat, present)),
                    4: ('weights', ('stock1', 'ARIMA', 'close')),
                    5: ('weights', ('stock1', 'ARIMA', 'open')),
                    6: ('weights', ('stock1', 'MA', 'close')),
                    7: ('weights', ('stock1', 'MA', 'open')),
                    8: ('weights', ('stock2', 'ARIMA', 'close')),
                    9: ('weights', ('stock2', 'ARIMA', 'open')),
                   10: ('weights', ('stock2', 'MA', 'close')),
                   11: ('weights', ('stock2', 'MA', 'open')),
                      }
    ref_bounds   = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    ref_x0 = np.array([0, 1, 0, 1, .25, .25, .25, .25, .25, .25, .25, .25])

    # actual response
    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # compares expected and actual responses
    success = ref_bounds == bounds and ref_de2twMap == de2twMap and np.allclose(ref_x0, x0)
    self.assertTrue(success)

  def test_condition04(self): # corresponds to condition01 with inflation

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = True
    present = datetime.fromtimestamp(timeline[2])

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    # expected response
    ref_de2twMap = {0: ('threshold', ('stock1', None, None)),
                    1: ('threshold', ('stock1', param_adjinflat, present)),
                    2: ('threshold', ('stock2', None, None)),
                    3: ('threshold', ('stock2', param_adjinflat, present)),
                    4: ('weights', ('stock1', 'MA', 'close')),
                    5: ('weights', ('stock2', 'MA', 'close'))
                      }
    ref_bounds   = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    ref_x0 = np.array([0, 1, 0, 1, 1, 1])

    # actual response
    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # compares expected and actual responses
    success = ref_bounds == bounds and ref_de2twMap == de2twMap and np.allclose(ref_x0, x0)
    self.assertTrue(success)

  def test_condition05(self): # corresponds to condition02 with inflation

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = True
    present = datetime.fromtimestamp(timeline[2])

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    # expected response
    ref_de2twMap = {0: ('threshold', ('stock1', None, None)),
                    1: ('threshold', ('stock1', param_adjinflat, present)),
                    2: ('threshold', ('stock2', None, None)),
                    3: ('threshold', ('stock2', param_adjinflat, present)),
                    4: ('weights', ('stock1', 'ARIMA', 'close')),
                    5: ('weights', ('stock1', 'MA', 'close')),
                    6: ('weights', ('stock2', 'ARIMA', 'close')),
                    7: ('weights', ('stock2', 'MA', 'close')),
                      }
    ref_bounds   = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    ref_x0 = np.array([0, 1, 0, 1, .5, .5, .5, .5])

    # actual response
    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # compares expected and actual responses
    success = ref_bounds == bounds and ref_de2twMap == de2twMap and np.allclose(ref_x0, x0)
    self.assertTrue(success)

  def test_condition06(self): # corresponds to condition02 with inflation

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = True
    present = datetime.fromtimestamp(timeline[2])

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close', 'open']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    # expected response
    ref_de2twMap = {0: ('threshold', ('stock1', None, None)),
                    1: ('threshold', ('stock1', param_adjinflat, present)),
                    2: ('threshold', ('stock2', None, None)),
                    3: ('threshold', ('stock2', param_adjinflat, present)),
                    4: ('weights', ('stock1', 'ARIMA', 'close')),
                    5: ('weights', ('stock1', 'ARIMA', 'open')),
                    6: ('weights', ('stock1', 'MA', 'close')),
                    7: ('weights', ('stock1', 'MA', 'open')),
                    8: ('weights', ('stock2', 'ARIMA', 'close')),
                    9: ('weights', ('stock2', 'ARIMA', 'open')),
                   10: ('weights', ('stock2', 'MA', 'close')),
                   11: ('weights', ('stock2', 'MA', 'open')),
                      }
    ref_bounds   = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    ref_x0 = np.array([0, 1, 0, 1, .25, .25, .25, .25, .25, .25, .25, .25])

    # actual response
    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # compares expected and actual responses
    success = ref_bounds == bounds and ref_de2twMap == de2twMap and np.allclose(ref_x0, x0)
    self.assertTrue(success)

class Test_dev2tw(unittest.TestCase):

  def test_condition01(self):

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = False
    present = None

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # expected response
    ref_de2twMap = (thresholds, weights)

    # actual response
    (_thresholds, _weights) = ud.dev2tw(x0, de2twMap, limits)

    # compares expected and actual responses
    success = thresholds == _thresholds and weights == _weights
    self.assertTrue(success)

  def test_condition02(self):

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = False
    present = None

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # expected response
    ref_de2twMap = (thresholds, weights)

    # actual response
    (_thresholds, _weights) = ud.dev2tw(x0, de2twMap, limits)

    # compares expected and actual responses
    success = thresholds == _thresholds and weights == _weights
    self.assertTrue(success)

  def test_condition03(self):

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = False
    present = None

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close', 'open']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # expected response
    ref_de2twMap = (thresholds, weights)

    # actual response
    (_thresholds, _weights) = ud.dev2tw(x0, de2twMap, limits)

    # compares expected and actual responses
    success = thresholds == _thresholds and weights == _weights
    self.assertTrue(success)

  def test_condition04(self): # corresponds to condition01 with inflation

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = True
    present = datetime.fromtimestamp(timeline[2])

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # expected response
    ref_de2twMap = (thresholds, weights)

    # actual response
    (_thresholds, _weights) = ud.dev2tw(x0, de2twMap, limits)

    # compares expected and actual responses
    success = thresholds == _thresholds and weights == _weights
    self.assertTrue(success)

  def test_condition05(self): # corresponds to condition02 with inflation

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = True
    present = datetime.fromtimestamp(timeline[2])

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # expected response
    ref_de2twMap = (thresholds, weights)

    # actual response
    (_thresholds, _weights) = ud.dev2tw(x0, de2twMap, limits)

    # compares expected and actual responses
    success = thresholds == _thresholds and weights == _weights
    self.assertTrue(success)

  def test_condition06(self): # corresponds to condition02 with inflation

    # test data
    thresholds = {}
    limits     = {}

    timeline = [ud.datestr2ts(strdate) for strdate in ['2014-06-01', '2015-10-01', '2017-06-01']]
    realSeries = {'stock1': [(0, 10.0), (1, 11.0), (2, 12.0)],
                  'stock2': [(0, 10.0), (1, 11.0), (2, 12.0)],
                   }

    param_adjinflat = True
    present = datetime.fromtimestamp(timeline[2])

    thresholds['stock1'] = ud.findThresholds(realSeries['stock1'], timeline, param_adjinflat)
    thresholds['stock2'] = ud.findThresholds(realSeries['stock2'], timeline, param_adjinflat)
    limits['stock1'] = ud.findLimits(realSeries['stock1'], timeline, param_adjinflat)
    limits['stock2'] = ud.findLimits(realSeries['stock2'], timeline, param_adjinflat)

    param_models = [('MA', None), ('ARIMA', None)]
    param_prices = ['close', 'open']

    weights = {(ticker, modelType, priceType): 1.0 for ticker in ['stock2', 'stock1'] for (modelType, _) in param_models for priceType in param_prices}
    ws = sum(weights.values())
    nt = len(realSeries.keys())
    weights = {key: weights[key] / ws * nt for key in weights}

    bounds, de2twMap, x0 = ud.tw2debv(thresholds, weights, limits, param_models, param_prices)

    # expected response
    ref_de2twMap = (thresholds, weights)

    # actual response
    (_thresholds, _weights) = ud.dev2tw(x0, de2twMap, limits)

    # compares expected and actual responses
    success = thresholds == _thresholds and weights == _weights
    self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
