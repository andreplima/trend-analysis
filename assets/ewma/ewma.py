# this code failed: https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
# data obtained from https://catalog.data.gov/dataset/real-estate-sales-2001-2016 and enriched
# data obtained from

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def ses(train, test, alpha=0.0):

  if(alpha > 0.0):
    model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
    _alpha = '{0:2.1f}'.format(alpha)
  else:
    model = SimpleExpSmoothing(train).fit()
    _alpha = model.model.params['smoothing_level']

  pred  = model.predict(start=test.index[0], end=test.index[-1])

  plt.plot(train.index, train, label='Train')
  plt.plot(test.index, test, label='Test')
  plt.plot(pred.index, pred, label=r'SES, $\alpha={0}$'.format(_alpha))
  plt.legend(loc='best')

def prot01(train, test):

  print("Forecasting sales of properties using SES method.")
  ses(train, test, .2)
  plt.show()

  ses(train, test, .6)
  plt.show()

  ses(train, test)
  plt.show()

def holt(train, test, alpha=0.0, beta=0.0, exponential=False, damped=False):

  if(alpha > 0.0 and beta > 0.0):

    if(exponential or damped):
      model = Holt(train, exponential=exponential, damped=damped).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)
    else:
      model = Holt(train).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)

    _alpha = '{0:2.1f}'.format(alpha)
    _beta  = '{0:2.1f}'.format(beta)

  else:
    if(exponential or damped):
      model = Holt(train, exponential=exponential, damped=damped).fit()
    else:
      model = Holt(train).fit()

    _alpha = model.model.params['smoothing_level']
    _beta  = model.model.params['smoothing_slope']

  pred  = model.predict(start=test.index[0], end=test.index[-1])

  plt.plot(train.index, train, label='Train')
  plt.plot(test.index, test, label='Test')
  plt.plot(pred.index, pred, label=r'Holt, $\alpha={0}$'.format(_alpha))
  plt.legend(loc='best')

def prot02(train, test):

  print("Forecasting sales of properties using Holt method with both additive and exponential trend.")

  holt(train, test, alpha=.8, beta=.2, exponential=False, damped=False)
  plt.show()

  holt(train, test, alpha=.8, beta=.2, exponential=True,  damped=False)
  plt.show()

  holt(train, test, exponential=False, damped=True)
  plt.show()


def hw(train, test, seasonal_periods=4, trend='add', seasonal='add', damped=False, use_boxcox=True):

  model = ExponentialSmoothing(train, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal).fit(use_boxcox)
  pred  = model.predict(start=test.index[0], end=test.index[-1])

  plt.plot(train.index, train, label='Train')
  plt.plot(test.index, test, label='Test')
  plt.plot(pred.index, pred, label='Holt-Winters, p({0})t({1})s({2})$'.format(seasonal_periods, trend, seasonal))
  plt.legend(loc='best')

def prot03(train, test):

  print("Forecasting sales of properties using Holt-Winters method with both additive and multiplicative seasonality.")

  hw(train, test, seasonal_periods=4, trend='add', seasonal='add', damped=False, use_boxcox=True)
  plt.show()

  hw(train, test, seasonal_periods=4, trend='add', seasonal='mul', damped=False, use_boxcox=True)
  plt.show()

  hw(train, test, seasonal_periods=4, trend='add', seasonal='add', damped=True, use_boxcox=True)
  plt.show()

  hw(train, test, seasonal_periods=4, trend='add', seasonal='mul', damped=True, use_boxcox=True)
  plt.show()

def prot04(train, test):
  model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12).fit()
  pred = model.predict(start=test.index[0], end=test.index[-1])

  plt.plot(train.index, train, label='Train')
  plt.plot(test.index, test, label='Test')
  plt.plot(pred.index, pred, label='Holt-Winters')
  plt.legend(loc='best')
  plt.show()

def main(datasetid):

  if(datasetid == 'airport'):
    df = pd.read_csv('international-airline-passengers.csv', sep=';', parse_dates=['Month'], index_col='Month')
    #df.index.freq = 'MS'
    #print(df.head())
    #df.plot.line()
    #plt.show()

    train, test = df.iloc[:130, 0], df.iloc[130:, 0]

  else:
    raise ValueError

  prot04(train, test)

if(__name__ == '__main__'):
  main(sys.argv[1])