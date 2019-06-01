import sys
import numpy as np
import pmdarima as pm
from   pmdarima.datasets import load_wineind

from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
  return datetime.strptime('190'+x, '%Y-%m')

def prot01(series):
  print(series.head())
  series.plot()
  pyplot.show()

def prot02(series):
  autocorrelation_plot(series)
  pyplot.show()

def prot03(series):
  # fit model
  model = ARIMA(series, order=(5,1,0))
  model_fit = model.fit(disp=0)
  print(model_fit.summary())
  # plot residual errors
  residuals = DataFrame(model_fit.resid)
  residuals.plot()
  pyplot.show()
  residuals.plot(kind='kde')
  pyplot.show()
  print(residuals.describe())

def prot04(series):

  X = series.values
  size = int(len(X) * 0.66)
  train, test = X[0:size], X[size:len(X)]
  history = [x for x in train]
  predictions = list()
  for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%.1f, expected=%.1f' % (yhat, obs))
  error = mean_squared_error(test, predictions)
  print('Test MSE: %.3f' % error)
  # plot
  pyplot.plot(test)
  pyplot.plot(predictions, color='red')
  pyplot.show()

def prot05(series):

  # fit stepwise auto-ARIMA
  stepwise_fit = pm.auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                               start_P=0, seasonal=True,
                               d=1, D=1, trace=True,
                               error_action='ignore',  # don't want to know if an order does not work
                               suppress_warnings=True,  # don't want convergence warnings
                               stepwise=True)  # set to stepwise

  print(stepwise_fit.summary())
  print(stepwise_fit.predict(n_periods=1)[0])

def main(datasetid):
  if(datasetid == 'shampoo'):
    series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
  else:
    series = load_wineind().astype(np.float64)

  prot05(series)

if(__name__ == '__main__'):
  main(sys.argv[1])