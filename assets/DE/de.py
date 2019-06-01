import numpy as np
from scipy.optimize import rosen, differential_evolution

def ackley(x):
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

def rastrigin(x):
    A = 10
    n = 2
    return A*n + sum([x[i]**2 - A * np.cos(2 * np.pi * x[i]) for i in range(len(x))])

def quadraticEq(parameters, *data):

  a,b,c = parameters
  x,y = data
  result = 0
  for i in range(len(x)):
      result += (a*x[i]**2 + b*x[i] + c - y[i])**2

  return result**0.5

def prot01():
  bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
  result = differential_evolution(rosen, bounds)
  print('DE applied to minimise the Rosenbrock function: x = {0}, f(x) = {1}'.format(result.x, result.fun))

def prot02():
  bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
  result = differential_evolution(rosen, bounds, updating='deferred', workers=2)
  print('DE applied to minimise the Rosenbrock function: x = {0}, f(x) = {1}'.format(result.x, result.fun))

def prot03():
  bounds = [(-5, 5), (-5, 5)]
  result = differential_evolution(ackley, bounds)
  print('DE applied to minimise the Ackley function: x = {0}, f(x) = {1}'.format(result.x, result.fun))

def prot04():
  bounds = [(-5, 5), (-5, 5)]
  result = differential_evolution(ackley, bounds, updating='deferred', workers=2)
  print('DE applied to minimise the Ackley function: x = {0}, f(x) = {1}'.format(result.x, result.fun))

def prot05():
  bounds = [(-5, 5), (-5, 5)]
  result = differential_evolution(rastrigin, bounds)
  print('DE applied to minimise the Rastrigin function: x = {0}, f(x) = {1}'.format(result.x, result.fun))

def prot06():
  #initial guess for variation of parameters
  #             a            b            c
  bounds = [(1.5, 0.5), (-0.3, 0.3), (0.1, -0.1)]

  #producing "experimental" data
  x = [i for i in range(6)]
  y = [x**2 for x in x]

  #packing "experimental" data into args
  args = (x,y)

  result = differential_evolution(quadraticEq, bounds, args=args)
  print('DE applied to solve quadratic equation: x = {0}, f(x) = {1}'.format(result.x, result.fun))

def main():
  prot06()

if(__name__ == '__main__'):
  main()
