import sys
import pickle
import codecs
import os
import os.path

import numpy   as np
import matplotlib.pyplot as plt

from collections import namedtuple
from os.path     import join, exists
from itertools   import combinations
from userDefs    import ECO_TICKER_ENSEMBLE, TypeResult, getMountedOn, getFolderOptimise, tsprint, deserialise, saveAsText

# sets up a qualitative colormap with 10 colors
cmap = list(plt.cm.get_cmap('tab10').colors) + ['black', '#128C87', '#FF9130']

styleErrorDefault   = cmap[7]
styleErrorHighlight = cmap[3]

styleFill1 = cmap[2]
styleFill2 = cmap[1]
styleFill3 = cmap[0]

# sets up graphical element sizes
lw, fs, tsp = 1, 12, .30

TypeSummary = namedtuple('TypeSummary', 'ss accuracy accuracy_lb accuracy_ub highlight')

ECO_OP_EQUIVALENT = 'equivalent to'
ECO_OP_SMALLER    = 'smaller than'
ECO_OP_GREATER    = 'greater than'

def compareConfigs(c1, c2, summary):
  c1_ss, c1_vm, c1_vl, c1_vh = summary[c1]
  c2_ss, c2_vm, c2_vl, c2_vh = summary[c2]

  if(min(c1_vh, c2_vh) - max(c1_vl, c2_vl) > 0):
    op = ECO_OP_EQUIVALENT
  elif(c1_vh < c2_vh):
    op = ECO_OP_SMALLER
  else:
    op = ECO_OP_GREATER

  (configid, model_name, condition) = c1
  c1_name = '{0}, {1}/{2}'.format(configid, model_name, condition)

  (configid, model_name, condition) = c2
  c2_name = '{0}, {1}/{2}'.format(configid, model_name, condition)

  return (c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh)


def getConditionLabel(condition):
  (param_sampling, param_adjinflat, param_optimode) = condition
  return param_sampling[0].upper() + ('A' if param_adjinflat else 'U') + ('O' if param_optimode else 'N')


def gencpH1(all_configs, all_models, all_conditions, summary):
  """
    generates comparison pairs for hypothesis H1
  """

  hypothesis = 'H1'
  configid   = 'C1'
  cp, highlight = [], {}
  for (param_model, _) in all_models + [('Ensemble', None)]:
    model_name = param_model if type(param_model) is str else 'Ensemble'
    for condition in all_conditions:
      condition_lbl = getConditionLabel(condition)

      c1 = (configid, model_name, condition_lbl)
      c2 = (configid, 'Baseline', condition_lbl)
      (c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh) = compareConfigs(c1, c2, summary)

      cp.append('{0}\t{1}\t{2}\t{3}\t{4:3d}\t{5:5.3f}\t{6:5.3f}\t{7:3d}\t{8:5.3f}\t{9:5.3f}'.format(hypothesis, c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh))
      highlight[c1] = (op != ECO_OP_EQUIVALENT)
      highlight[c2] = False

  return cp, highlight


def gencpH2(all_configs, all_models, all_conditions_, summary):

  hypothesis = 'H2'
  all_conditions = [(False, False), (False, True), (True, False), (True, True)]
  cp, highlight = [], {}
  for configid in all_configs:
    for (param_model, _) in all_models + [('Ensemble', None)]:
      model_name = param_model if type(param_model) is str else 'Ensemble'

      for (param_adjinflat, param_optimode) in all_conditions:
        for param_sampling in ['H', 'R']:
          condition1 = (param_sampling, param_adjinflat, param_optimode)
          condition2 = ('L',            param_adjinflat, param_optimode)

          condition1_lbl = getConditionLabel(condition1)
          condition2_lbl = getConditionLabel(condition2)

          c1 = (configid, model_name, condition1_lbl)
          c2 = (configid, model_name, condition2_lbl)
          (c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh) = compareConfigs(c1, c2, summary)

          cp.append('{0}\t{1}\t{2}\t{3}\t{4:3d}\t{5:5.3f}\t{6:5.3f}\t{7:3d}\t{8:5.3f}\t{9:5.3f}'.format(hypothesis, c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh))
          highlight[c1] = (op != ECO_OP_EQUIVALENT)
          highlight[c2] = False

  return cp, highlight


def gencpH3(all_configs, all_models, all_conditions_, summary):

  hypothesis = 'H3'
  all_conditions = [('L', False), ('H', False), ('R', False), ('L', True), ('H', True), ('R', True), ]
  cp, highlight = [], {}
  for configid in all_configs:
    for (param_model, _) in all_models + [('Ensemble', None)]:
      model_name = param_model if type(param_model) is str else 'Ensemble'

      for (param_sampling, param_optimode) in all_conditions:
        condition1 = (param_sampling, True,  param_optimode)
        condition2 = (param_sampling, False, param_optimode)

        condition1_lbl = getConditionLabel(condition1)
        condition2_lbl = getConditionLabel(condition2)

        c1 = (configid, model_name, condition1_lbl)
        c2 = (configid, model_name, condition2_lbl)
        (c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh) = compareConfigs(c1, c2, summary)

        cp.append('{0}\t{1}\t{2}\t{3}\t{4:3d}\t{5:5.3f}\t{6:5.3f}\t{7:3d}\t{8:5.3f}\t{9:5.3f}'.format(hypothesis, c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh))
        highlight[c1] = (op != ECO_OP_EQUIVALENT)
        highlight[c2] = False

  return cp, highlight


def gencpH4(all_configs, all_models, all_conditions_, summary):

  hypothesis = 'H4'
  all_conditions = [('L', False), ('H', False), ('R', False), ('L', True), ('H', True), ('R', True), ]
  cp, highlight = [], {}
  for configid in all_configs:
    for (param_model, _) in all_models + [('Ensemble', None)]:
      model_name = param_model if type(param_model) is str else 'Ensemble'

      for (param_sampling, param_adjinflat) in all_conditions:
        condition1 = (param_sampling, param_adjinflat, True)
        condition2 = (param_sampling, param_adjinflat, False)

        condition1_lbl = getConditionLabel(condition1)
        condition2_lbl = getConditionLabel(condition2)

        c1 = (configid, model_name, condition1_lbl)
        c2 = (configid, model_name, condition2_lbl)
        (c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh) = compareConfigs(c1, c2, summary)

        cp.append('{0}\t{1}\t{2}\t{3}\t{4:3d}\t{5:5.3f}\t{6:5.3f}\t{7:3d}\t{8:5.3f}\t{9:5.3f}'.format(hypothesis, c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh))
        highlight[c1] = (op != ECO_OP_EQUIVALENT)
        highlight[c2] = False

  return cp, highlight


def gencpH5(all_configs, all_models, all_conditions, summary):

  hypothesis = 'H5'
  cp , highlight= [], {}
  configid1 = 'C3'
  configid2 = 'C2'
  for (param_model, _) in all_models + [('Ensemble', None)]:
    model_name = param_model if type(param_model) is str else 'Ensemble'

    for condition in all_conditions:
      condition_lbl = getConditionLabel(condition)

      c1 = (configid1, model_name, condition_lbl)
      c2 = (configid2, model_name, condition_lbl)
      (c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh) = compareConfigs(c1, c2, summary)

      cp.append('{0}\t{1}\t{2}\t{3}\t{4:3d}\t{5:5.3f}\t{6:5.3f}\t{7:3d}\t{8:5.3f}\t{9:5.3f}'.format(hypothesis, c1_name, op, c2_name, c1_ss, c1_vl, c1_vh, c2_ss, c2_vl, c2_vh))
      highlight[c1] = (op != ECO_OP_EQUIVALENT)
      highlight[c2] = False

  return cp, highlight


def gatherResults(all_configs, all_models, all_conditions, base_path):

  summary = {}
  content = ['Group\tModel\tCondition\tss\tacc.m\tacc.lb\tacc.ub']
  for (param_model, _) in all_models + [('Ensemble', None)]:
    model_name = param_model if type(param_model) is str else 'Ensemble'

    for (param_sampling, param_adjinflat, param_optimode) in all_conditions:
      condition = param_sampling[0].upper() + ('A' if param_adjinflat else 'U') + ('O' if param_optimode else 'N')

      for configid in all_configs:
        path = base_path + [configid, getFolderOptimise(param_sampling, [(param_model,_)], param_adjinflat, param_optimode)]

        # recover experiment results for specific model, condition and config
        if(exists(join(*path, 'results.pkl'))):
          results = deserialise(join(*path, 'results'))
          # collect mean accuracy and bootstraped confidence interval
          ss = results[ECO_TICKER_ENSEMBLE].ss
          vm = results[ECO_TICKER_ENSEMBLE].accuracy
          vl = results[ECO_TICKER_ENSEMBLE].accuracy_lb
          vh = results[ECO_TICKER_ENSEMBLE].accuracy_ub
        else:
          ss =  0
          vm = -1
          vl = -1
          vh = -1

        # stores the results

        summary[(configid, model_name, condition)] = (ss, vm, vl, vh)
        content.append('{0}\t{1}\t{2}\t{3:3d}\t{4:5.3f}\t{5:5.3f}\t{6:5.3f}'.format(configid, model_name, condition, ss, vm, vl, vh))

        if(configid == 'C1'):
          #(vm, vl, vh) = (.791, .787, .795)
          (ss, vm, vl, vh) = (0, .793, .761, .825)
          summary[(configid, 'Baseline', condition)] = (ss, vm, vl, vh)
          content.append('{0}\t{1}\t{2}\t{3:3d}\t{4:5.3f}\t{5:5.3f}\t{6:5.3f}'.format(configid, 'Baseline', condition, ss, vm, vl, vh))

  return summary, content


def plotH1(summary, highlight, all_models, params):

  (saveit, param_targetpath, filename) = params

  # sets up scope of the plot for the current hypothesis
  configs = ['C1']

  # initialises subplots
  nrows, ncols = len(configs) * 2, len(all_models) + 1
  figsizew, figsizeh = 17.06, len(configs) * 5.1
  fig, axes = plt.subplots(nrows, ncols, figsize=(figsizew, figsizeh), sharey='all')
  x_min, y_max = .4, 1.4

  # sets up the bottom-up order in which each semi-condition (sampling and inflation schemes) appears in a subplot
  all_conditions = [('R', False), ('R', True), ('H', False), ('H', True), ('L', False), ('L', True)]

  pos = 0
  for configid in configs:
    for param_optimode in [False, True]:
      for (param_model, _) in [('Ensemble', None)] +  all_models:
        model_name = param_model if type(param_model) is str else 'Ensemble'

        # recovers the data for each semicondition presented in a error plot <group, model, (LHR) x (UA)>
        x, x_lb, x_ub = [], [], []
        y, y_lbl, y_clr  = [], [], [] # y-coordinate, label and color for each semi-condition in a error plot
        yi = 1 # index of the semi-condition
        for (param_sampling, param_adjinflat) in all_conditions:
          condition = getConditionLabel((param_sampling, param_adjinflat, param_optimode))

          (ss, vm, vl, vh) = summary[(configid, model_name, condition)]
          x.append(vm)
          x_lb.append(vm - vl)
          x_ub.append(vh - vm)
          y.append(yi * .2)
          y_lbl.append(condition)
          y_clr.append(styleErrorHighlight if highlight[(configid, model_name, condition)] else styleErrorDefault)
          yi += 1

        x     = np.array(x)
        x_lb  = np.array(x_lb)
        x_ub  = np.array(x_ub)
        y     = np.array(y)
        y_lbl = tuple([''] + y_lbl + [''])
        y_clr = tuple(y_clr)
        asymmetric_error = [x_lb, x_ub]

        # draws the subplot
        plt.subplot(nrows, ncols, 1 + pos)
        plt.grid(True, color='w', linestyle='solid', linewidth=lw/2)
        plt.gca().patch.set_facecolor('0.95')
        plt.gca().set_xlim(x_min, 1.0)
        plt.gca().set_ylim(0.0,   y_max)
        plt.subplots_adjust(left=.06, bottom=.06, right=.94, top=.94, wspace=.30, hspace=.25)

        #plt.errorbar(x, y, xerr=asymmetric_error, fmt='o', lw=lw, capsize=5)
        for i in range(len(x)):
          plt.errorbar(x[i], y[i], xerr=[[x_lb[i]],[x_ub[i]]], fmt='o', color=y_clr[i], lw=lw, capsize=5)

        locs, labels = plt.yticks()
        plt.yticks(locs, y_lbl)

        # add details to the subplot

        # 1. draws the baseline range for Group C1 (Hypothesis H1)
        (ss, vm, vl, vh) = summary[(configid, 'Baseline', condition)]
        plt.fill_betweenx((0, 1.4), vl, vh, alpha=tsp, color=styleFill1)

        # 2. sets column titles above the first row
        if(pos < ncols):
          plt.text(0.5, 1.05, model_name, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # 3. sets row titles after the last column
        if(pos % ncols == ncols - 1):
          plt.text(1.15, 0.45, configid, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # sets plot indicator to the next subplot
        pos += 1

  if(saveit):
    print('-- saving the figures.')
    if(not exists(join(*param_targetpath))):
      makedirs(join(*param_targetpath))
    plt.savefig(join(*param_targetpath, filename), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the figure.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))


def plotH2(summary, highlight, all_models, params, configs):

  (saveit, param_targetpath, filename) = params

  # initialises subplots
  nrows, ncols = len(configs) * 2, len(all_models) + 1
  figsizew, figsizeh = 17.06, len(configs) * 5.1
  fig, axes = plt.subplots(nrows, ncols, figsize=(figsizew, figsizeh), sharey='all')
  x_min, y_max = .4, 1.4

  # sets up the bottom-up order in which each semi-condition (sampling and inflation schemes) appears in a subplot
  all_conditions = [('L', False), ('H', False), ('R', False), ('L', True), ('H', True), ('R', True)]

  pos = 0
  for configid in configs:
    for param_optimode in [False, True]:
      for (param_model, _) in [('Ensemble', None)] +  all_models:
        model_name = param_model if type(param_model) is str else 'Ensemble'

        # recovers the data for each semicondition presented in a error plot <group, model, (LHR) x (UA)>
        x, x_lb, x_ub = [], [], []
        y, y_lbl, y_clr  = [], [], [] # y-coordinate, label and color for each semi-condition in a error plot
        yi = 1 # index of the semi-condition
        for (param_sampling, param_adjinflat) in all_conditions:
          condition = getConditionLabel((param_sampling, param_adjinflat, param_optimode))

          (ss, vm, vl, vh) = summary[(configid, model_name, condition)]
          x.append(vm)
          x_lb.append(vm - vl)
          x_ub.append(vh - vm)
          y.append(yi * .2)
          y_lbl.append(condition)
          y_clr.append(styleErrorHighlight if highlight[(configid, model_name, condition)] else styleErrorDefault)
          yi += 1

          if(param_sampling == 'L'):
            if(param_adjinflat == False):
              vl_LU, vh_LU = vl, vh
            else:
              vl_LA, vh_LA = vl, vh

        x     = np.array(x)
        x_lb  = np.array(x_lb)
        x_ub  = np.array(x_ub)
        y     = np.array(y)
        y_lbl = tuple([''] + y_lbl + [''])
        y_clr = tuple(y_clr)
        asymmetric_error = [x_lb, x_ub]

        # draws the subplot
        plt.subplot(nrows, ncols, 1 + pos)
        plt.grid(True, color='w', linestyle='solid', linewidth=lw/2)
        plt.gca().patch.set_facecolor('0.95')
        plt.gca().set_xlim(x_min, 1.0)
        plt.gca().set_ylim(0.0,   y_max)
        plt.subplots_adjust(left=.06, bottom=.06, right=.94, top=.94, wspace=.30, hspace=.25)

        #plt.errorbar(x, y, xerr=asymmetric_error, fmt='o', lw=lw, capsize=5)
        for i in range(len(x)):
          plt.errorbar(x[i], y[i], xerr=[[x_lb[i]],[x_ub[i]]], fmt='o', color=y_clr[i], lw=lw, capsize=5)

        locs, labels = plt.yticks()
        plt.yticks(locs, y_lbl)

        # add details to the subplot

        # 1. draws the baseline range for Group C1 (Hypothesis H2)
        plt.fill_betweenx((0.0, 0.7), vl_LU, vh_LU, alpha=tsp, color=styleFill2)
        plt.fill_betweenx((0.7, 1.4), vl_LA, vh_LA, alpha=tsp, color=styleFill1)

        # 2. sets column titles above the first row
        if(pos < ncols):
          plt.text(0.5, 1.05, model_name, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # 3. sets row titles after the last column
        if(pos % ncols == ncols - 1):
          plt.text(1.15, 0.45, configid, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # sets plot indicator to the next subplot
        pos += 1

  if(saveit):
    print('-- saving the figures.')
    if(not exists(join(*param_targetpath))):
      makedirs(join(*param_targetpath))
    plt.savefig(join(*param_targetpath, filename), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the figure.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))


def plotH3(summary, highlight, all_models, params, configs):

  (saveit, param_targetpath, filename) = params

  # initialises subplots
  nrows, ncols = len(configs) * 2, len(all_models) + 1
  figsizew, figsizeh = 17.06, len(configs) * 5.1
  fig, axes = plt.subplots(nrows, ncols, figsize=(figsizew, figsizeh), sharey='all')
  x_min, y_max = .4, 1.4

  # sets up the bottom-up order in which each semi-condition (sampling and inflation schemes) appears in a subplot
  all_conditions = [('R', False), ('R', True), ('H', False), ('H', True), ('L', False), ('L', True)]

  pos = 0
  for configid in configs:
    for param_optimode in [False, True]:
      for (param_model, _) in [('Ensemble', None)] +  all_models:
        model_name = param_model if type(param_model) is str else 'Ensemble'

        # recovers the data for each semicondition presented in a error plot <group, model, (LHR) x (UA)>
        x, x_lb, x_ub = [], [], []
        y, y_lbl, y_clr  = [], [], [] # y-coordinate, label and color for each semi-condition in a error plot
        yi = 1 # index of the semi-condition
        for (param_sampling, param_adjinflat) in all_conditions:
          condition = getConditionLabel((param_sampling, param_adjinflat, param_optimode))

          (ss, vm, vl, vh) = summary[(configid, model_name, condition)]
          x.append(vm)
          x_lb.append(vm - vl)
          x_ub.append(vh - vm)
          y.append(yi * .2)
          y_lbl.append(condition)
          y_clr.append(styleErrorHighlight if highlight[(configid, model_name, condition)] else styleErrorDefault)
          yi += 1

          if(param_adjinflat == False):
            if(param_sampling == 'L'):
              vl_LU, vh_LU = vl, vh
            elif(param_sampling == 'H'):
              vl_HU, vh_HU = vl, vh
            else:
              vl_RU, vh_RU = vl, vh

        x     = np.array(x)
        x_lb  = np.array(x_lb)
        x_ub  = np.array(x_ub)
        y     = np.array(y)
        y_lbl = tuple([''] + y_lbl + [''])
        y_clr = tuple(y_clr)
        asymmetric_error = [x_lb, x_ub]

        # draws the subplot
        plt.subplot(nrows, ncols, 1 + pos)
        plt.grid(True, color='w', linestyle='solid', linewidth=lw/2)
        plt.gca().patch.set_facecolor('0.95')
        plt.gca().set_xlim(x_min, 1.0)
        plt.gca().set_ylim(0.0,   y_max)
        plt.subplots_adjust(left=.06, bottom=.06, right=.94, top=.94, wspace=.30, hspace=.25)

        #plt.errorbar(x, y, xerr=asymmetric_error, fmt='o', lw=lw, capsize=5)
        for i in range(len(x)):
          plt.errorbar(x[i], y[i], xerr=[[x_lb[i]],[x_ub[i]]], fmt='o', color=y_clr[i], lw=lw, capsize=5)

        locs, labels = plt.yticks()
        plt.yticks(locs, y_lbl)

        # add details to the subplot

        # 1. draws the baseline range for Group C1 (Hypothesis H3)
        plt.fill_betweenx((0.0, 0.5), vl_RU, vh_RU, alpha=tsp, color=styleFill3)
        plt.fill_betweenx((0.5, 0.9), vl_HU, vh_HU, alpha=tsp, color=styleFill2)
        plt.fill_betweenx((0.9, 1.4), vl_LU, vh_LU, alpha=tsp, color=styleFill1)

        # 2. sets column titles above the first row
        if(pos < ncols):
          plt.text(0.5, 1.05, model_name, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # 3. sets row titles after the last column
        if(pos % ncols == ncols - 1):
          plt.text(1.15, 0.45, configid, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # sets plot indicator to the next subplot
        pos += 1

  if(saveit):
    print('-- saving the figures.')
    if(not exists(join(*param_targetpath))):
      makedirs(join(*param_targetpath))
    plt.savefig(join(*param_targetpath, filename), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the figure.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))


def plotH4(summary, highlight, all_models, params, configs):

  (saveit, param_targetpath, filename) = params

  # initialises subplots
  nrows, ncols = len(configs) * 2, len(all_models) + 1
  figsizew, figsizeh = 17.06, len(configs) * 5.1
  fig, axes = plt.subplots(nrows, ncols, figsize=(figsizew, figsizeh), sharey='all')
  x_min, y_max = .4, 1.4

  # sets up the bottom-up order in which each semi-condition (sampling and inflation schemes) appears in a subplot
  all_conditions = [('R', False), ('R', True), ('H', False), ('H', True), ('L', False), ('L', True)]

  pos = 0
  for configid in configs:
    for param_adjinflat in [False, True]:
      for (param_model, _) in [('Ensemble', None)] +  all_models:
        model_name = param_model if type(param_model) is str else 'Ensemble'

        # recovers the data for each semicondition presented in a error plot <group, model, (LHR) x (UA)>
        x, x_lb, x_ub = [], [], []
        y, y_lbl, y_clr  = [], [], [] # y-coordinate, label and color for each semi-condition in a error plot
        yi = 1 # index of the semi-condition
        for (param_sampling, param_optimode) in all_conditions:
          condition = getConditionLabel((param_sampling, param_adjinflat, param_optimode))

          (ss, vm, vl, vh) = summary[(configid, model_name, condition)]
          x.append(vm)
          x_lb.append(vm - vl)
          x_ub.append(vh - vm)
          y.append(yi * .2)
          y_lbl.append(condition)
          y_clr.append(styleErrorHighlight if highlight[(configid, model_name, condition)] else styleErrorDefault)
          yi += 1

          if(param_optimode == False):
            if(param_sampling == 'L'):
              vl_LN, vh_LN = vl, vh
            elif(param_sampling == 'H'):
              vl_HN, vh_HN = vl, vh
            else:
              vl_RN, vh_RN = vl, vh

        x     = np.array(x)
        x_lb  = np.array(x_lb)
        x_ub  = np.array(x_ub)
        y     = np.array(y)
        y_lbl = tuple([''] + y_lbl + [''])
        y_clr = tuple(y_clr)
        asymmetric_error = [x_lb, x_ub]

        # draws the subplot
        plt.subplot(nrows, ncols, 1 + pos)
        plt.grid(True, color='w', linestyle='solid', linewidth=lw/2)
        plt.gca().patch.set_facecolor('0.95')
        plt.gca().set_xlim(x_min, 1.0)
        plt.gca().set_ylim(0.0,   y_max)
        plt.subplots_adjust(left=.06, bottom=.06, right=.94, top=.94, wspace=.30, hspace=.25)

        #plt.errorbar(x, y, xerr=asymmetric_error, fmt='o', lw=lw, capsize=5)
        for i in range(len(x)):
          plt.errorbar(x[i], y[i], xerr=[[x_lb[i]],[x_ub[i]]], fmt='o', color=y_clr[i], lw=lw, capsize=5)

        locs, labels = plt.yticks()
        plt.yticks(locs, y_lbl)

        # add details to the subplot

        # 1. draws the baseline range for Group C1 (Hypothesis H3)
        plt.fill_betweenx((0.0, 0.5), vl_RN, vh_RN, alpha=tsp, color=styleFill3)
        plt.fill_betweenx((0.5, 0.9), vl_HN, vh_HN, alpha=tsp, color=styleFill2)
        plt.fill_betweenx((0.9, 1.4), vl_LN, vh_LN, alpha=tsp, color=styleFill1)

        # 2. sets column titles above the first row
        if(pos < ncols):
          plt.text(0.5, 1.05, model_name, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # 3. sets row titles after the last column
        if(pos % ncols == ncols - 1):
          plt.text(1.15, 0.45, configid, transform=plt.gca().transAxes, fontsize=fs, horizontalalignment='center')

        # sets plot indicator to the next subplot
        pos += 1

  if(saveit):
    print('-- saving the figures.')
    if(not exists(join(*param_targetpath))):
      makedirs(join(*param_targetpath))
    plt.savefig(join(*param_targetpath, filename), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the figure.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))


def main(essayid):

  tsprint('Process started.')

  # sets up the scope delimiting variables
  base_path = [getMountedOn(), 'Task Stage', 'Task - Trend Analysis', 'datasets', 'sp500', essayid, 'measure']
  all_models = [('MA', None), ('ARIMA', None), ('EWMA', None), ('KNN', None), ('SAX', None), ('LSTM', None)]
  all_conditions = [('linear', False, False),
                    ('linear', False, True),
                    ('linear', True, False),
                    ('linear', True, True),
                    ('heuristic', False, False),
                    ('heuristic', False, True),
                    ('heuristic', True, False),
                    ('heuristic', True, True),
                    ('random', False, False),
                    ('random', False, True),
                    ('random', True, False),
                    ('random', True, True),
                   ]
  all_configs = ['C1', 'C2', 'C3']
  tsprint('-- analysing experiment results in {0}.'.format(join(*base_path)))

  # goes through the measurement results and collects accuracy metrics
  tsprint('Summarising results obtained from each model and conditions.')
  summary, content = gatherResults(all_configs, all_models, all_conditions, base_path)
  saveAsText('\n'.join(content), join(*base_path, 'summary.csv'))

  # compares results according to the set of hypothesis in the study
  tsprint('Performing comparisons for all hypotheses.')
  header = ['HH\tComparand\tRelation\tReference\tc1.ss\tc1.lb\tc1.ub\tc2.ss\tc2.lb\tc2.ub']
  cpH1, hl_H1 = gencpH1(all_configs, all_models, all_conditions, summary)
  cpH2, hl_H2 = gencpH2(all_configs, all_models, all_conditions, summary)
  cpH3, hl_H3 = gencpH3(all_configs, all_models, all_conditions, summary)
  cpH4, hl_H4 = gencpH4(all_configs, all_models, all_conditions, summary)
  cpH5, hl_H5 = gencpH5(all_configs, all_models, all_conditions, summary)
  content = header + cpH1 + cpH2 + cpH3 + cpH4 + cpH5
  saveAsText('\n'.join(content), join(*base_path, 'hypotheses.csv'))

  # draws the result panels
  tsprint('Plotting the results panel.')
  saveit = True
  param_targetpath = [getMountedOn(), 'Task Stage', 'Task - Trend Analysis', 'datasets', 'sp500', essayid, 'measure']

  filename = 'panel_H1'
  plotH1(summary, hl_H1, all_models, (saveit, param_targetpath, filename))

  for configid in ['C1', 'C2', 'C3']:
    filename = 'panel_H2_{0}'.format(configid)
    plotH2(summary, hl_H2, all_models, (saveit, param_targetpath, filename), [configid])

  for configid in ['C1', 'C2', 'C3']:
    filename = 'panel_H3_{0}'.format(configid)
    plotH3(summary, hl_H3, all_models, (saveit, param_targetpath, filename), [configid])

  for configid in ['C1', 'C2', 'C3']:
    filename = 'panel_H4_{0}'.format(configid)
    plotH4(summary, hl_H4, all_models, (saveit, param_targetpath, filename), [configid])

  # creates a text file with the table contents and the panels for Figures 1 and 2
  tsprint('Process completed.')

if __name__ == '__main__':
  main(sys.argv[1])



