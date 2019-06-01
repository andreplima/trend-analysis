import sys
import pickle
import codecs
import os
import os.path

from os.path  import join, exists
from userDefs import ECO_TICKER_ENSEMBLE, TypeResult, getMountedOn, getFolderOptimise, tsprint, deserialise, saveAsText

def fmtaccci(vl, vh):
  # an invalid entry is represented as a dash
  return '{0:4.3f} - {1:4.3f}'.format(vl, vh) if (vl >= 0 and vh >= 0) else '---'

def main(essayid):

  tsprint('Process started.')
  # sets up the scope delimiting variables
  base_path = [getMountedOn(), 'Task Stage', 'Task - Trend Analysis', 'datasets', 'sp500', essayid, 'measure']
  all_models = [('MA', None), ('EWMA', None), ('ARIMA', None), ('KNN', None), ('SAX', None), ('LSTM', None)]
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

  # goes through the measurement results of each model and collect accuracy metrics
  tsprint('Rendering the table with the summary of results.')
  table1 = ['*** Insert this snippet in the TEX document, after the "%--- snippet from gentex.py -- START" remark']
  table1.append('% data obtained with python gentex.py {0}'.format(essayid))
  performance = {}
  for (param_model, _) in all_models:
    state = 0
    for (param_sampling, param_adjinflat, param_optimode) in all_conditions:
      condition = param_sampling[0].upper() + ('A' if param_adjinflat else 'O') + ('O' if param_optimode else 'N')
      for configid in all_configs:
        path = base_path + [configid, getFolderOptimise(param_sampling, [(param_model,_)], param_adjinflat, param_optimode)]

        # recover experiment results for specific model, condition and config
        if(exists(join(*path, 'results.pkl'))):
          results = deserialise(join(*path, 'results'))
          #! xxx bootstrap to compute confidence intervals
          vl = results[ECO_TICKER_ENSEMBLE].accuracy
          vh = results[ECO_TICKER_ENSEMBLE].accuracy
        else:
          vl = -1
          vh = -1

        #! missing entries will be represented by a dash
        performance[(param_model, param_sampling, param_adjinflat, param_optimode, configid)] = (vl,vh)
        if(state == 0):
          newrow = [param_model, condition, fmtaccci(vl, vh)]
        elif(state == 3):
          newrow += [condition, fmtaccci(vl, vh)]
        else:
          newrow.append(fmtaccci(vl, vh))
        state += 1

      # creates a new row in the table
      if(state % 6 == 0):
        table1.append('proposed&{0}&{1}&{2}&{3}&{4}&{5}&{6}&{7}&{8}\\\\'.format(*newrow))
        table1.append('\\midrule')
        state = 0
  table1.append('')

  # Figure 1 - different models produce different errors under different conditions?
  # it consists of three panels, one for each config
  # each panel is a 2x3 grid with confusion matrices for each individual model
  # only for the LON condition: linear sampling, original prices, non-optimised weights
  tsprint('Rendering the panels for Figure 1.')
  figure1 = ['*** Insert snippet in the TEX document, after the "%--- snippet from gentex.py -- START" remark']
  figure1.append('% data obtained with python gentex.py {0}'.format(essayid))
  (param_sampling, param_adjinflat, param_optimode) = ('linear', False, False)
  for configid in all_configs:
    figure1.append('%--- Panel for config {0}'.format(configid))
    for i in range(len(all_models)):
      (param_model, _) = all_models[i]
      config_desc = getFolderOptimise(param_sampling, [(param_model, None)], param_adjinflat, param_optimode)
      filename    = 'cm_{0}_{1}'.format(configid, config_desc)
      if(not exists(join(*path, filename + '.pkl'))):
        filename = 'placeholder'
      figure1.append('% new row' if (i % 3 == 0) else '\\hfill')
      figure1.append('\\begin{subfigure}[t]{0.32\\textwidth}')
      figure1.append('  \\includegraphics[width=\\textwidth]{{images/{0}}}'.format(filename))
      figure1.append('  \\caption{{\\scriptsize {0}}}'.format(param_model))
      figure1.append('  \\label{{fig:{0}}}'.format(filename))
      figure1.append('\\end{subfigure}')
    figure1.append('')

  # Figure 2 - the ensemble produces different errors under different conditions?
  # it consists of three panels, one for each config
  # each panel is a 2x3 grid with confusion matrices for each individual model
  # only for the LON condition: linear sampling, original prices, non-optimised weights
  tsprint('Rendering the panels for Figure 2.')
  figure2 = []


  # creates a text file with the table contents and the panels for Figures 1 and 2
  saveAsText('\n'.join(table1 + figure1 + figure2), join(*base_path, 'measure_summary.tex'))
  tsprint('Process completed.')

if __name__ == '__main__':
  main(sys.argv[1])
