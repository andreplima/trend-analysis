import sys
import os
#import shutil
import random
import userDefs as ud

from os.path   import join, isfile, isdir, exists
from os        import listdir, makedirs, remove
from itertools import chain

from userDefs import ECO_SEED, tsprint, saveLog, setupEssayConfig, getEssayParameter, stimestamp, deserialise, serialise, saveAsText
from userDefs import getMountedOn, getFolderOptimise, getPlotDesc, computeMetrics, plot_confusion_matrix, plot_ROC_curve
from userDefs import ECO_TICKER_ENSEMBLE

def main(essay_configs):

  """
  Purpose: read configuration files in a essay folder and execute them
  Arguments:
  - essay_configs: configuration file or subdiretory containing multiple configuration files,
                   representing essay(s) that must be executed
  Returns: None, but each essay run creates multiple files at Essays folder:
  """

  # identifies all config files that comprises the essay (must be in dir_essay_configs directory)
  tsprint('Processing essay specs at [{0}]\n'.format(essay_configs))
  if(isdir(essay_configs)):
    configFiles = [join(essay_configs, f) for f in listdir(essay_configs) if isfile(join(essay_configs, f))]
  elif(isfile(essay_configs)):
    configFiles = [essay_configs]
  else:
    print('Command line parameter is neither a recognised file nor directory: {0}'.format(essay_configs))
    raise ValueError

  # recovers each essay configuration and runs it
  for configFile in configFiles:

    # loads the essay configuration stored in the current config file
    ud.LogBuffer = []
    tsprint('Processing essay configuration file [{0}]\n{1}'.format(configFile, setupEssayConfig(configFile)))

    # recovers attributes related to essay identification
    essayid  = getEssayParameter('ESSAY_ESSAYID')
    configid = getEssayParameter('ESSAY_CONFIGID')
    scenario = getEssayParameter('ESSAY_SCENARIO')

    # assures the essay slot (where files will be created during a run) is available
    essay_beginning_ts = stimestamp()
    slot  = join('..', 'Essays', essayid, configid, essay_beginning_ts)
    if(not exists(slot)): makedirs(slot)

    # recovers parameters related to the problem instance
    param_dataset    = getEssayParameter('PARAM_DATASET')
    param_sourcepath = [getMountedOn()] + getEssayParameter('PARAM_SOURCEPATH') + [essayid]
    param_targetpath = [getMountedOn()] + getEssayParameter('PARAM_TARGETPATH') + [essayid, 'measure', configid]
    param_sampling   = getEssayParameter('PARAM_SAMPLING')
    param_adjinflat  = getEssayParameter('PARAM_ADJINFLAT')
    param_models     = getEssayParameter('PARAM_MODELS')
    param_optimode   = getEssayParameter('PARAM_OPTIMODE')
    param_saveimages = getEssayParameter('PARAM_SAVEIMAGES')

    if('PARAM_SAMPLING' in os.environ):
      param_sampling = os.environ['PARAM_SAMPLING']
      tsprint('-- option {0} updated from {1} to {2} (environment variable setting)'.format('PARAM_SAMPLING',
                                                                                              getEssayParameter('PARAM_SAMPLING'),
                                                                                              param_sampling))

    if('PARAM_MODELS' in os.environ):
      param_models = [(os.environ['PARAM_MODELS'], None)]
      tsprint('-- option {0} updated from {1} to {2} (environment variable setting)'.format('PARAM_MODELS',
                                                                                              getEssayParameter('PARAM_MODELS'),
                                                                                              param_models))
    if('PARAM_ADJINFLAT' in os.environ):
      param_adjinflat = os.environ['PARAM_ADJINFLAT'] == 'True'
      tsprint('-- option {0} updated from {1} to {2} (environment variable setting)'.format('PARAM_ADJINFLAT',
                                                                                              getEssayParameter('PARAM_ADJINFLAT'),
                                                                                              param_adjinflat))

    if('PARAM_OPTIMODE' in os.environ):
      param_optimode = os.environ['PARAM_OPTIMODE'] == 'True'
      tsprint('-- option {0} updated from {1} to {2} (environment variable setting)'.format('PARAM_OPTIMODE',
                                                                                              getEssayParameter('PARAM_OPTIMODE'),
                                                                                              param_optimode))


    # runs the essay configuration the required number of times
    (runID, maxRuns) = (0, getEssayParameter('ESSAY_RUNS'))
    while(runID < maxRuns):

      # decreases the number of runs and changes the random seed
      runID = runID + 1
      random.seed(ECO_SEED + runID)
      run_beginning_ts = stimestamp()

      # prints the run header
      print()
      tsprint('Starting run {0} of {1} for scenario [{2}]'.format(runID, maxRuns, scenario))
      tsprint('Essay [{0}], Config [{1}], Label [{2}]'.format(essayid, configid, run_beginning_ts))
      tsprint('Files will be created in [{0}]'.format(slot))
      print()

      # assures the target path (where results will be saved) is available
      param_targetpath.append(getFolderOptimise(param_sampling, param_models, param_adjinflat, param_optimode))
      if(exists(join(*param_targetpath))):
        for f in listdir(join(*param_targetpath)):
          remove(join(*param_targetpath, f))
      else:
        makedirs(join(*param_targetpath))

      # performs the problem-specific task
      # ------------------------------------------------------------------------------------------------------

      tsprint('Loading preprocessed data.')
      folder = getFolderOptimise(param_sampling, param_models, param_adjinflat, param_optimode)
      pairs = deserialise(join(*param_sourcepath, 'optimise', configid, folder, 'pairs'))
      tsprint('-- {0} predictions are available for model evaluation.'.format(len(list(chain(*pairs.values())))))

      tsprint('Computing ensemble accuracy.')
      results, all_true, all_pred = computeMetrics(pairs)
      res = results[ECO_TICKER_ENSEMBLE]
      tsprint('-- ensemble metrics: sample size: {0}, accuracy: {1:5.3f}, error: {2:5.3f}'.format(res.ss, res.accuracy, res.smape))

      tsprint('Plotting the confusion matrix and ROC curve for the ensemble.')
      plotDesc = getPlotDesc(configid, param_sampling, param_models, param_adjinflat, param_optimode)
      flag = plot_confusion_matrix(all_true, all_pred,
                            '{0}'.format(plotDesc),
                            param_saveimages, param_targetpath, 'cm_{0}_{1}'.format(configid, folder))

      if(flag):
        plot_ROC_curve(all_true, all_pred,
                              '{0}'.format(plotDesc),
                              param_saveimages, param_targetpath, 'roc_{0}_{1}'.format(configid, folder))
      else:
        tsprint('-- ROC curve was not produced because confusion matrix misses at least one class.')

      # ------------------------------------------------------------------------------------------------------

      # saves the data produced during the run
      serialise(results, join(*param_targetpath, 'results'))

      # saving the results dictionary into a csv file for easier visual inspection
      content = ['{0}\t{1}\t{2}\t{3}'.format('Ticker', 'Sample Size', 'Accuracy', 'SMAPE')]
      for ticker in results:
        content.append('{0}\t{1}\t{2:4.3f}\t{3:4.3f}'.format(ticker, results[ticker].ss, results[ticker].accuracy, results[ticker].smape))
      saveAsText('\n'.join(content), join(*param_targetpath, 'results.csv'))

      content = 'Execution details can be found in the essay config file at {0}\n\n{1}'.format(slot, setupEssayConfig(configFile))
      saveAsText(content, join(*param_targetpath, 'config.log'))

    tsprint('Finished processing essay specs at [{0}]\n'.format(essay_configs))
    saveLog(join(slot, 'config.log'))
    saveLog(join(*param_targetpath, 'config.log'))

  print()
  tsprint('Essay completed.')


if __name__ == "__main__":

  main(sys.argv[1])
