import sys
import os
#import shutil
import random
import userDefs as ud

from os        import listdir, makedirs, remove
from os.path   import join, isfile, isdir, exists
from itertools import chain

from userDefs  import ECO_SEED, tsprint, saveLog, setupEssayConfig, getEssayParameter, stimestamp, deserialise, serialise, saveAsText
from userDefs  import getMountedOn, ts2datestr, applyWeights, getFolderForecast, getFolderOptimise

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
    param_targetpath = [getMountedOn()] + getEssayParameter('PARAM_TARGETPATH') + [essayid, 'optimise', configid]
    param_maxcores   = getEssayParameter('PARAM_MAXCORES')
    param_stocklist  = getEssayParameter('PARAM_STOCKLIST')
    param_sampling   = getEssayParameter('PARAM_SAMPLING')
    param_adjinflat  = getEssayParameter('PARAM_ADJINFLAT')
    param_prices     = getEssayParameter('PARAM_PRICES')
    param_models     = getEssayParameter('PARAM_MODELS')
    param_optimode   = getEssayParameter('PARAM_OPTIMODE')

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

      # ensures output directory exists abd previous results are removed
      param_targetpath.append(getFolderOptimise(param_sampling, param_models, param_adjinflat, param_optimode))
      if(exists(join(*param_targetpath))):
        for f in listdir(join(*param_targetpath)):
          remove(join(*param_targetpath, f))
      else:
        makedirs(join(*param_targetpath))

      # performs the problem-specific task
      # ------------------------------------------------------------------------------------------------------

      tsprint('Loading preprocessed data.')
      constituents = deserialise(join(*param_sourcepath, 'preprocess', 'constituents'))
      stocks       = deserialise(join(*param_sourcepath, 'preprocess', 'stocks'))
      timeline     = deserialise(join(*param_sourcepath, 'preprocess', 'timeline'))

      folder = getFolderForecast(param_sampling, param_models)
      testPoints              = deserialise(join(*param_sourcepath, 'forecast', configid, folder, 'testPoints'))
      (forecasts, failures)   = deserialise(join(*param_sourcepath, 'forecast', configid, folder, 'forecasts'))

      tsprint('-- {0} test points allocated to the sample to be employed in model evaluation.'.format(len(testPoints)))
      tsprint('   {0}'.format(', '.join([ts2datestr(timeline[timepos]) + ' ({0})'.format(timepos) for timepos in testPoints])), False)
      tsprint('-- {0} forecasts were performed (#stocks:{1}, #models:{2}, #prices:{3}, #testpoints: {4}).'.format(len(forecasts), len(param_stocklist), len(param_models), len(param_prices), len(testPoints)))
      tsprint('-- {0} forecasts failed because the raw data was unavailable or the model failed.'.format(failures))

      tsprint('Performing iterative weights optimisation.')
      weights, thresholds, limits, pairs = applyWeights(param_stocklist, param_prices, param_models, param_adjinflat, param_optimode, timeline, forecasts, testPoints)
      tsprint('-- {0} forecasts are available for model evaluation.'.format(len(list(chain(*pairs.values())))))

      # ------------------------------------------------------------------------------------------------------

      # saves main datastructures produce in this module
      serialise(weights,     join(*param_targetpath, 'weights'))
      serialise(thresholds,  join(*param_targetpath, 'thresholds'))
      serialise(limits,      join(*param_targetpath, 'limits'))
      serialise(pairs,       join(*param_targetpath, 'pairs'))

      # saving the pairs dictionary into a csv file for easier visual inspection
      content = ['{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format('Ticker', 'Date', 'Actual', 'Predicted', 'Actual', 'Predicted', 'Result')]
      for ticker in pairs:
        for (timepos, realVal, predVal, realClass, predClass) in pairs[ticker]:
          content.append('{0}\t{1}\t{2:7.2f}\t{3:7.2f}\t{4}\t{5}\t{6}'.format(ticker, ts2datestr(timeline[timepos]), realVal, predVal, realClass, predClass, 'match' if realClass == predClass else 'miss'))
      saveAsText('\n'.join(content), join(*param_targetpath, 'pairs.csv'))

      content = 'Execution details can be found in the essay config file at {0}\n\n{1}'.format(slot, setupEssayConfig(configFile))
      saveAsText(content, join(*param_targetpath, 'config.log'))

    tsprint('Finished processing essay specs at [{0}]\n'.format(essay_configs))
    saveLog(join(slot, 'config.log'))
    saveLog(join(*param_targetpath, 'config.log'))

  print()
  tsprint('Essay completed.')


if __name__ == "__main__":

  main(sys.argv[1])
