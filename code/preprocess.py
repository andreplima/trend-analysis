import sys
#import os
#import shutil
import random
import userDefs as ud

from os.path import join, isfile, isdir, exists
from os      import listdir, makedirs, remove

from userDefs import ECO_SEED, tsprint, saveLog, setupEssayConfig, getEssayParameter, stimestamp, deserialise, serialise, saveAsText
from userDefs import getMountedOn, loadConstituents, loadStocks, buildTimeline

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
    param_sourcepath = [getMountedOn()] + getEssayParameter('PARAM_SOURCEPATH')
    param_targetpath = [getMountedOn()] + getEssayParameter('PARAM_TARGETPATH') + [essayid, 'preprocess']
    param_consttfile = getEssayParameter('PARAM_CONSTTFILE')
    param_stocksfile = getEssayParameter('PARAM_STOCKSFILE')
    param_encoding   = getEssayParameter('PARAM_ENCODING')
    param_maxcores   = getEssayParameter('PARAM_MAXCORES')
    param_loadprices = getEssayParameter('PARAM_LOADPRICES')

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

      # performs the problem-specific task
      # ------------------------------------------------------------------------------------------------------

      tsprint('Loading constituents file.')
      constituents = loadConstituents(param_sourcepath, param_consttfile, param_encoding)

      tsprint('Loading from the {0} dataset the following series: {1}.'.format(param_dataset, ', '.join(param_loadprices)))
      stocks = loadStocks(param_sourcepath, param_stocksfile, param_encoding, param_loadprices)

      tsprint('Building the master timeline and updating the constituents time ranges')
      timeline, newConstituents = buildTimeline(stocks, constituents)

      # ------------------------------------------------------------------------------------------------------

      # saves the data generated during each run
      tsprint('-- saving preprocessed data.')

      if(exists(join(*param_targetpath))):
        for f in listdir(join(*param_targetpath)):
          remove(join(*param_targetpath, f))
      else:
        makedirs(join(*param_targetpath))

      serialise(newConstituents, join(*param_targetpath, 'constituents'))
      serialise(stocks,          join(*param_targetpath, 'stocks'))
      serialise(timeline,        join(*param_targetpath, 'timeline'))

      content = 'Execution details can be found in the essay config file at {0}\n\n{1}'.format(slot, setupEssayConfig(configFile))
      saveAsText(content, join(*param_targetpath, 'config.log'))

    tsprint('Finished processing essay specs at [{0}]\n'.format(essay_configs))
    saveLog(join(slot, 'config.log'))
    saveLog(join(*param_targetpath, 'config.log'))

  print()
  tsprint('Essay completed.')


if __name__ == "__main__":

  main(sys.argv[1])
