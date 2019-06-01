@echo off

@REM usage:   run <essayID> <config> [step]
@REM example: run T01 C1    runs experiment T01 with config C1, including the preprocess, predict and measure steps
@REM example: run T01 C1 p  same as before, but performs only the preprocess step

@REM export PYTHONHASHSEED = 23
echo running %1 %2 %3 with PARAM_MODELS=%PARAM_MODELS%, PARAM_SAMPLING=%PARAM_SAMPLING%, PARAM_ADJINFLAT=%PARAM_ADJINFLAT%, PARAM_OPTIMODE=%PARAM_OPTIMODE%

set flag_preprocess="0"
set flag_forecast="0"
set flag_optimise="0"
set flag_measure="0"

IF "%3"=="" (
  set flag_preprocess="1"
  set flag_forecast="1"
  set flag_optimise="1"
  set flag_measure="1"
)
IF "%3"=="p" (
  set flag_preprocess="1"
  set flag_forecast="0"
  set flag_optimise="0"
  set flag_measure="0"
)
IF "%3"=="f" (
  set flag_preprocess="0"
  set flag_forecast="1"
  set flag_optimise="0"
  set flag_measure="0"
)
IF "%3"=="o" (
  set flag_preprocess="0"
  set flag_forecast="0"
  set flag_optimise="1"
  set flag_measure="0"
)
IF "%3"=="m" (
  set flag_preprocess="0"
  set flag_forecast="0"
  set flag_optimise="0"
  set flag_measure="1"
)
IF "%3"=="om" (
  set flag_preprocess="0"
  set flag_forecast="0"
  set flag_optimise="1"
  set flag_measure="1"
)

IF "%3"=="fom" (
  set flag_preprocess="0"
  set flag_forecast="1"
  set flag_optimise="1"
  set flag_measure="1"
)

IF %flag_preprocess%=="1" (
  python -W ignore preprocess.py ..\Configs\%1\%2\preprocess_%1_%2.cfg
)

IF %flag_forecast%=="1" (
  python -W ignore forecast.py   ..\Configs\%1\%2\forecast_%1_%2.cfg
)

IF %flag_optimise%=="1" (
  python -W ignore optimise.py   ..\Configs\%1\%2\optimise_%1_%2.cfg
)

IF %flag_measure%=="1" (
  python -W ignore measure.py    ..\Configs\%1\%2\measure_%1_%2.cfg
)
