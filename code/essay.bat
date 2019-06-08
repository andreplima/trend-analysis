@echo off
rem Essay L0 (Level 0) - execute the FOM (Forecast, Optimise, and Measure) pipeline for each model

@REM usage:   essay <essayID>
@REM example: run T01 - runs experiment T01 with configs C1,C2,C3 under varying conditions

@REM export PYTHONHASHSEED = 23

echo Essay run started at %date%-%time% > essay.log

rem  proprocess S&P500 data, which is used in all configs/conditions
rem  call run %1 C1 p

rem  performs the FOM (Forecast, Optimise, and Measure) pipeline for the ensemble of models
rem  -- all configs and conditions are considered
set  PARAM_MODELS=
call essay_L1 %1
 
rem  performs the FOM pipeline for the MA model; all configs and conditions are considered
set  PARAM_MODELS=MA
call essay_L1 %1

rem  performs the FOM pipeline for the ARIMA model; all configs and conditions are considered
set  PARAM_MODELS=ARIMA
call essay_L1 %1

rem  performs the FOM pipeline for the EWMA model; all configs and conditions are considered
set  PARAM_MODELS=EWMA
call essay_L1 %1

rem  performs the FOM pipeline for the KNN model; all configs and conditions are considered
set  PARAM_MODELS=KNN
call essay_L1 %1

rem  performs the FOM pipeline for the SAX model; all configs and conditions are considered
set  PARAM_MODELS=SAX
call essay_L1 %1

rem  performs the FOM pipeline for the LSTM model; all configs and conditions are considered
set  PARAM_MODELS=LSTM
call essay_L1 %1

rem  remove setting from the environment
set  PARAM_MODELS=
