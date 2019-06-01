@echo off
rem Essay L2 (Level 2) - execute the FOM (Forecast, Optimise, and Measure) pipeline for each condition
rem setlocal

@REM export PYTHONHASHSEED = 23

rem  performs the forecast step for an ensemble of models under all conditions
rem individual models reuse forecasts previously computed for the ensemble
if not "%PARAM_MODELS%"=="" goto optimiseAndMeasure
set PARAM_SAMPLING=linear
call run %1 %2 f
set PARAM_SAMPLING=heuristic
call run %1 %2 f
set PARAM_SAMPLING=random
call run %1 %2 f

:optimiseAndMeasure
rem  performs the optimise and measure steps for the current model under all conditions

rem  all conditions related to linear sampling
set PARAM_SAMPLING=linear
set PARAM_ADJINFLAT=False
set PARAM_OPTIMODE=False
call run %1 %2 om

set PARAM_SAMPLING=linear
set PARAM_ADJINFLAT=False
set PARAM_OPTIMODE=True
call run %1 %2 om

set PARAM_SAMPLING=linear
set PARAM_ADJINFLAT=True
set PARAM_OPTIMODE=False
call run %1 %2 om

set PARAM_SAMPLING=linear
set PARAM_ADJINFLAT=True
set PARAM_OPTIMODE=True
call run %1 %2 om

rem  all conditions related to heuristic sampling
set PARAM_SAMPLING=heuristic
set PARAM_ADJINFLAT=False
set PARAM_OPTIMODE=False
call run %1 %2 om

set PARAM_SAMPLING=heuristic
set PARAM_ADJINFLAT=False
set PARAM_OPTIMODE=True
call run %1 %2 om

set PARAM_SAMPLING=heuristic
set PARAM_ADJINFLAT=True
set PARAM_OPTIMODE=False
call run %1 %2 om

set PARAM_SAMPLING=heuristic
set PARAM_ADJINFLAT=True
set PARAM_OPTIMODE=True
call run %1 %2 om

rem  all conditions related to random sampling
set PARAM_SAMPLING=random
set PARAM_ADJINFLAT=False
set PARAM_OPTIMODE=False
call run %1 %2 om

set PARAM_SAMPLING=random
set PARAM_ADJINFLAT=False
set PARAM_OPTIMODE=True
call run %1 %2 om

set PARAM_SAMPLING=random
set PARAM_ADJINFLAT=True
set PARAM_OPTIMODE=False
call run %1 %2 om

set PARAM_SAMPLING=random
set PARAM_ADJINFLAT=True
set PARAM_OPTIMODE=True
call run %1 %2 om

set PARAM_SAMPLING=
set PARAM_ADJINFLAT=
set PARAM_OPTIMODE=
