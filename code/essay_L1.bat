@echo off
rem Essay L1 (Level 1) - execute the FOM (Forecast, Optimise, and Measure) pipeline for each config
rem setlocal

@REM export PYTHONHASHSEED = 23

rem  submits data from the Baseline stocks (C1) to the FOM pipeline; all configs are considered
call essay_L2 %1 C1

rem  submits data from the Defensive stocks (C1) to the FOM pipeline; all configs are considered
call essay_L2 %1 C2

rem  submits data from the Cyclical stocks (C1) to the FOM pipeline; all configs are considered
call essay_L2 %1 C3
