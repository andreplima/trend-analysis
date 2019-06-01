@echo off
rem Essay L1 (Level 1) - execute the FOM (Forecast, Optimise, and Measure) pipeline for each config
rem setlocal

@REM export PYTHONHASHSEED = 23

call essay_L2 %1 C1
call essay_L2 %1 C2
call essay_L2 %1 C3
