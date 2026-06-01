@echo off
REM =============================================
REM SlotPi Stage 2 Training Script
REM Usage: run_stage2_train.bat <dataset> [extra_args]
REM Example: run_stage2_train.bat clevrer
REM =============================================
setlocal

set DATASET=%1
if "%DATASET%"=="" set DATASET=obj3d

set CONFIG=config/stage2/%DATASET%.yaml
set WORKDIR=experiments/stage2/%DATASET%

if not exist %CONFIG% (
    echo Error: Config file %CONFIG% not found!
    exit /b 1
)

python scripts/train_stage2.py --config %CONFIG% --workdir %WORKDIR% %2 %3 %4 %5 %6 %7 %8 %9

if %ERRORLEVEL% neq 0 (
    echo Training failed!
    exit /b %ERRORLEVEL%
)

echo Stage 2 training complete. Output saved to %WORKDIR%