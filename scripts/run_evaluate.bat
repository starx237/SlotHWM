@echo off
REM =============================================
REM SlotPi Evaluation Script
REM Usage: run_evaluate.bat <stage> <dataset> <checkpoint>
REM =============================================
setlocal

set STAGE=%1
if "%STAGE%"=="" set STAGE=2

set DATASET=%2
if "%DATASET%"=="" set DATASET=obj3d

set CHECKPOINT=%3
if "%CHECKPOINT%"=="" set CHECKPOINT=experiments/stage%STAGE%/%DATASET%/checkpoints/best.pth

set CONFIG=config/stage%STAGE%/%DATASET%.yaml
set WORKDIR=experiments/eval/stage%STAGE%/%DATASET%

if not exist %CONFIG% (
    echo Error: Config file %CONFIG% not found!
    exit /b 1
)

if not exist %CHECKPOINT% (
    echo Error: Checkpoint %CHECKPOINT% not found!
    exit /b 1
)

python scripts/evaluate.py --config %CONFIG% --stage %STAGE% --checkpoint %CHECKPOINT% --workdir %WORKDIR% %4 %5 %6 %7 %8 %9

if %ERRORLEVEL% neq 0 (
    echo Evaluation failed!
    exit /b %ERRORLEVEL%
)

echo Evaluation complete. Results saved to %WORKDIR%