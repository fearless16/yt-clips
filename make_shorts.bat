@echo off
setlocal
title yt-clips - Make Cricket Shorts
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python environment missing: %PY%
    echo Run setup first, then try again.
    pause
    exit /b 1
)

if not "%~1"=="" goto run_args

set /p "URL=Paste YouTube video URL: "
if not defined URL (
    echo [ERROR] YouTube URL is required.
    pause
    exit /b 2
)

echo.
echo Making cricket Shorts locally. Nothing will be uploaded.
echo.
"%PY%" -m automation.cli "%URL%"
goto finished

:run_args
echo.
echo Making cricket Shorts locally. Nothing will be uploaded.
echo.
"%PY%" -m automation.cli %*

:finished
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo DONE - open the newest folder inside shorts\
) else (
    echo FAILED - check logs\pipeline.log
)

if "%~1"=="" pause
exit /b %EXIT_CODE%
