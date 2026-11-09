@echo off
setlocal
chcp 65001 >nul
title YT // CLIPS - Pipeline Control
cd /d "%~dp0"

echo ============================================
echo   YT // CLIPS  -  PIPELINE CONTROL
echo ============================================
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    pause
    exit /b 1
)

rem ---- [1/3] Collect all answers before starting work ----
echo Modes:
echo   1) NEW RUN  - URL se download, transcribe, select, export, SEO
echo   2) QUICK    - existing input\video.mp4 + transcript reuse (no download)
echo   3) DRY RUN  - stubbed external APIs, plumbing check only
echo.
set "MODE="
set /p "MODE=Choice [1/2/3]: "

if "%MODE%"=="2" goto quick
if "%MODE%"=="3" goto dry

set "URL="
set /p "URL=YouTube URL: "
if "%URL%"=="" (
    echo [WARN] No URL given - falling back to QUICK mode.
    goto quick
)

rem Publishing is OPT-IN: no YouTube or Instagram upload without explicit y.
set "UPLOAD="
set /p "UPLOAD=Upload bhi karna hai? (scheduled slots pe) [y/N]: "
if /i "%UPLOAD%"=="y" goto upload

echo.
echo [3/3] Full run, local only (no upload)...
"%PY%" -m automation.cli "%URL%" --source-scoped --skip-instagram
if errorlevel 1 goto failed
goto done

:upload
echo.
echo [2/3] Checking OAuth tokens...
"%PY%" -c "import sys, utils.token_refresh as t; sys.exit(0 if t.ensure_fresh_tokens(True) else 1)"
if errorlevel 1 (
    echo [ERROR] OAuth authorization failed. Upload aborted.
    goto failed
)
echo.
echo [3/3] Full run + Drive sync + scheduled YouTube upload...
"%PY%" -m automation.cli "%URL%" --source-scoped --sync --upload --schedule --skip-instagram
if errorlevel 1 goto failed
goto done

:quick
echo.
echo [3/3] Local export from existing video...
"%PY%" -m automation.cli "https://youtu.be/local" --skip-download --skip-transcribe --skip-instagram
if errorlevel 1 goto failed
goto done

:dry
echo.
echo [3/3] Dry run...
"%PY%" dry_run.py "https://youtu.be/test" --skip-download --skip-transcribe
if errorlevel 1 goto failed
goto done

:done
echo.
echo ============================================
echo   RUN CLOSED - see the outcome and artifact paths above
echo ============================================
pause
exit /b 0

:failed
echo.
echo ============================================
echo   PIPELINE FAILED - check the error above and logs\pipeline.log
echo ============================================
pause
exit /b 1
