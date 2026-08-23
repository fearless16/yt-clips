@echo off
setlocal
title yt-clips - One Click Pipeline
cd /d "%~dp0"

echo ============================================
echo   YT CLIPS - One Click Pipeline
echo ============================================
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    pause
    exit /b 1
)

rem ---- [1/3] OAuth token check + auto re-auth ----
echo [1/3] Checking OAuth tokens...
"%PY%" -c "from utils.token_refresh import ensure_fresh_tokens; import sys; sys.exit(0 if ensure_fresh_tokens(False) else 1)"
if errorlevel 1 (
    echo [WARN] Tokens expired - opening browser for re-auth...
    echo        Login PERSONAL Google account first, then CHANNEL account.
    "%PY%" setup_auth.py
)
echo.

rem ---- [2/3] Mode ----
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

rem Upload is OPT-IN: nothing goes to YouTube without an explicit y here.
set "UPLOAD="
set /p "UPLOAD=Upload bhi karna hai? (scheduled slots pe) [y/N]: "
if /i "%UPLOAD%"=="y" (
    echo.
    echo [3/3] Full run + scheduled YouTube upload...
    "%PY%" -m automation.cli "%URL%" --upload --schedule
    goto done
)
echo.
echo [3/3] Full run, local only (no upload)...
"%PY%" -m automation.cli "%URL%"
goto done

:quick
echo.
echo [3/3] Local export from existing video...
"%PY%" -m automation.cli "https://youtu.be/local" --skip-download --skip-transcribe
goto done

:dry
echo.
echo [3/3] Dry run...
"%PY%" dry_run.py "https://youtu.be/test" --skip-download --skip-transcribe
goto done

:done
echo.
echo ============================================
echo   PIPELINE FINISHED  -  outputs: shorts\  logs\pipeline.log
echo ============================================
pause
