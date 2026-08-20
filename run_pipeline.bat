@echo off
setlocal enabledelayedexpansion
title yt-clips - One Click Pipeline
cd /d "%~dp0"

echo ============================================
echo   YT CLIPS - One Click Pipeline
echo ============================================
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv not found. Run: python -m venv .venv
    pause
    exit /b 1
)

rem ---- Step 1: token check + auto re-auth ----
echo [1/3] Checking OAuth tokens...
"%PY%" -c "from utils.token_refresh import ensure_fresh_tokens; import sys; sys.exit(0 if ensure_fresh_tokens(False) else 1)"
if errorlevel 1 (
    echo [WARN] Tokens expired - opening browser for re-auth...
    echo        Login with PERSONAL then CHANNEL account.
    "%PY%" setup_auth.py
)
echo.

rem ---- Step 2: mode menu ----
echo Choose mode:
echo   1) FULL     - download, export, SEO, upload  (needs URL)
echo   2) QUICK    - reuse input\video.mp4 + transcript, no upload
echo   3) DRY RUN  - test pipeline, no external calls
echo.
set "MODE="
set /p "MODE=Choice [1/2/3]: "

if "%MODE%"=="2" goto quick
if "%MODE%"=="3" goto dry

rem ---- mode 1: full run, ask URL ----
set "URL="
set /p "URL=Paste YouTube URL: "
if "%URL%"=="" (
    echo [WARN] No URL - falling back to QUICK mode.
    goto quick
)
echo.
echo [2/3] Running full pipeline...
echo [3/3] This will download, transcribe, export, SEO and UPLOAD.
echo.
"%PY%" pipeline.py "%URL%"
goto done

:quick
echo.
echo [2/3] Reusing existing input\video.mp4...
echo [3/3] Exporting clips from local video (no upload).
echo.
"%PY%" pipeline.py "https://youtu.be/local" --skip-download --skip-transcribe --no-upload --no-sync --no-schedule
goto done

:dry
echo.
echo [2/3] Running dry run (all external APIs stubbed)...
"%PY%" dry_run.py "https://youtu.be/test" --skip-download --skip-transcribe
goto done

:done
echo.
echo ============================================
echo   PIPELINE FINISHED
echo ============================================
pause