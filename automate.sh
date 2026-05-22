#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# ── 1. System deps ──
echo "Checking deps..."
if ! command -v ffmpeg &>/dev/null; then brew install ffmpeg; fi

# ── 2. Venv ──
VENV_READY=0
if [ -f ".venv/bin/python" ]; then
    if .venv/bin/python -c "import yaml" &>/dev/null; then VENV_READY=1; fi
fi
if [ "$VENV_READY" -eq 0 ]; then
    echo "Setting up venv..."
    PY=""
    for py in /usr/local/bin/python3 python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$py" &>/dev/null || continue
        "$py" -c "import xml.parsers.expat; import sys; exit(0 if sys.version_info>=(3,11) else 1)" 2>/dev/null && PY="$py" && break
    done
    [ -z "$PY" ] && echo "Need Python 3.11+" && exit 1
    rm -rf .venv; "$PY" -m venv .venv
    .venv/bin/python -m pip install -q -r requirements.txt
fi

# ── 3. Menu ──
echo ""
echo "=== yt-clips ==="
echo "1) Local    (download→transcribe→export→sync→upload)"
echo "2) Remote   (beam to Colab via tunnel)"
echo "3) Sync     (upload shorts/ to Google Drive)"
echo "4) AutoPilot(watch channel for new VODs)"
echo "5) Tunnel   (check Colab tunnel status)"
read -p "Choice [1-5]: " mode

case "$mode" in
    1) .venv/bin/python -m automation.cli "$@" --sync --upload --schedule ;;
    2) .venv/bin/python -m automation.cli --remote "$@" ;;
    3) .venv/bin/python -m automation.cli --sync-only ;;
    4) .venv/bin/python -m automation.cli --auto-pilot "https://www.youtube.com/@CricketWithPrajjwal2.0" ;;
    5) .venv/bin/python -m automation.cli --tunnel-status ;;
    *) echo "Invalid" && exit 1 ;;
esac
