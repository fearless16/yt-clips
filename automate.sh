#!/usr/bin/env bash
# automate.sh — One-command entry point for yt-clips (STABLE MAC VERSION)

set -e

# ─── 1. System Dependencies ──────────────────────────────────────────────────
echo "Checking system dependencies..."

# Ensure we are in the project folder
cd "$(dirname "$0")"

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    brew install ffmpeg || echo "⚠️ Please install FFmpeg manually: brew install ffmpeg"
fi

# ─── 2. Stable Environment Setup ─────────────────────────────────────────────
VENV_READY=0
if [ -f ".venv/bin/python" ]; then
    if .venv/bin/python -c "import googleapiclient" &>/dev/null; then
        VENV_READY=1
    fi
fi

if [ "$VENV_READY" -eq 0 ]; then
    echo "→  Discovering best available Python …"
    # Find a suitable python (3.10+). Prioritize python.org install as Homebrew is often broken on Mac.
    PYTHON_CMD=""
    for py in /usr/local/bin/python3 python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$py" &> /dev/null; then
            # TEST: Verify the python isn't broken by the macOS expat linkage bug
            if ! "$py" -c "import xml.parsers.expat" &>/dev/null; then
                echo "⚠️  Skipping $py (broken library linkage)"
                continue
            fi
            
            # Check if version >= 3.10 using the candidate Python itself
            if "$py" -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
                PYTHON_CMD="$py"
                break
            fi
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        echo "❌ No suitable/working Python (3.10+) found. Please install a stable version: brew install python@3.12"
        exit 1
    fi

    echo "→  Creating virtual environment using $PYTHON_CMD ($($PYTHON_CMD --version 2>&1)) …"
    rm -rf .venv
    # Try standard venv first, fallback to without-pip if ensurepip is broken
    if ! "$PYTHON_CMD" -m venv .venv 2>/dev/null; then
        "$PYTHON_CMD" -m venv .venv --without-pip
    fi
    
    echo "→  Ensuring pip is available …"
    # If pip is missing in venv, bootstrap it using system python
    if [ ! -f ".venv/bin/pip" ]; then
        "$PYTHON_CMD" -m pip install --upgrade --target ".venv/lib/python$VER/site-packages" pip &>/dev/null || true
    fi
    
    echo "→  Installing project dependencies …"
    # Use the venv's python -m pip (most reliable across all setups)
    .venv/bin/python -m pip install -q -r requirements.txt
fi

# ─── 3. Auto-Discovery (GCloud & Tools) ───────────────────────────────────────
# Check for Homebrew gcloud or local install
if command -v gcloud &>/dev/null; then
    GCLOUD_BIN=$(command -v gcloud)
elif [ -d "/opt/homebrew/bin" ] && [ -f "/opt/homebrew/bin/gcloud" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [ -d "/usr/local/bin" ] && [ -f "/usr/local/bin/gcloud" ]; then
    export PATH="/usr/local/bin:$PATH"
elif [ -d "$HOME/google-cloud-sdk/bin" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# ─── 4. Run Choice ────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  🏏 yt-clips — Premium Cricket Shorts Factory"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Select execution mode:"
echo "  1) 🖥  Local Run   (Uses your PC's CPU)"
echo "  2) ☁  Remote Run  (Offload to Google Colab GPU)"
echo "  3) 📤 Sync Only   (Upload shorts/ to Google Drive)"
echo "  4) 🤖 Auto-Pilot  (Watch channel for new VODs)"
echo ""
echo "═══════════════════════════════════════════════════"
read -p "Choice [1/2/3/4]: " mode

# ─── Auto-Pilot Mode ─────────────────────────────────────────────────────────
if [ "$mode" == "4" ]; then
    echo "🚀 Auto-Pilot Active: Watching @CricketWithPrajjwal2.0..."
    source .venv/bin/activate
    python channel_watcher.py "https://www.youtube.com/@CricketWithPrajjwal2.0"
    exit 0
fi

# ─── Sync-Only Mode ──────────────────────────────────────────────────────────
if [ "$mode" == "3" ]; then
    echo "📤 Syncing shorts to Google Drive..."
    source .venv/bin/activate
    python sync.py
    exit 0
fi

# ─── Remote Mode (Colab Bridge) ──────────────────────────────────────────────
if [ "$mode" == "2" ]; then
    if [ $# -eq 0 ]; then
        echo "Error: No YouTube URL provided."
        echo "Usage: ./automate.sh \"https://youtu.be/URL\""
        exit 1
    fi
    
    source .venv/bin/activate
    echo ""
    echo "─── Step 1: Syncing Code to Google Drive ─────────────"
    # Push code first so the Colab worker runs the latest downloader/pipeline.
    python push_code.py
    
    echo ""
    echo "─── Step 2: Beaming job to Cloud Bridge ───────────────"
    # Do not skip download: Colab should download the source video itself.
    python bridge.py "$@" --sync --upload --schedule
    exit 0
fi

# ─── Local Mode ──────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "Error: No YouTube URL provided."
    echo "Usage: ./automate.sh \"https://youtu.be/URL\""
    exit 1
fi

echo ""
echo "🚀 Starting Premium Pipeline (Full Automation)..."
source .venv/bin/activate
# --sync --upload --schedule handles everything end-to-end
python pipeline.py "$@" --sync --upload --schedule

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Pipeline Complete!"
echo "═══════════════════════════════════════════════════"
