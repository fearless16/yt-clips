#!/usr/bin/env bash
# setup.sh — One-shot environment setup for yt-clips
# Run once:  bash setup.sh
set -euo pipefail

VENV_DIR=".venv"
PYTHON_MIN="3.10"

# ─── Find a working Python ────────────────────────────────────────────────────
find_python() {
  # Potential candidates
  for py in python3.12 python3.11 python3.10 python3.13 python3 /usr/bin/python3; do
    if command -v "$py" &>/dev/null; then
      # TEST: Verify the python isn't broken by the macOS expat linkage bug
      if "$py" -c "import xml.parsers.expat" &>/dev/null; then
        echo "$py"
        return 0
      fi
    fi
  done
  echo ""
}

echo "═══════════════════════════════════════"
echo "  yt-clips — Environment Setup"
echo "═══════════════════════════════════════"

# ─── Check FFmpeg ─────────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
  echo "⚠  FFmpeg not found."
  echo "   Install with: brew install ffmpeg"
  echo "   Then re-run this script."
  exit 1
else
  echo "✓  FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
fi

# ─── Find Python ──────────────────────────────────────────────────────────────
PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
  echo "✗  No suitable Python found (need 3.10+)."
  echo "   Install with: brew install python@3.12"
  exit 1
fi
echo "✓  Python: $PYTHON ($($PYTHON --version))"

# ─── Create venv ──────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "→  Creating virtual environment at $VENV_DIR …"
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "✓  Virtual environment already exists at $VENV_DIR"
fi

# ─── Activate + install ───────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "→  Upgrading pip …"
pip install --quiet --upgrade pip

echo "→  Installing requirements …"
pip install -r requirements.txt

echo ""
echo "═══════════════════════════════════════"
echo "  Setup complete! ✓"
echo ""
echo "  Activate your environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run the full pipeline:"
echo "    python pipeline.py https://youtu.be/YOUR_VIDEO_ID"
echo "═══════════════════════════════════════"
