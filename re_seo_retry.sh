#!/usr/bin/env bash
# Run re-SEO on all 5 Shorts (safe to re-run — skips quota-exhausted gracefully)
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python re_seo.py \
  "https://youtube.com/shorts/yhbGk20AFas" \
  "https://youtube.com/shorts/hgneVa0n_HM?feature=share" \
  "https://youtube.com/shorts/uRBajQdeILQ?feature=share" \
  "https://youtube.com/shorts/NPSPlNkkA6s?feature=share" \
  "https://youtube.com/shorts/6RiKMn5EYlA?feature=share"
