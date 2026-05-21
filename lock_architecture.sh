#!/bin/bash
# Re-lock LOCKED_ARCHITECTURE.md after editing
# Usage: ./lock_architecture.sh

chmod 444 LOCKED_ARCHITECTURE.md
echo "LOCKED: LOCKED_ARCHITECTURE.md is now read-only."
