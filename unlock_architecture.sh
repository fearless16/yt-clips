#!/bin/bash
# Unlock LOCKED_ARCHITECTURE.md for editing
# Usage: ./unlock_architecture.sh
# 
# WARNING: Only unlock when architectural changes are PROVEN, not planned.
# After editing, run: ./lock_architecture.sh

chmod 644 LOCKED_ARCHITECTURE.md
echo "UNLOCKED: LOCKED_ARCHITECTURE.md is now editable."
echo "Remember to re-lock after editing: ./lock_architecture.sh"
