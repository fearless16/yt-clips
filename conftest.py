"""
conftest.py — Root conftest for Face OS tests.

Adds the project root to sys.path so that face_os can be imported.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
