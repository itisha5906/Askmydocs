"""
conftest.py — Pytest configuration for AskMyDocs.

Sets PYTHONPATH so all test files can import core/* and config without
needing pip install -e.
"""

import sys
from pathlib import Path

# Make the askmydocs/ directory importable
sys.path.insert(0, str(Path(__file__).parent))
