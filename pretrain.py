"""Backward-compatible training entry point.

Prefer `python -m omnisleep.cli` after installing the package, or run this file
directly from the repository root for the original workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omnisleep.cli import main


if __name__ == "__main__":
    main()
