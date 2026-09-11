"""Compatibility facade for the Agent Mail intake CLI owner."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "inbox_collector" / "inbox_poll.py"


def main() -> None:
    """Execute the unchanged inbox collector CLI with the caller's argv."""
    source_dir = str(SOURCE.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
