"""Compatibility wrapper for `python scripts/inseason_trades.py projections`."""

from __future__ import annotations

import sys

from draft_buddy.inseason.cli import main


if __name__ == "__main__":
    sys.exit(main(["projections", *sys.argv[1:]]))
