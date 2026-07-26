#!/usr/bin/env python
"""Evaluate closed-loop ObjectNav performance.

Not implemented yet: this lands in Phase 13 of checklist.md. The file
exists so the repository layout and the entry-point names are fixed from the
start, and so nothing downstream has to guess what a script will be called.
"""

from __future__ import annotations

import sys

PHASE = "13"


def main() -> int:
    print(
        "scripts/evaluate_navigation.py is a placeholder for Phase " + PHASE + ".\n"
        "Phase 1 provides scripts/run_episode.py; see checklist.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
