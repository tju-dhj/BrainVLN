#!/usr/bin/env python3
"""AllenAct entry point with RobustNav-style visual-corruption flags."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _inject_corruptions(argv: List[str]) -> List[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-vc", "--visual-corruption", default=None)
    parser.add_argument("-vs", "--visual-severity", type=int, default=5)
    known, rest = parser.parse_known_args(argv)
    if known.visual_corruption:
        from spikingnav.corruptions import canonicalize
        name = canonicalize(known.visual_corruption)
        # Stash on env for experiment constructors that read it.
        import os

        os.environ["SPIKINGNAV_CORRUPTION"] = name
        os.environ["SPIKINGNAV_SEVERITY"] = str(known.visual_severity)
    return rest


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = _inject_corruptions(argv)
    try:
        from allenact.main import get_args, main as allenact_main
    except ImportError as exc:
        raise SystemExit(
            "AllenAct is not installed. See README.md for the simulation stack.\n"
            f"Original error: {exc}"
        )

    # Re-dispatch to AllenAct with remaining flags.
    sys.argv = [sys.argv[0]] + rest
    allenact_main()


if __name__ == "__main__":
    main()
