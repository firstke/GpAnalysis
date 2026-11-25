"""Command-line entry point for gp."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__, greet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gp", description="Print a friendly greeting.")
    parser.add_argument(
        "-n",
        "--name",
        default="world",
        help="Name to greet (default: world)",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(greet(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
