"""Command-line interface for svgsmith.

This module wires up the ``svgsmith`` console entrypoint and its ``convert``
subcommand. The flag contract here is binding for the rest of the project; the
conversion engine itself lands in later tickets (T2–T7), so an actual
``convert`` invocation raises :class:`NotImplementedError` for now while
``--help`` still short-circuits and exits cleanly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from svgsmith import __version__

# Reserved process exit codes. Later tickets (see #8) map failures onto these;
# argparse already emits EXIT_USAGE for malformed command lines.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _convert(args: argparse.Namespace) -> int:
    """Handle ``svgsmith convert`` once arguments have been parsed."""
    raise NotImplementedError(
        "svgsmith convert is not wired up yet — the conversion engine arrives "
        "in a later ticket. Run `svgsmith convert --help` to see the flags."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="svgsmith",
        description="Convert raster images into clean, editable SVG.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    convert = subparsers.add_parser(
        "convert",
        help="Convert a raster image into SVG.",
        description="Convert a raster image into clean, editable SVG.",
    )
    convert.add_argument(
        "input",
        nargs="?",
        help="Path to the input raster image (PNG, JPEG, …).",
    )
    convert.add_argument(
        "--mode",
        default="auto",
        help="Conversion mode (default: auto).",
    )
    convert.add_argument(
        "--quality",
        type=float,
        default=0.9,
        help="Target fidelity in [0, 1] (default: 0.9).",
    )
    convert.add_argument(
        "--max-iters",
        type=int,
        default=4,
        help="Maximum verify/refine iterations (default: 4).",
    )
    convert.add_argument(
        "--editable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit editable, human-friendly SVG (default: on).",
    )
    convert.add_argument(
        "--out",
        default=None,
        help="Output path; defaults to stdout when omitted.",
    )
    convert.add_argument(
        "--report",
        choices=["off", "json"],
        default="off",
        help="Emit a JSON report alongside the SVG (default: off).",
    )
    convert.set_defaults(func=_convert)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``svgsmith`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
