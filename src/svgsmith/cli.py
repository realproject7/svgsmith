"""Command-line interface for svgsmith.

This module wires up the ``svgsmith`` console entrypoint and its ``convert``
subcommand. ``convert`` runs the full pipeline (classify → preprocess → trace →
postprocess → verify) and, with ``--report json``, prints the canonical JSON
report to stdout — and nothing else: all logs and errors go to stderr.

Exit codes:
    0  success (similarity met the --quality threshold)
    2  an SVG was produced but its similarity is below --quality
    1  hard error (could not produce output)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from svgsmith import __version__
from svgsmith.pipeline import MODES, ConvertOptions, convert

# Process exit codes (documented in the module docstring and --help epilog).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BELOW_THRESHOLD = 2


def _log(message: str) -> None:
    """Write a human-facing line to stderr (stdout is reserved for the report)."""
    print(message, file=sys.stderr)


def _convert(args: argparse.Namespace) -> int:
    """Handle ``svgsmith convert`` once arguments have been parsed."""
    if not args.input:
        _log("error: an input image path is required")
        return EXIT_ERROR

    # Validate bounds before any conversion work so invalid input fails fast
    # with a clear message instead of an opaque downstream error.
    if not 0.0 <= args.quality <= 1.0:
        _log(f"error: --quality must be between 0 and 1 (got {args.quality})")
        return EXIT_ERROR
    if args.max_iters < 1:
        _log(f"error: --max-iters must be at least 1 (got {args.max_iters})")
        return EXIT_ERROR

    opts = ConvertOptions(
        mode=args.mode,
        quality=args.quality,
        max_iters=args.max_iters,
        editable=args.editable,
        smooth=args.smooth,
        uniform_outline=args.uniform_outline,
        solid_background=args.solid_background,
        detail=args.detail,
        out=args.out,
    )

    try:
        svg, report = convert(args.input, opts)
    except FileNotFoundError:
        _log(f"error: input not found: {args.input}")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - surface any failure as a hard error
        _log(f"error: conversion failed: {exc}")
        return EXIT_ERROR

    try:
        with open(report.output, "w", encoding="utf-8") as handle:
            handle.write(svg)
    except OSError as exc:
        _log(f"error: could not write output {report.output!r}: {exc}")
        return EXIT_ERROR

    if args.report == "json":
        # stdout carries the report and nothing else.
        print(report.to_json())
    else:
        _log(
            f"wrote {report.output} "
            f"(mode={report.mode_used}, similarity={report.similarity:.3f}, "
            f"passed={report.passed_threshold})"
        )

    for warning in report.warnings:
        _log(f"warning: {warning}")

    return EXIT_OK if report.passed_threshold else EXIT_BELOW_THRESHOLD


def _rasterize(args: argparse.Namespace) -> int:
    """Handle ``svgsmith rasterize`` — render an SVG to PNG."""
    if not args.input:
        _log("error: an input SVG path is required")
        return EXIT_ERROR
    from pathlib import Path

    from svgsmith.render import rasterize as render_png

    out = args.out or str(Path(args.input).with_suffix(".png"))
    try:
        render_png(
            args.input,
            out,
            width=args.width,
            height=args.height,
            scale=args.scale,
            background=args.background,
        )
    except FileNotFoundError:
        _log(f"error: input not found: {args.input}")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - surface any failure as a hard error
        _log(f"error: rasterize failed: {exc}")
        return EXIT_ERROR
    _log(f"wrote {out}")
    return EXIT_OK


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
        epilog=(
            "exit codes: 0 success (similarity >= --quality); "
            "2 SVG produced but below --quality; 1 hard error."
        ),
    )
    convert.add_argument(
        "input",
        nargs="?",
        help="Path to the input raster image (PNG, JPEG, …).",
    )
    convert.add_argument(
        "--mode",
        choices=list(MODES),
        default="auto",
        help="Conversion mode: auto|binary|color|pixel (default: auto).",
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
        help=(
            "Emit editable, grouped/simplified SVG (default: on). "
            "Use --no-editable to skip postprocessing and emit the raw traced SVG."
        ),
    )
    convert.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Curve-refit color output into smooth, sparse Bezier contours "
            "(default: on). Use --no-smooth to keep the raw traced geometry."
        ),
    )
    convert.add_argument(
        "--uniform-outline",
        action="store_true",
        default=False,
        help=(
            "Force an even-width outline band (color mode). Opt-in: only for "
            "illustrations that already have a dark outline; would add a wrong "
            "border on line art."
        ),
    )
    convert.add_argument(
        "--detail",
        choices=["high", "normal", "clean", "poster"],
        default="normal",
        help=(
            "Color detail dial (default: normal). high = maximum detail; "
            "clean = edge-preserving cleanup, less noise; poster = bold flat graphic, "
            "few colors."
        ),
    )
    convert.add_argument(
        "--solid-background",
        action="store_true",
        default=False,
        help=(
            "Isolate the subject and repaint the background as one clean solid "
            "color, removing texture/grain/specks while keeping subject detail."
        ),
    )
    convert.add_argument(
        "--out",
        default=None,
        help="Output SVG path (default: input path with a .svg extension).",
    )
    convert.add_argument(
        "--report",
        choices=["off", "json"],
        default="off",
        help="Emit a JSON report alongside the SVG (default: off).",
    )
    convert.set_defaults(func=_convert)

    rasterize = subparsers.add_parser(
        "rasterize",
        help="Render an SVG back to a PNG bitmap.",
        description="Rasterize an SVG to PNG (preview, thumbnail, round-trip).",
    )
    rasterize.add_argument("input", nargs="?", help="Path to the input SVG file.")
    rasterize.add_argument(
        "--out", default=None, help="Output PNG path (default: input with a .png extension)."
    )
    rasterize.add_argument("--width", type=int, default=None, help="Output width in px.")
    rasterize.add_argument("--height", type=int, default=None, help="Output height in px.")
    rasterize.add_argument(
        "--scale", type=float, default=None, help="Scale factor over the intrinsic size."
    )
    rasterize.add_argument(
        "--background",
        default=None,
        help="Background color (e.g. white, #ffffff). Default: transparent.",
    )
    rasterize.set_defaults(func=_rasterize)

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
