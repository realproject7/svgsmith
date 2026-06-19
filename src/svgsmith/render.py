"""Rasterize SVG back to PNG.

The inverse of the vectorizer: render an SVG to a PNG bitmap via CairoSVG (already
a dependency, used by the verify loop). Useful for previews, thumbnails, and
round-tripping. Sizing precedence: explicit ``--scale`` or ``--width``/``--height``,
otherwise the SVG's intrinsic size (its ``viewBox`` extents when present).
"""

from __future__ import annotations

import re

import cairosvg

# The input SVG may be arbitrary/user-supplied, so we do NOT run a full XML parse
# over it (XXE / entity-expansion surface). We only scrape the root <svg> tag's
# size attributes with a regex; CairoSVG handles the actual (safe) rendering.
_SVG_TAG = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
_ATTR = lambda name: re.compile(rf'{name}\s*=\s*"([^"]*)"', re.IGNORECASE)  # noqa: E731


def _viewbox_size(svg: str) -> tuple[int, int] | None:
    """Intrinsic pixel size from the root viewBox (or width/height), if any."""
    tag_match = _SVG_TAG.search(svg)
    if not tag_match:
        return None
    tag = tag_match.group(0)

    vb = _ATTR("viewBox").search(tag)
    if vb:
        parts = [p for p in re.split(r"[,\s]+", vb.group(1).strip()) if p]
        if len(parts) == 4:
            try:
                return round(float(parts[2])), round(float(parts[3]))
            except ValueError:
                pass

    def _len(name: str) -> float:
        m = _ATTR(name).search(tag)
        if not m:
            return 0.0
        num = re.match(r"[-+]?(?:\d*\.\d+|\d+\.?)", m.group(1).strip())
        return float(num.group()) if num else 0.0

    w, h = _len("width"), _len("height")
    return (round(w), round(h)) if w and h else None


def rasterize(
    svg_path: str,
    out_path: str,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
    background: str | None = None,
) -> str:
    """Render the SVG at ``svg_path`` to a PNG at ``out_path``; return ``out_path``."""
    with open(svg_path, encoding="utf-8") as handle:
        svg = handle.read()

    kwargs: dict = {}
    if scale is not None:
        kwargs["scale"] = scale
    if width is not None:
        kwargs["output_width"] = width
    if height is not None:
        kwargs["output_height"] = height
    if background is not None:
        kwargs["background_color"] = background
    # Give CairoSVG an explicit size when none was requested — a responsive SVG
    # (viewBox, no fixed width/height) would otherwise rasterize at a default size.
    if not kwargs.get("scale") and "output_width" not in kwargs and "output_height" not in kwargs:
        size = _viewbox_size(svg)
        if size:
            kwargs["output_width"], kwargs["output_height"] = size

    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=out_path, **kwargs)
    return out_path
