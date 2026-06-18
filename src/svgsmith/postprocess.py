"""Postprocess raw tracer output into editable SVG.

Turns tracer "path soup" into editable SVG: simplified paths (Douglas-Peucker
over flattened geometry), readable ``<g>`` layers grouped by fill, and a
consolidated palette. Operates on the SVG through a real XML parser
(:mod:`xml.etree.ElementTree`), never regex, and parses path ``d`` data with a
small dedicated parser.

This module exposes an explicit ``simplify_level`` and never calls the verify
loop — #7 owns the loop that varies ``simplify_level`` (avoids a T5/T6 cycle).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

SVG_NS = "http://www.w3.org/2000/svg"
_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|" + _NUMBER.pattern)
_HEX_COLOR = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")

Point = tuple[float, float]


@dataclass(frozen=True)
class PostprocessOptions:
    """Options for :func:`postprocess` (conservative defaults)."""

    simplify_level: float = 1.0  # 0 disables; T6 raises this while SSIM holds
    consolidate_palette: bool = True
    palette_threshold: float = 12.0  # RGB euclidean distance to merge fills
    group: bool = True
    precision: int = 2  # output coordinate decimals
    epsilon_ratio: float = 0.0015  # DP epsilon = level * diagonal * ratio


# A segment keeps its curve type so simplification can preserve curves:
#   ("L", end) | ("C", c1, c2, end) | ("Q", c1, end)
Segment = tuple


@dataclass
class _Subpath:
    start: Point
    segments: list[Segment]
    closed: bool


# --------------------------------------------------------------------------- #
# Path parsing / flattening
# --------------------------------------------------------------------------- #


def _reflect(cx: float, cy: float, ctrl: Point | None) -> Point:
    """Reflect the previous control point about the current point (for S/T)."""
    if ctrl is None:
        return (cx, cy)
    return (2 * cx - ctrl[0], 2 * cy - ctrl[1])


def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, samples: int) -> list[Point]:
    out = []
    for i in range(1, samples + 1):
        t = i / samples
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        out.append((x, y))
    return out


def _quadratic(p0: Point, p1: Point, p2: Point, samples: int) -> list[Point]:
    out = []
    for i in range(1, samples + 1):
        t = i / samples
        u = 1 - t
        x = u**2 * p0[0] + 2 * u * t * p1[0] + t**2 * p2[0]
        y = u**2 * p0[1] + 2 * u * t * p1[1] + t**2 * p2[1]
        out.append((x, y))
    return out


def parse_path(d: str) -> list[_Subpath]:
    """Parse a path ``d`` into typed subpaths, preserving each segment's curve type.

    Supports M/L/H/V/C/S/Q/T/Z in both absolute and relative forms. H/V become
    line segments; S/T expand to C/Q using the reflected control point.
    """
    tokens = _TOKEN.findall(d)
    i = 0
    subpaths: list[_Subpath] = []
    start: Point | None = None
    segments: list[Segment] = []
    closed = False
    cx = cy = sx = sy = 0.0
    prev_cubic_ctrl: Point | None = None
    prev_quad_ctrl: Point | None = None
    command = ""

    def flush(close: bool) -> None:
        nonlocal segments
        if start is not None and segments:
            subpaths.append(_Subpath(start, segments, close))
        segments = []

    def num() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            command = token
            i += 1
            if command in "Zz":
                closed = True
                flush(True)
                cx, cy = sx, sy
                start = (sx, sy)
                prev_cubic_ctrl = prev_quad_ctrl = None
                continue
        relative = command.islower()
        op = command.upper()

        if op == "M":
            flush(closed)
            closed = False
            x, y = num(), num()
            if relative:
                x, y = cx + x, cy + y
            cx, cy = sx, sy = x, y
            start = (x, y)
            command = "l" if relative else "L"  # subsequent pairs are lineto
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif op == "L":
            x, y = num(), num()
            if relative:
                x, y = cx + x, cy + y
            cx, cy = x, y
            segments.append(("L", (x, y)))
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif op == "H":
            x = num()
            cx = cx + x if relative else x
            segments.append(("L", (cx, cy)))
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif op == "V":
            y = num()
            cy = cy + y if relative else y
            segments.append(("L", (cx, cy)))
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif op == "C":
            x1, y1, x2, y2, x, y = num(), num(), num(), num(), num(), num()
            if relative:
                x1, y1, x2, y2, x, y = (
                    cx + x1, cy + y1, cx + x2, cy + y2, cx + x, cy + y,
                )
            segments.append(("C", (x1, y1), (x2, y2), (x, y)))
            prev_cubic_ctrl = (x2, y2)
            prev_quad_ctrl = None
            cx, cy = x, y
        elif op == "S":
            x2, y2, x, y = num(), num(), num(), num()
            if relative:
                x2, y2, x, y = cx + x2, cy + y2, cx + x, cy + y
            c1 = _reflect(cx, cy, prev_cubic_ctrl)
            segments.append(("C", c1, (x2, y2), (x, y)))
            prev_cubic_ctrl = (x2, y2)
            prev_quad_ctrl = None
            cx, cy = x, y
        elif op == "Q":
            x1, y1, x, y = num(), num(), num(), num()
            if relative:
                x1, y1, x, y = cx + x1, cy + y1, cx + x, cy + y
            segments.append(("Q", (x1, y1), (x, y)))
            prev_quad_ctrl = (x1, y1)
            prev_cubic_ctrl = None
            cx, cy = x, y
        elif op == "T":
            x, y = num(), num()
            if relative:
                x, y = cx + x, cy + y
            c1 = _reflect(cx, cy, prev_quad_ctrl)
            segments.append(("Q", c1, (x, y)))
            prev_quad_ctrl = c1
            prev_cubic_ctrl = None
            cx, cy = x, y
        else:
            i += 1  # unknown token; skip defensively

    flush(closed)
    return subpaths


def _flatten_segment(p0: Point, seg: Segment, samples: int) -> list[Point]:
    kind = seg[0]
    if kind == "C":
        return _cubic(p0, seg[1], seg[2], seg[3], samples)
    if kind == "Q":
        return _quadratic(p0, seg[1], seg[2], samples)
    return [seg[-1]]


def _subpath_points(sub: _Subpath, samples: int) -> list[Point]:
    points = [sub.start]
    p = sub.start
    for seg in sub.segments:
        points.extend(_flatten_segment(p, seg, samples))
        p = seg[-1]
    if sub.closed:
        points.append(sub.start)
    return points


def _is_linear_segment(p0: Point, seg: Segment, epsilon: float) -> bool:
    """True for a line, or a curve whose control points hug its chord."""
    if seg[0] == "L":
        return True
    end = seg[-1]
    return all(_perpendicular_distance(c, p0, end) <= epsilon for c in seg[1:-1])


def simplify_subpath(sub: _Subpath, epsilon: float) -> _Subpath:
    """Collapse runs of effectively-linear segments with Douglas-Peucker.

    Genuinely-curved segments are kept verbatim, so the output is never coarser
    than its input — only redundant straight runs (common in tracer output where
    one edge is split into many collinear segments) are reduced.
    """
    if epsilon <= 0 or not sub.segments:
        return sub

    flags = []
    p = sub.start
    for seg in sub.segments:
        flags.append(_is_linear_segment(p, seg, epsilon))
        p = seg[-1]

    result: list[Segment] = []
    p = sub.start
    i = 0
    n = len(sub.segments)
    while i < n:
        if flags[i]:
            run = [p]
            j = i
            while j < n and flags[j]:
                run.append(sub.segments[j][-1])
                j += 1
            for point in douglas_peucker(run, epsilon)[1:]:
                result.append(("L", point))
            p = run[-1]
            i = j
        else:
            result.append(sub.segments[i])
            p = sub.segments[i][-1]
            i += 1
    return _Subpath(sub.start, result, sub.closed)


# --------------------------------------------------------------------------- #
# Douglas-Peucker
# --------------------------------------------------------------------------- #


def douglas_peucker(points: list[Point], epsilon: float) -> list[Point]:
    """Simplify a polyline, keeping its endpoints and shape within epsilon."""
    if len(points) < 3 or epsilon <= 0:
        return points
    start, end = points[0], points[-1]
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perpendicular_distance(points[i], start, end)
        if d > dmax:
            dmax, index = d, i
    if dmax > epsilon:
        left = douglas_peucker(points[: index + 1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _perpendicular_distance(pt: Point, a: Point, b: Point) -> float:
    if a == b:
        return ((pt[0] - a[0]) ** 2 + (pt[1] - a[1]) ** 2) ** 0.5
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    return abs(dy * pt[0] - dx * pt[1] + b[0] * a[1] - b[1] * a[0]) / length


# --------------------------------------------------------------------------- #
# Color helpers
# --------------------------------------------------------------------------- #


def _normalize_hex(color: str) -> str | None:
    match = _HEX_COLOR.fullmatch(color.strip())
    if not match:
        return None
    body = match.group(1)
    if len(body) == 3:
        body = "".join(ch * 2 for ch in body)
    return "#" + body.lower()


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _color_distance(a: str, b: str) -> float:
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return ((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5


def _consolidate(colors: list[str], threshold: float) -> dict[str, str]:
    """Map each color to a representative; near-identical colors share one."""
    representatives: list[str] = []
    mapping: dict[str, str] = {}
    for color in colors:
        match = next(
            (rep for rep in representatives if _color_distance(color, rep) <= threshold),
            None,
        )
        if match is None:
            representatives.append(color)
            mapping[color] = color
        else:
            mapping[color] = match
    return mapping


# --------------------------------------------------------------------------- #
# SVG traversal
# --------------------------------------------------------------------------- #


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _effective_fill(attrs: dict[str, str]) -> str | None:
    if "fill" in attrs:
        return attrs["fill"]
    style = attrs.get("style", "")
    for decl in style.split(";"):
        name, _, value = decl.partition(":")
        if name.strip() == "fill":
            return value.strip()
    return None


def _collect_paths(root: ET.Element) -> list[dict]:
    """Walk the tree, returning each path with its fill and accumulated transform."""
    collected: list[dict] = []

    def walk(element: ET.Element, fill: str | None, transforms: list[str]) -> None:
        local = _local(element.tag)
        attrs = element.attrib
        own_fill = _effective_fill(attrs)
        current_fill = own_fill if own_fill is not None else fill
        current_transforms = transforms + ([attrs["transform"]] if "transform" in attrs else [])
        if local == "path" and attrs.get("d"):
            collected.append(
                {
                    "d": attrs["d"],
                    "fill": current_fill,
                    "transform": " ".join(current_transforms),
                }
            )
        for child in element:
            walk(child, current_fill, current_transforms)

    walk(root, None, [])
    return collected


def _format_number(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _emit_d(subpaths: list[_Subpath], precision: int) -> str:
    def fmt(p: Point) -> str:
        return f"{_format_number(p[0], precision)} {_format_number(p[1], precision)}"

    chunks: list[str] = []
    for sub in subpaths:
        if not sub.segments:
            continue
        chunks.append(f"M{fmt(sub.start)}")
        for seg in sub.segments:
            if seg[0] == "C":
                chunks.append(f"C{fmt(seg[1])} {fmt(seg[2])} {fmt(seg[3])}")
            elif seg[0] == "Q":
                chunks.append(f"Q{fmt(seg[1])} {fmt(seg[2])}")
            else:
                chunks.append(f"L{fmt(seg[1])}")
        if sub.closed:
            chunks.append("Z")
    return "".join(chunks)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def postprocess(svg_str: str, opts: PostprocessOptions | None = None) -> str:
    """Return an editable SVG: simplified paths, grouped layers, merged palette."""
    opts = opts or PostprocessOptions()
    source = ET.fromstring(svg_str)
    geom_width, geom_height = _geometry_size(source)
    diagonal = (geom_width**2 + geom_height**2) ** 0.5
    epsilon = opts.simplify_level * diagonal * opts.epsilon_ratio

    paths = _collect_paths(source)

    # Palette consolidation across all path fills.
    if opts.consolidate_palette:
        hexes = [h for h in (_normalize_hex(p["fill"] or "") for p in paths) if h]
        mapping = _consolidate(list(dict.fromkeys(hexes)), opts.palette_threshold)
        for path in paths:
            norm = _normalize_hex(path["fill"] or "")
            if norm:
                path["fill"] = mapping.get(norm, norm)

    # Simplify each path's geometry. Genuinely-curved segments are preserved
    # verbatim; only runs of effectively-linear segments are reduced (tracer
    # output commonly splits one straight edge into many collinear segments).
    # So the output is never coarser than its input.
    if opts.simplify_level > 0:
        for path in paths:
            simplified = [simplify_subpath(s, epsilon) for s in parse_path(path["d"])]
            new_d = _emit_d(simplified, opts.precision)
            if new_d:
                path["d"] = new_d

    return _build_svg(source, paths, opts.group)


def _length(value: str | None) -> float:
    """Parse the leading number from an SVG length (drops units like px/%)."""
    if not value:
        return 0.0
    match = re.match(r"[-+]?(?:\d*\.\d+|\d+\.?)", value.strip())
    return float(match.group()) if match else 0.0


def _geometry_size(root: ET.Element) -> tuple[float, float]:
    """Coordinate-space size used for epsilon scaling.

    Uses the ``viewBox`` extents when present (path coordinates live in that
    space); otherwise the root ``width``/``height``. This is independent of the
    rendered ``width``/``height`` attributes, which are preserved verbatim.
    """
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(v) for v in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]
    return _length(root.get("width")), _length(root.get("height"))


def _build_svg(source: ET.Element, paths: list[dict], group: bool) -> str:
    ET.register_namespace("", SVG_NS)
    svg = ET.Element(f"{{{SVG_NS}}}svg", {"version": source.get("version", "1.1")})
    # Preserve the root sizing attributes exactly as authored.
    for attr in ("viewBox", "width", "height"):
        value = source.get(attr)
        if value is not None:
            svg.set(attr, value)

    if group:
        # Group *consecutive* same-fill paths into a <g>, preserving the tracer's
        # original path order. Grouping by color globally would collect a color's
        # paths from all depths together and reorder them, breaking the stacked
        # paint order (later = on top) so light fills cover dark ones and dark
        # regions vanish. Consecutive-run grouping keeps z-order exact while still
        # producing editable, fill-labelled layers.
        runs: list[tuple[str, list[dict]]] = []
        for path in paths:
            fill = path["fill"] or "none"
            if runs and runs[-1][0] == fill:
                runs[-1][1].append(path)
            else:
                runs.append((fill, [path]))
        for index, (fill, members) in enumerate(runs, start=1):
            slug = fill.lstrip("#") if fill.startswith("#") else fill
            layer = ET.SubElement(
                svg, f"{{{SVG_NS}}}g", {"id": f"layer-{index:02d}-{slug}", "fill": fill}
            )
            for path in members:
                _append_path(layer, path, include_fill=False)
    else:
        for path in paths:
            _append_path(svg, path, include_fill=True)

    body = ET.tostring(svg, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body


def _append_path(parent: ET.Element, path: dict, include_fill: bool) -> None:
    attrs = {"d": path["d"]}
    if include_fill and path["fill"]:
        attrs["fill"] = path["fill"]
    if path["transform"]:
        attrs["transform"] = path["transform"]
    ET.SubElement(parent, f"{{{SVG_NS}}}path", attrs)


def count_path_points(svg_str: str) -> int:
    """Total on-curve anchor points (start + one per segment) across all paths."""
    root = ET.fromstring(svg_str)
    return sum(
        1 + len(sub.segments)
        for p in _collect_paths(root)
        for sub in parse_path(p["d"])
    )


def svg_bbox(svg_str: str, samples: int = 18) -> tuple[float, float, float, float] | None:
    """Overall geometry bounding box ``(minx, miny, maxx, maxy)``, or None."""
    root = ET.fromstring(svg_str)
    xs: list[float] = []
    ys: list[float] = []
    for path in _collect_paths(root):
        for sub in parse_path(path["d"]):
            for x, y in _subpath_points(sub, samples):
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)
