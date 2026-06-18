"""Curve refitting: smooth the wobble out of traced contours.

VTracer fits splines to *quantized pixel boundaries*, so contour edges inherit a
fine staircase wobble — outlines look subtly jagged and busy regions look faceted.
This pass re-fits each closed contour so genuinely-curved arcs become smooth while
straight edges and sharp corners stay crisp:

1. flatten the contour to a dense polyline and resample it evenly,
2. detect corners (sharp turning angles) and split the contour into arcs,
3. per arc: if it is effectively straight, keep one line segment (preserves
   stripe/rectangle edges); otherwise Gaussian-smooth it and refit a Catmull-Rom
   cubic spline (preserves the corners as anchors).

Small/thin features (below ``min_perim``) are left untouched so detail survives.
Operates purely on path geometry; colors, grouping, and order are unchanged.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from svgsmith.postprocess import SVG_NS, _subpath_points, parse_path

Point = tuple[float, float]


def _perim(pts: list[Point]) -> float:
    return sum(
        math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))
    )


def _resample(pts: list[Point], n: int) -> list[Point]:
    """Resample a closed polyline to ``n`` evenly-spaced points."""
    closed = pts + [pts[0]]
    seglen = [math.dist(closed[i], closed[i + 1]) for i in range(len(pts))]
    total = sum(seglen)
    if total == 0 or n < 4:
        return pts
    cum = [0.0]
    for s in seglen:
        cum.append(cum[-1] + s)
    out: list[Point] = []
    step = total / n
    j = 0
    for k in range(n):
        target = k * step
        while j < len(seglen) and cum[j + 1] < target:
            j += 1
        if j >= len(seglen):
            out.append(pts[0])
            continue
        t = (target - cum[j]) / seglen[j] if seglen[j] else 0.0
        a, b = closed[j], closed[j + 1]
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _corner_indices(pts: list[Point], angle_deg: float) -> list[int]:
    """Indices whose turning angle is sharper than ``angle_deg`` (corners)."""
    n = len(pts)
    thr = math.cos(math.radians(180 - angle_deg))
    corners = []
    for i in range(n):
        ax, ay = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        bx, by = pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na < 1e-6 or nb < 1e-6:
            continue
        if (ax * bx + ay * by) / (na * nb) < thr:
            corners.append(i)
    return corners


def _max_deviation(arc: list[Point]) -> float:
    """Max perpendicular distance of arc points from the arc's chord."""
    a, b = arc[0], arc[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return max(abs((p[0] - a[0]) * (-dy) + (p[1] - a[1]) * dx) / length for p in arc)


def _gaussian_arc(arc: list[Point], sigma: float) -> list[Point]:
    """Gaussian-smooth an open arc, pinning the endpoints (corners)."""
    if sigma <= 0 or len(arc) < 5:
        return arc
    r = max(1, int(sigma * 3))
    kernel = [math.exp(-0.5 * ((i - r) / sigma) ** 2) for i in range(2 * r + 1)]
    ksum = sum(kernel)
    kernel = [k / ksum for k in kernel]
    padded = [arc[0]] * r + arc + [arc[-1]] * r
    out: list[Point] = []
    for i in range(len(arc)):
        sx = sy = 0.0
        for j, k in enumerate(kernel):
            sx += padded[i + j][0] * k
            sy += padded[i + j][1] * k
        out.append((sx, sy))
    out[0], out[-1] = arc[0], arc[-1]
    return out


def _catmull_rom(arc: list[Point]) -> list[tuple]:
    """Open Catmull-Rom -> cubic bezier segments through ``arc``."""
    n = len(arc)
    segs = []
    for i in range(n - 1):
        p0 = arc[max(0, i - 1)]
        p1, p2 = arc[i], arc[i + 1]
        p3 = arc[min(n - 1, i + 2)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        segs.append(("C", c1, c2, p2))
    return segs


def _smooth_d(
    d: str,
    *,
    min_perim: float,
    sigma: float,
    pts_per_100: float,
    corner_deg: float,
    straight_tol: float,
    samples: int,
    precision: int,
) -> str:
    chunks: list[str] = []

    def fmt(p: Point) -> str:
        return f"{p[0]:.{precision}f} {p[1]:.{precision}f}"

    def emit_raw(sub) -> None:
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

    for sub in parse_path(d):
        if not sub.closed or len(sub.segments) < 3:
            emit_raw(sub)
            continue
        pts = _subpath_points(sub, samples)[:-1]  # drop closing duplicate
        perim = _perim(pts)
        if perim < min_perim:
            emit_raw(sub)  # tiny/thin feature: keep crisp detail
            continue
        n = max(12, int(perim / 100 * pts_per_100))
        rs = _resample(pts, n)
        corners = sorted(set(_corner_indices(rs, corner_deg)))
        if len(corners) < 2:
            corners = [0, len(rs) // 2]
        chunks.append(f"M{fmt(rs[corners[0]])}")
        for j in range(len(corners)):
            a, b = corners[j], corners[(j + 1) % len(corners)]
            if b > a:
                idx = list(range(a, b + 1))
            else:
                idx = list(range(a, len(rs))) + list(range(0, b + 1))
            arc = [rs[k] for k in idx]
            if len(arc) < 3 or _max_deviation(arc) <= straight_tol:
                chunks.append(f"L{fmt(arc[-1])}")  # straight edge stays crisp
            else:
                for _, c1, c2, end in _catmull_rom(_gaussian_arc(arc, sigma)):
                    chunks.append(f"C{fmt(c1)} {fmt(c2)} {fmt(end)}")
        chunks.append("Z")
    return "".join(chunks)


def smooth_svg(
    svg: str,
    *,
    min_perim: float = 120.0,
    sigma: float = 1.2,
    pts_per_100: float = 10.0,
    corner_deg: float = 40.0,
    straight_tol: float = 2.0,
    samples: int = 6,
    precision: int = 2,
) -> str:
    """Return ``svg`` with every path's geometry curve-refit for smoothness."""
    root = ET.fromstring(svg)
    for path in root.iter(f"{{{SVG_NS}}}path"):
        d = path.get("d")
        if d:
            path.set(
                "d",
                _smooth_d(
                    d,
                    min_perim=min_perim,
                    sigma=sigma,
                    pts_per_100=pts_per_100,
                    corner_deg=corner_deg,
                    straight_tol=straight_tol,
                    samples=samples,
                    precision=precision,
                ),
            )
    body = ET.tostring(root, encoding="unicode")
    if not body.startswith("<?xml"):
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + body
    return body
