"""Curve refitting: smooth the wobble out of traced contours.

VTracer approximates curved edges with many short straight segments following a
quantized pixel boundary, so contours look faceted/jagged (measured: ~20% of
segments are curves vs ~87% in professional output). This pass re-fits each closed
contour so genuinely-curved arcs become smooth, sparse cubic Béziers while straight
edges and sharp corners stay crisp:

1. flatten the contour to a dense polyline and resample it evenly,
2. detect corners (sharp turning angles) and split the contour into arcs,
3. per arc: a near-straight arc keeps one line segment (preserves stripe/rectangle
   edges); a curved arc is fit with **Schneider least-squares cubic Bézier fitting**
   (recursive: least-squares solve → Newton-Raphson reparameterize → split at max
   error). Node count is governed by a canvas-relative error tolerance, so the output
   is smooth *and* sparse (unlike a dense per-point spline).

Small/thin features (below ``min_perim``) are left untouched so detail survives.
Operates purely on path geometry; colors, grouping, and order are unchanged.
Thresholds are canvas-relative (fractions of the viewBox diagonal) to generalize
across image sizes.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from svgsmith.postprocess import SVG_NS, _subpath_points, parse_path

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# Polyline helpers
# --------------------------------------------------------------------------- #


def _perim(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)))


def _area(pts: np.ndarray) -> float:
    """Absolute polygon area (shoelace) of a closed point ring."""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to ``n`` evenly-spaced points."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = seg.sum()
    if total == 0 or n < 4:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([xs, ys])


def _corner_indices(pts: np.ndarray, angle_deg: float) -> list[int]:
    """Indices whose turning angle is sharper than ``angle_deg`` (corners)."""
    n = len(pts)
    thr = math.cos(math.radians(180 - angle_deg))
    prev = pts - np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0) - pts
    corners = []
    for i in range(n):
        na = np.hypot(*prev[i])
        nb = np.hypot(*nxt[i])
        if na < 1e-6 or nb < 1e-6:
            continue
        if float(np.dot(prev[i], nxt[i])) / (na * nb) < thr:
            corners.append(i)
    return corners


def _inflection_indices(pts: np.ndarray, min_turn: float) -> list[int]:
    """Indices where the signed curvature changes sign (contour reversals).

    Splitting an arc at its inflections keeps each piece single-curvature, so the
    Bézier fit cannot average across a reversal — this is what preserves concave
    features like a mouth 'W' (smooth curvature flips, but no sharp corner, so
    corner detection alone misses them). Only reversals with meaningful turn on at
    least one side count, so gentle wiggle on a smooth edge does not over-split.
    """
    n = len(pts)
    prev = pts - np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0) - pts
    cross = prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0]  # signed curvature proxy
    out = []
    for i in range(n):
        c0, c1 = cross[i], cross[(i + 1) % n]
        if c0 * c1 < 0 and (abs(c0) > min_turn or abs(c1) > min_turn):
            out.append((i + 1) % n)
    return out


def _max_deviation(arc: np.ndarray) -> float:
    """Max perpendicular distance of arc points from the arc's chord."""
    a, b = arc[0], arc[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return float(np.max(np.linalg.norm(arc - a, axis=1)))
    return float(np.max(np.abs((arc[:, 0] - a[0]) * (-dy) + (arc[:, 1] - a[1]) * dx)) / length)


def _gaussian(arc: np.ndarray, sigma: float) -> np.ndarray:
    """Light Gaussian denoise of an open arc, pinning the endpoints."""
    if sigma <= 0 or len(arc) < 5:
        return arc
    r = max(1, int(sigma * 3))
    k = np.exp(-0.5 * ((np.arange(2 * r + 1) - r) / sigma) ** 2)
    k /= k.sum()
    pad = np.vstack([np.repeat(arc[:1], r, 0), arc, np.repeat(arc[-1:], r, 0)])
    sx = np.convolve(pad[:, 0], k, "same")[r:-r]
    sy = np.convolve(pad[:, 1], k, "same")[r:-r]
    out = np.column_stack([sx, sy])
    out[0], out[-1] = arc[0], arc[-1]
    return out


# --------------------------------------------------------------------------- #
# Schneider least-squares cubic Bézier fitting (Graphics Gems, 1990)
# --------------------------------------------------------------------------- #


def _bezier_point(ctrl: np.ndarray, t: float) -> np.ndarray:
    u = 1 - t
    return (u**3) * ctrl[0] + 3 * (u**2) * t * ctrl[1] + 3 * u * (t**2) * ctrl[2] + (t**3) * ctrl[3]


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(pts))


def _generate_bezier(
    pts: np.ndarray, u: np.ndarray, t_left: np.ndarray, t_right: np.ndarray
) -> np.ndarray:
    p0, p3 = pts[0], pts[-1]
    a = np.zeros((len(pts), 2, 2))
    a[:, 0] = t_left * (3 * u * (1 - u) ** 2)[:, None]
    a[:, 1] = t_right * (3 * u**2 * (1 - u))[:, None]
    c = np.zeros((2, 2))
    x = np.zeros(2)
    for i in range(len(pts)):
        c[0, 0] += a[i, 0] @ a[i, 0]
        c[0, 1] += a[i, 0] @ a[i, 1]
        c[1, 1] += a[i, 1] @ a[i, 1]
        ui = u[i]
        # contribution with c1=p0, c2=p3 as baseline
        base = ((1 - ui) ** 3 + 3 * (1 - ui) ** 2 * ui) * p0 + (3 * (1 - ui) * ui**2 + ui**3) * p3
        tmp = pts[i] - base
        x[0] += a[i, 0] @ tmp
        x[1] += a[i, 1] @ tmp
    c[1, 0] = c[0, 1]
    det = c[0, 0] * c[1, 1] - c[1, 0] * c[0, 1]
    if abs(det) > 1e-12:
        alpha_l = (x[0] * c[1, 1] - x[1] * c[0, 1]) / det
        alpha_r = (c[0, 0] * x[1] - c[0, 1] * x[0]) / det
    else:
        alpha_l = alpha_r = 0.0
    seg_len = np.linalg.norm(p3 - p0)
    if alpha_l < 1e-6 or alpha_r < 1e-6:
        alpha_l = alpha_r = seg_len / 3.0  # Wu/Barsky heuristic fallback
    return np.array([p0, p0 + t_left * alpha_l, p3 + t_right * alpha_r, p3])


def _max_error(pts: np.ndarray, u: np.ndarray, ctrl: np.ndarray) -> tuple[float, int]:
    worst, split = 0.0, len(pts) // 2
    for i in range(len(pts)):
        d = _bezier_point(ctrl, u[i]) - pts[i]
        e = float(d @ d)
        if e > worst:
            worst, split = e, i
    return worst, split


def _reparameterize(pts: np.ndarray, u: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    out = u.copy()
    for i in range(len(pts)):
        t = u[i]
        v = 1 - t
        q = _bezier_point(ctrl, t)
        d1 = (
            3 * v**2 * (ctrl[1] - ctrl[0])
            + 6 * v * t * (ctrl[2] - ctrl[1])
            + 3 * t**2 * (ctrl[3] - ctrl[2])
        )
        d2 = 6 * v * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])
        diff = q - pts[i]
        num = diff @ d1
        den = d1 @ d1 + diff @ d2
        if abs(den) > 1e-12:
            out[i] = t - num / den
    return np.clip(out, 0.0, 1.0)


def _fit_cubic(
    pts: np.ndarray, t_left: np.ndarray, t_right: np.ndarray, tol: float, depth: int = 0
) -> list[np.ndarray]:
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        d = np.linalg.norm(pts[1] - pts[0]) / 3.0
        return [np.array([pts[0], pts[0] + t_left * d, pts[1] + t_right * d, pts[1]])]
    u = _chord_params(pts)
    ctrl = _generate_bezier(pts, u, t_left, t_right)
    err, split = _max_error(pts, u, ctrl)
    if err < tol:
        return [ctrl]
    if err < tol * tol and depth < 16:
        for _ in range(4):
            u = _reparameterize(pts, u, ctrl)
            ctrl = _generate_bezier(pts, u, t_left, t_right)
            err, split = _max_error(pts, u, ctrl)
            if err < tol:
                return [ctrl]
    if depth >= 16 or split <= 0 or split >= len(pts) - 1:
        return [ctrl]  # give up splitting; accept current fit
    center = pts[split - 1] - pts[split + 1]
    nrm = np.hypot(*center)
    t_center = center / nrm if nrm > 1e-6 else np.zeros(2)
    left = _fit_cubic(pts[: split + 1], t_left, t_center, tol, depth + 1)
    right = _fit_cubic(pts[split:], -t_center, t_right, tol, depth + 1)
    return left + right


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.hypot(*v)
    return v / n if n > 1e-6 else np.zeros(2)


def _axis_snap(start: np.ndarray, end: np.ndarray, snap_deg: float, max_drift: float):
    """Snap an already-straight segment's direction to the nearest 0/45/90/135 axis.

    VTracer copies straight-run endpoints verbatim, so a "horizontal" edge inherits
    a few degrees of pixel tilt and reads as 1px stair-creep. If the segment from
    ``start`` to ``end`` lies within ``snap_deg`` of an exact axis, rotate ``end``
    onto that axis about ``start`` (preserving the segment length so the contour does
    not shrink). Returns ``(snapped_end, drift)`` where ``drift`` is how far the
    endpoint moved; the caller refuses the snap if it would exceed ``max_drift`` so
    error cannot accumulate over a long contour. Genuine off-axis diagonals (angle
    error > ``snap_deg``) and true curves (never routed here) are left untouched.
    """
    v = end - start
    length = math.hypot(v[0], v[1])
    if length < 1e-6:
        return end, 0.0
    ang = math.degrees(math.atan2(v[1], v[0]))
    nearest = round(ang / 45.0) * 45.0  # closest of 0/45/90/135/180/-45/...
    err = ang - nearest
    if abs(err) > snap_deg:
        return end, 0.0  # genuine off-axis diagonal: leave it
    rad = math.radians(nearest)
    snapped = start + length * np.array([math.cos(rad), math.sin(rad)])
    drift = float(math.hypot(snapped[0] - end[0], snapped[1] - end[1]))
    if drift > max_drift:
        return end, 0.0
    return snapped, drift


def _fit_arc(arc: np.ndarray, tol: float) -> list[np.ndarray]:
    """Fit an open arc with Schneider cubics; tangents from the arc ends."""
    t_left = _unit(arc[1] - arc[0])
    t_right = _unit(arc[-2] - arc[-1])
    return _fit_cubic(arc, t_left, t_right, tol)


# --------------------------------------------------------------------------- #
# Path-level smoothing
# --------------------------------------------------------------------------- #


def _smooth_d(
    d: str,
    *,
    min_perim: float,
    tol: float,
    sigma: float,
    pts_per_100: float,
    corner_deg: float,
    straight_tol: float,
    snap_deg: float,
    max_drift: float,
    samples: int,
    precision: int,
) -> str:
    chunks: list[str] = []

    def fmt(p) -> str:
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
        pts = np.asarray(_subpath_points(sub, samples), dtype=float)[:-1]  # drop closing dup
        perim = _perim(pts)
        if perim < min_perim:
            emit_raw(sub)  # tiny/thin feature: keep crisp detail
            continue
        # Denser floor (24) so tight curvature reversals — e.g. a small mouth W —
        # land between resample points rather than being missed and smoothed over.
        n = max(24, int(perim / 100 * pts_per_100))
        # Detect split points on the RAW resample (before any smoothing) so sharp
        # corners (stripe vertices) AND curvature reversals (mouth W) are preserved.
        rs = _resample(pts, n)
        min_turn = (perim / n) ** 2 * 0.15  # scale to edge length; ignore tiny wiggle
        splits = set(_corner_indices(rs, corner_deg)) | set(_inflection_indices(rs, min_turn))
        corners = sorted(splits)
        if len(corners) < 2:
            corners = [0, len(rs) // 2]
        pen = rs[corners[0]].astype(float)  # snapped pen position (propagates exactly)
        sub_chunks = [f"M{fmt(pen)}"]
        contour = [pen.copy()]  # sampled smoothed outline, for the fidelity check
        for j in range(len(corners)):
            a, b = corners[j], corners[(j + 1) % len(corners)]
            if b > a:
                idx = list(range(a, b + 1))
            else:
                idx = list(range(a, len(rs))) + list(range(0, b + 1))
            arc = rs[idx]
            if len(arc) < 3 or _max_deviation(arc) <= straight_tol:
                # Already classified straight: snap its angle to the nearest exact
                # axis (H/V/diagonal) and carry the snapped endpoint forward as the
                # pen so the next segment stays connected (no seam).
                end = _axis_snap(pen, arc[-1].astype(float), snap_deg, max_drift)[0]
                sub_chunks.append(f"L{fmt(end)}")  # straight edge stays crisp
                contour.append(end)
                pen = end
            else:
                # Light per-arc denoise (endpoints/corners pinned) then fit.
                for ctrl in _fit_arc(_gaussian(arc, sigma), tol):
                    sub_chunks.append(f"C{fmt(ctrl[1])} {fmt(ctrl[2])} {fmt(ctrl[3])}")
                    contour.extend(_bezier_point(ctrl, t) for t in (0.25, 0.5, 0.75, 1.0))
                    pen = np.asarray(ctrl[3], dtype=float)
        # Per-feature fidelity guard: if smoothing changed the enclosed area too much
        # (it filled a concavity — e.g. averaged across a mouth 'W' into a blob), keep
        # the crisp raw geometry instead. Protects small concave features the global
        # SSIM guard is too coarse to catch.
        if abs(_area(np.array(contour)) - _area(rs)) <= _area(rs) * 0.14:
            chunks.extend(sub_chunks)
            chunks.append("Z")
        else:
            emit_raw(sub)
    return "".join(chunks)


def _diagonal(root: ET.Element) -> float:
    vb = root.get("viewBox")
    if vb:
        parts = [float(v) for v in vb.replace(",", " ").split()]
        if len(parts) == 4:
            return math.hypot(parts[2], parts[3])

    def _len(value: str | None) -> float:
        if not value:
            return 0.0
        m = __import__("re").match(r"[-+]?(?:\d*\.\d+|\d+\.?)", value.strip())
        return float(m.group()) if m else 0.0

    return math.hypot(_len(root.get("width")), _len(root.get("height"))) or 1000.0


def smooth_svg(
    svg: str,
    *,
    tol_ratio: float = 0.0016,
    min_perim_ratio: float = 0.085,
    sigma: float = 1.0,
    pts_per_100: float = 8.0,
    corner_deg: float = 42.0,
    straight_tol_ratio: float = 0.0016,
    snap_deg: float = 10.0,
    max_drift_ratio: float = 0.012,
    samples: int = 6,
    precision: int = 2,
) -> str:
    """Return ``svg`` with every path's geometry curve-refit for smooth, sparse Béziers.

    ``tol_ratio`` / ``min_perim_ratio`` / ``straight_tol_ratio`` are fractions of the
    viewBox diagonal, so behavior is independent of canvas size. ``tol_ratio`` is the
    Bézier fit error budget — larger means fewer, smoother curves.
    """
    root = ET.fromstring(svg)
    diag = _diagonal(root)
    tol = (tol_ratio * diag) ** 2  # _max_error works in squared distance
    min_perim = min_perim_ratio * diag
    straight_tol = straight_tol_ratio * diag
    max_drift = max_drift_ratio * diag
    for path in root.iter(f"{{{SVG_NS}}}path"):
        d = path.get("d")
        if d:
            path.set(
                "d",
                _smooth_d(
                    d,
                    min_perim=min_perim,
                    tol=tol,
                    sigma=sigma,
                    pts_per_100=pts_per_100,
                    corner_deg=corner_deg,
                    straight_tol=straight_tol,
                    snap_deg=snap_deg,
                    max_drift=max_drift,
                    samples=samples,
                    precision=precision,
                ),
            )
    body = ET.tostring(root, encoding="unicode")
    if not body.startswith("<?xml"):
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + body
    return body
