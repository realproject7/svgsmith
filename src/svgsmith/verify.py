"""Self-verify loop: rasterize → SSIM → re-tune.

The headline feature. Renders produced SVG back to raster, scores it against the
original with SSIM, and re-tunes trace/postprocess parameters until a quality
target is met or the iteration budget runs out — returning the best-scoring
result.

This module returns a **lightweight internal result** (:class:`VerifyResult`):
per-iteration scores, the chosen params, and the iteration count. It does NOT
define the public ``Report`` — that is owned by the CLI ticket (T7).
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from svgsmith.engines import BinaryTracer, ColorTracer, Preset, get_preset
from svgsmith.engines.base import ImageInput, load_image
from svgsmith.postprocess import PostprocessOptions, postprocess

MAX_COLOR_PRECISION = 8


@dataclass(frozen=True)
class VerifyResult:
    """Lightweight result of :func:`run_loop` (not the public Report)."""

    scores: tuple[float, ...]  # per-iteration SSIM
    params: dict  # chosen (best) parameters
    iterations: int
    best_score: float


def rasterize(svg: str, size: tuple[int, int], renderer: str | None = None) -> Image.Image:
    """Render an SVG string to an RGB raster at ``size`` (width, height).

    Uses ``cairosvg`` by default (pip-installable, self-contained in CI). If the
    ``resvg`` binary is present it is used instead, unless ``renderer`` forces a
    choice (``"cairosvg"`` or ``"resvg"``).
    """
    width, height = size
    chosen = renderer or ("resvg" if shutil.which("resvg") else "cairosvg")
    if chosen == "resvg":
        return _rasterize_resvg(svg, width, height)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, output_height=height)
    return Image.open(io.BytesIO(png)).convert("RGB")


def _rasterize_resvg(svg: str, width: int, height: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.svg"
        dst = Path(tmp) / "out.png"
        src.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["resvg", "--width", str(width), "--height", str(height), str(src), str(dst)],
            check=True,
            capture_output=True,
        )
        return Image.open(dst).convert("RGB")


def score(original: Image.Image, rendered: Image.Image) -> float:
    """Structural similarity (SSIM) in ``[0, 1]`` between two images."""
    a = np.asarray(original.convert("RGB"))
    if rendered.size != original.size:
        rendered = rendered.resize(original.size)
    b = np.asarray(rendered.convert("RGB"))
    return float(structural_similarity(a, b, channel_axis=2))


def _tune_preset(base: Preset, color_level: int) -> Preset:
    """Ramp color fidelity: more colors, fewer speckles, at higher levels."""
    color_precision = min(MAX_COLOR_PRECISION, 1 + 3 * color_level)
    filter_speckle = max(0, base.filter_speckle - color_level)
    return replace(base, color_precision=color_precision, filter_speckle=filter_speckle)


def _trace_and_post(image: Image.Image, mode: str, preset: Preset, simplify_level: float) -> str:
    engine = BinaryTracer() if mode == "binary" else ColorTracer()
    raw = engine.trace(image, preset)
    return postprocess(raw, PostprocessOptions(simplify_level=simplify_level))


def run_loop(
    image: ImageInput,
    classification,
    quality: float = 0.9,
    max_iters: int = 4,
    renderer: str | None = None,
) -> tuple[str, VerifyResult]:
    """Trace+postprocess, score, and re-tune up to ``max_iters``; return the best.

    ``classification`` is anything exposing ``.mode`` and ``.preset`` (e.g. the
    result of :func:`svgsmith.classify.classify`). The loop ramps color fidelity
    while the score is below ``quality``; once the target is reached it spends any
    remaining budget raising ``simplify_level`` (fewer points) — but only while
    the score stays at or above the target. If the first pass already meets the
    target, it returns immediately (cost discipline).
    """
    original = load_image(image, "RGB")
    base = get_preset(classification.preset)
    mode = classification.mode

    scores: list[float] = []
    best: dict | None = None
    reached = False
    color_level = 0
    simplify_level = 1.0

    for iteration in range(max_iters):
        if not reached:
            color_level = iteration
            simplify_level = 1.0
        else:
            # Headroom: cut points by simplifying more, gated on staying on target.
            simplify_level += 1.0

        preset = _tune_preset(base, color_level)
        svg = _trace_and_post(original, mode, preset, simplify_level)
        current = score(original, rasterize(svg, original.size, renderer))
        scores.append(current)
        params = {
            "mode": mode,
            "preset": preset.name,
            "color_precision": preset.color_precision,
            "filter_speckle": preset.filter_speckle,
            "simplify_level": simplify_level,
        }

        if not reached:
            if best is None or current > best["score"]:
                best = {"score": current, "svg": svg, "params": params}
            if current >= quality:
                reached = True
                if iteration == 0:
                    break  # first pass already good enough — don't keep iterating
        else:
            if current >= quality:
                best = {"score": current, "svg": svg, "params": params}
            else:
                break  # simplifying dropped us below target; keep the prior best

    assert best is not None  # max_iters >= 1, so the loop always records one result
    return best["svg"], VerifyResult(
        scores=tuple(scores),
        params=best["params"],
        iterations=len(scores),
        best_score=best["score"],
    )
