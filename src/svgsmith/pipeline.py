"""End-to-end conversion pipeline.

Orchestrates classify → preprocess → trace → postprocess → verify and assembles
the canonical :class:`~svgsmith.report.Report`. The CLI is a thin wrapper around
:func:`convert`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from svgsmith.classify import Classification, classify
from svgsmith.engines.base import ImageInput, load_image
from svgsmith.postprocess import drop_background_paths
from svgsmith.preprocess import PreprocessOptions, _edge_flood_fill_mask, preprocess
from svgsmith.report import Report, svg_stats
from svgsmith.smooth import smooth_svg
from svgsmith.verify import rasterize, run_loop, score

_DEFAULT_BG_TOLERANCE = PreprocessOptions().background_tolerance

# Mode → engine label and the preset used when --mode is given explicitly.
_ENGINE = {"binary": "potrace", "color": "vtracer", "pixel": "vtracer"}
# Detail levels for color mode — the dial between maximum fidelity and a flat
# poster look. Each maps to (edge-preserving flatten sigma, pre-trace palette size,
# perceptual LAB ΔE merge threshold). "normal" is the default and reproduces the
# prior behavior exactly. Higher levels flatten more and keep fewer colors.
DETAIL_LEVELS = {
    "high": (0.02, 64, 10.0),  # maximum detail: least flattening, richest palette
    "normal": (0.04, 48, 14.0),  # balanced (default)
    "clean": (0.10, 48, 18.0),  # tidied: edge-preserving cleanup, noise reduced
    "poster": (0.13, 28, 30.0),  # bold flat graphic: few colors, strong flattening
}
# Pre-trace supersampling target (#60) and per-detail fixed-K palette size (#41)
# for color mode. Upscaling smooths contours; the small k-means palette + black
# anchor snaps each region to one clean color and keeps the outline pure black.
# K scales with detail so a genuinely rich illustration is not crushed at "high".
_TRACE_RESOLUTION = 2048
# Only low-resolution color inputs get the supersample + k-means treatment;
# already-large clean art traces smoothly on the proven path and upscaling it
# just bloats node count (a 1024px PNG shiba regresses 95→219 paths if upscaled).
# Gate for the supersample + fixed-K palette path. It targets exactly one class:
# low-resolution, raster-degraded FLAT cartoon/illustration art (a 640px JPEG
# cartoon). Three signals keep it off everything else:
#   * resolution  — already-large art traces smoothly; upscaling only bloats it.
#   * distinct colors — synthetic/clean flats (a handful of exact colors) have
#     nothing to consolidate; only JPEG/anti-aliased art (thousands) does.
#   * edge density (native) — bounded BOTH ways: a smooth gradient/photo has
#     almost no hard edges (below the floor), a hatched/sketch drawing has edges
#     everywhere and explodes when upscaled (above the ceiling); a flat cartoon's
#     bold outlines sit in between.
_SUPERSAMPLE_BELOW = 1024
_MIN_DISTINCT_COLORS = 256
_MIN_EDGE_DENSITY = 0.02
_MAX_EDGE_DENSITY = 0.12
DETAIL_KMEANS_K = {"high": 14, "normal": 11, "clean": 9, "poster": 6}
# A genuinely flat cartoon supersamples to a low path count (the pigeon ~380). If
# the supersampled trace blows past this, the input was NOT flat (a textured/photo
# input that slipped through the cheap gate) — fall back to the baseline path so a
# misclassification costs one extra trace, never a bloated SVG.
_SUPERSAMPLE_MAX_PATHS = 1200


def _supersample_candidate(image: ImageInput) -> bool:
    """Whether ``image`` is the low-res, degraded, flat-cartoon class (see above).

    This is a cheap PRE-filter; it can still admit a textured/photo input whose
    edge density happens to land in the band. ``convert`` therefore caps the result:
    if the supersampled trace is not actually flat (too many paths) it falls back to
    the baseline, so a misclassification costs an extra trace, never a bloated SVG.
    """
    rgb = load_image(image, "RGB")
    if max(rgb.size) >= _SUPERSAMPLE_BELOW:
        return False
    # getcolors early-exits at the cap (None iff > _MIN_DISTINCT_COLORS colors), far
    # cheaper than np.unique over every pixel on the common color-convert path.
    if rgb.getcolors(maxcolors=_MIN_DISTINCT_COLORS) is not None:
        return False  # synthetic / already-clean flat: nothing to consolidate
    # Edge density is a fraction of all pixels, so it scales ~1/linear-dimension;
    # the band is calibrated for the ~640px low-res target this gate applies to.
    from scipy.ndimage import sobel

    gray = np.asarray(rgb.convert("L"), dtype=float)
    edge_density = float((np.hypot(sobel(gray, 0), sobel(gray, 1)) > 64).mean())
    return _MIN_EDGE_DENSITY < edge_density < _MAX_EDGE_DENSITY
# Max SSIM the curve-smoothing pass may cost before we fall back to un-smoothed
# output (smoothing reduces SSIM slightly by design; a big drop = lost feature).
_SMOOTH_SSIM_TOLERANCE = 0.06
_MODE_PRESET = {"binary": "logo", "color": "illustration", "pixel": "pixel"}
MODES = ("auto", *_MODE_PRESET)


@dataclass(frozen=True)
class ConvertOptions:
    """Options for :func:`convert` (mirrors the ``svgsmith convert`` flags)."""

    mode: str = "auto"
    quality: float = 0.9
    max_iters: int = 4
    editable: bool = True
    smooth: bool = True  # curve-refit color output (Schneider Bezier fit) for smooth contours
    uniform_outline: bool = False  # opt-in: force an even outline band (outlined art only)
    solid_background: bool = False  # opt-in: isolate subject, repaint background one solid color
    background: str | None = None  # exact bg color (#RRGGBB/named); "auto" == --solid-background
    transparent_background: bool = False  # opt-in: drop the background → transparent SVG
    flatten_shading: bool = False  # opt-in: collapse soft/glossy shading before tracing
    detail: str = "normal"  # color detail dial: high | normal | clean | poster
    out: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0, 1], got {self.quality}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if self.detail not in DETAIL_LEVELS:
            raise ValueError(
                f"detail must be one of {', '.join(DETAIL_LEVELS)}, got {self.detail!r}"
            )
        if self.background not in (None, "auto"):
            # Fail fast on a bad color rather than surfacing it mid-pipeline.
            from svgsmith.preprocess import _parse_color

            _parse_color(self.background)


def _resolve_classification(image, mode: str) -> Classification:
    """Classify the image; honor an explicit --mode while keeping any warnings."""
    auto = classify(image)
    if mode == "auto":
        return auto
    if mode not in _MODE_PRESET:
        raise ValueError(f"unknown mode {mode!r}; choose one of: {', '.join(MODES)}")
    # Keep the classifier's warnings (e.g. the photo signal) but use the
    # caller-forced mode and its default preset.
    return Classification(mode=mode, preset=_MODE_PRESET[mode], warnings=auto.warnings)


def _output_path(input_path: str, out: str | None) -> str:
    if out:
        return out
    return str(Path(input_path).with_suffix(".svg"))


def _preprocess_opts(mode: str) -> PreprocessOptions:
    """Mode-aware preprocessing.

    Color illustrations must not be pre-quantized, over-denoised, or have their
    solid background flood-filled away — those steps crush color and delete small
    features (faces). VTracer owns color reduction downstream. Line art and pixel
    art keep light, mode-appropriate cleanup. Backgrounds are never removed unless
    a caller asks (a solid background is content, not noise).
    """
    if mode == "color":
        # Quantize to a generous palette (clean flat regions for VTracer, keeps
        # dark fills like outlines/hoodies) but never strip the background, and
        # skip denoise so small features (eyes, faces) survive.
        return PreprocessOptions(
            denoise=False,
            flatten=True,
            quantize=True,
            palette_size=48,
            remove_background=False,
        )
    if mode == "pixel":
        # Pixel art keeps the original cleanup (upscale + quantize); it relies on
        # quantization for crisp flat cells.
        return PreprocessOptions()
    # binary / line art: keep the background (a solid bg is content, not noise).
    return PreprocessOptions(remove_background=False)


def convert(input_path: str, opts: ConvertOptions | None = None) -> tuple[str, Report]:
    """Convert a raster image to SVG and return ``(svg, Report)``.

    Pure with respect to the filesystem: it does not write the SVG — the caller
    (the CLI) is responsible for writing ``svg`` to ``report.output``.
    """
    opts = opts or ConvertOptions()
    image: ImageInput = load_image(input_path, "RGBA")

    classification = _resolve_classification(image, opts.mode)
    flatten_sigma, palette_size, palette_threshold = DETAIL_LEVELS[opts.detail]
    is_color = classification.mode == "color"

    def resolve_pre_opts(use_supersample: bool) -> PreprocessOptions:
        """Preprocess options for this convert, with or without the supersample path.

        Low-resolution, FLAT color inputs (a 640px JPEG cartoon) trace staircased,
        near-duplicate-colored contours; supersampling + a small fixed-K k-means
        palette with a black/outline anchor fixes that. Everything else uses the
        proven median-cut + perceptual-LAB-merge path. The opt-in flags
        (--flatten-shading, --solid-background, --uniform-outline) apply to both.
        """
        pre = _preprocess_opts(classification.mode)
        if is_color and use_supersample:
            pre = replace(
                pre,
                flatten=False,
                trace_resolution=_TRACE_RESOLUTION,
                kmeans_palette=True,
                palette_k=DETAIL_KMEANS_K[opts.detail],
            )
        elif is_color:
            pre = replace(pre, flatten_sigma=flatten_sigma, palette_size=palette_size)
        if opts.uniform_outline and is_color:
            pre = replace(pre, uniform_outline=True)
        if opts.flatten_shading and is_color:
            pre = replace(pre, flatten=True, flatten_sigma=0.18, flatten_spatial=8)
        if opts.solid_background or opts.background is not None:
            target = None if opts.background in (None, "auto") else opts.background
            pre = replace(pre, solid_background=True, background_color=target)
        return pre

    def render(pre: PreprocessOptions) -> tuple[str, float, int]:
        """Trace + verify-loop + smooth for one preprocess config -> (svg, score, iters)."""
        svg, result = run_loop(
            preprocess(image, pre),
            classification,
            quality=opts.quality,
            max_iters=opts.max_iters,
            editable=opts.editable,
            reference=image,  # score against the true original, not the preprocessed image
            palette_threshold=palette_threshold if is_color else None,
        )
        # Curve-refit color output so contours are smooth (the verify loop traces
        # quantized pixel edges, which wobble). Keep the smoothed result only if it
        # does not materially hurt fidelity — smoothing lowers SSIM a little by
        # design, but a large drop means it destroyed a feature.
        similarity = result.best_score
        if opts.smooth and opts.editable and is_color:
            reference = load_image(image, "RGB")
            smoothed = smooth_svg(svg)
            smoothed_score = score(reference, rasterize(smoothed, reference.size))
            if smoothed_score >= result.best_score - _SMOOTH_SSIM_TOLERANCE:
                svg, similarity = smoothed, smoothed_score
        return svg, similarity, result.iterations

    supersampled = is_color and _supersample_candidate(image)
    svg, similarity, iterations = render(resolve_pre_opts(supersampled))
    # Output-complexity guard: a real flat cartoon supersamples to few paths. If the
    # trace exploded, the gate admitted a non-flat input — fall back to the baseline.
    if supersampled and svg.count("<path") > _SUPERSAMPLE_MAX_PATHS:
        svg, similarity, iterations = render(resolve_pre_opts(False))

    # Transparent background: trace/verify ran normally (with the background), so
    # the loop is unaffected; here we cut the edge-connected background paths from
    # the final SVG using a region mask (subjects sharing the bg colour are kept).
    # Color mode only — its paths live in viewBox space; binary/pixel line-art is
    # already foreground-only, so there is no background layer to remove.
    if opts.transparent_background and classification.mode == "color":
        bg_mask = _edge_flood_fill_mask(
            np.array(load_image(input_path, "RGBA")), _DEFAULT_BG_TOLERANCE
        )
        svg = drop_background_paths(svg, bg_mask)

    output = _output_path(input_path, opts.out)
    report = Report(
        output=output,
        mode_used=classification.mode,
        engine=_ENGINE[classification.mode],
        preset=classification.preset,
        iterations=iterations,
        similarity=similarity,
        passed_threshold=similarity >= opts.quality,
        svg=svg_stats(svg),
        warnings=list(classification.warnings),
    )
    return svg, report
