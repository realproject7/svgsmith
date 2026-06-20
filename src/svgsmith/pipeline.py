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
    pre_opts = _preprocess_opts(classification.mode)
    if classification.mode == "color":
        pre_opts = replace(pre_opts, flatten_sigma=flatten_sigma, palette_size=palette_size)
    if opts.uniform_outline and classification.mode == "color":
        pre_opts = replace(pre_opts, uniform_outline=True)
    # --solid-background is the auto case of --background; an explicit color (other
    # than "auto") repaints the detected background to that exact color.
    if opts.solid_background or opts.background is not None:
        target = None if opts.background in (None, "auto") else opts.background
        pre_opts = replace(pre_opts, solid_background=True, background_color=target)
    prepared = preprocess(image, pre_opts)

    svg, result = run_loop(
        prepared,
        classification,
        quality=opts.quality,
        max_iters=opts.max_iters,
        editable=opts.editable,
        reference=image,  # score against the true original, not the preprocessed image
        palette_threshold=palette_threshold if classification.mode == "color" else None,
    )

    # Curve-refit color output so contours are smooth (the verify loop traces
    # quantized pixel edges, which wobble). Re-score the smoothed result and keep
    # it only if it does not materially hurt fidelity — smoothing legitimately
    # lowers SSIM a little (SSIM penalizes the small shape change), but a large
    # drop means it destroyed a feature, so we fall back to the un-smoothed SVG.
    similarity = result.best_score
    if opts.smooth and opts.editable and classification.mode == "color":
        reference = load_image(image, "RGB")
        smoothed = smooth_svg(svg)
        smoothed_score = score(reference, rasterize(smoothed, reference.size))
        if smoothed_score >= result.best_score - _SMOOTH_SSIM_TOLERANCE:
            svg, similarity = smoothed, smoothed_score

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
        iterations=result.iterations,
        similarity=similarity,
        passed_threshold=similarity >= opts.quality,
        svg=svg_stats(svg),
        warnings=list(classification.warnings),
    )
    return svg, report
