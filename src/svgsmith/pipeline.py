"""End-to-end conversion pipeline.

Orchestrates classify → preprocess → trace → postprocess → verify and assembles
the canonical :class:`~svgsmith.report.Report`. The CLI is a thin wrapper around
:func:`convert`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from svgsmith.classify import Classification, classify
from svgsmith.engines.base import ImageInput, load_image
from svgsmith.postprocess import drop_background_paths, snap_background_layer
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
# Outline isolation (black-hat) pulls the dark linework out of the fills, so fewer
# clusters are needed for the actual colors — a tighter palette closer to the
# artist's true few flats (reference vectorizers land ~8 on a flat cartoon).
DETAIL_KMEANS_K = {"high": 10, "normal": 8, "clean": 7, "poster": 5}
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


# Coverage-quant gate (#65, Tier 1). Smooth continuous-tone art (gradients) loses its
# colour to the fixed-K / median-cut + LAB-merge path (a 200-band gradient collapses to
# ~15 visible steps). The perceptual-coverage path keeps the bands. The gate is
# deliberately narrow: a near-zero hard-edge density isolates the smooth-gradient class
# with a wide margin below every other category (measured on the study corpus: gradients
# <= 0.0045, the next-lowest non-gradient >= 0.014, flat art >= 0.035), so flat/outlined
# illustration — the must-stay-clean class — never takes this path. A complexity fallback
# in ``convert`` backs out to the baseline if a trace still explodes.
_COVERAGE_MAX_EDGE_DENSITY = 0.008
_COVERAGE_MIN_DISTINCT = 256  # must be genuinely continuous-tone, not a low-edge flat
_COVERAGE_MAX_PATHS = 4000  # safety cap; over this, fall back to the baseline path

# Tier 2 (#67): perceptual-coverage + connected-component min-AREA cleanup as the default
# colour engine. The area filter makes coverage economical and clean on *any* rich-colour
# input — gradients inside hard-edged shapes (#69), ragged dark outlines (#66), and flat
# art alike — so the narrow edge-density gate is no longer needed. SAFE-REVERT: set this
# False to fall back to pure Tier 1 (narrow full-frame-gradient gate, no region cleanup).
_TIER2_REGION_COVERAGE = True


def _coverage_candidate(image: ImageInput) -> bool:
    """Whether ``image`` should take the perceptual-coverage colour path.

    Requires genuinely rich colour (>256 distinct) so a truly flat low-colour image keeps
    the proven baseline. Tier 2 then admits any such input (the region cleanup keeps it
    economical); Tier 1 additionally requires near-zero hard edges (full-frame gradients).
    """
    rgb = load_image(image, "RGB")
    if rgb.getcolors(maxcolors=_COVERAGE_MIN_DISTINCT) is not None:
        return False
    if _TIER2_REGION_COVERAGE:
        return True
    from scipy.ndimage import sobel

    gray = np.asarray(rgb.convert("L"), dtype=float)
    edge_density = float((np.hypot(sobel(gray, 0), sobel(gray, 1)) > 64).mean())
    return edge_density < _COVERAGE_MAX_EDGE_DENSITY


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
    # load_image's .convert() drops the source format; re-attach it so the lossy-input
    # denoise gate (#71) can tell a JPEG from a clean PNG downstream.
    try:
        with Image.open(input_path) as _src:
            image.format = _src.format
    except Exception:
        pass

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
                detect_linework=True,
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

    def render(
        pre: PreprocessOptions,
        classification: Classification = classification,
        palette_threshold: float | None = palette_threshold if is_color else None,
        max_iters: int = opts.max_iters,
    ) -> tuple[str, float, int]:
        """Trace + verify-loop + smooth for one preprocess config -> (svg, score, iters).

        ``classification`` / ``palette_threshold`` / ``max_iters`` default to the resolved
        values; the coverage-quant path overrides them (the ``continuous`` preset + a
        near-zero merge threshold so the rich emergent palette survives, and a single pass
        since the palette is already fixed in preprocessing — the colour-ramp loop only
        re-traces at the same palette and would multiply trace cost for no gain).
        """
        svg, result = run_loop(
            preprocess(image, pre),
            classification,
            quality=opts.quality,
            max_iters=max_iters,
            editable=opts.editable,
            reference=image,  # score against the true original, not the preprocessed image
            palette_threshold=palette_threshold,
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

    # Smooth continuous-tone art (gradients): perceptual-coverage quantization keeps the
    # many low-step bands a fixed-K path crushes into visible banding. Traced with the
    # ``continuous`` preset (preserve the palette) and no postprocess LAB merge. Flat /
    # outlined art never reaches this branch (see _coverage_candidate). A complexity
    # fallback backs out to the baseline if a misgated input still explodes.
    # --flatten-shading is the explicit "collapse shading to a clean flat look" opt-out,
    # the opposite of coverage's faithful many-band preservation — so it bypasses coverage.
    coverage = is_color and not opts.flatten_shading and _coverage_candidate(image)
    if coverage:
        cov_pre = replace(
            resolve_pre_opts(False),
            coverage_palette=True,
            coverage_region_cleanup=_TIER2_REGION_COVERAGE,
        )
        cov_class = classification._replace(preset="continuous")
        svg, similarity, iterations = render(
            cov_pre, cov_class, palette_threshold=0.0, max_iters=1
        )
        if svg.count("<path") > _COVERAGE_MAX_PATHS:
            coverage = False  # not actually a clean gradient — use the proven path

    if not coverage:
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
    if opts.transparent_background and is_color:
        bg_mask = _edge_flood_fill_mask(
            np.array(load_image(input_path, "RGBA")), _DEFAULT_BG_TOLERANCE
        )
        svg = drop_background_paths(svg, bg_mask)
    elif is_color and opts.editable:
        # Snap a wobbly full-canvas background to a crisp rectangle (no-op when the
        # image has no dominant solid background). Skipped when the background was
        # just removed above, and for --no-editable (which returns the raw trace).
        # Color mode only — line art has no background layer.
        svg = snap_background_layer(svg)

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
