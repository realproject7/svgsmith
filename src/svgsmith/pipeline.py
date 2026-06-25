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
from svgsmith.postprocess import (
    drop_background_paths,
    global_same_fill_merge,
    snap_background_layer,
    snap_dark_fills,
)
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
# Detail-aware region-merge ΔE76 threshold per detail level for the coverage path: a small
# region merges into its neighbour only when the neighbour is closer than this (AA/noise);
# distinct marks farther than it are KEPT. LOWER = keep more subtle texture (painterly grain,
# fine stipple) = faithful but heavier; HIGHER = collapse more = economical/flat. "normal" is
# the loop-validated economical-fidelity default; "high" preserves painterly brush-grain
# (reference-grade on textured art, at more paths/bytes — fidelity over economy, on demand).
COVERAGE_NOISE_DE = {"high": 5.0, "normal": 13.0, "clean": 18.0, "poster": 26.0}
# Pre-trace supersampling target (#60) and per-detail fixed-K palette size (#41)
# for color mode. Upscaling smooths contours; the small k-means palette + black
# anchor snaps each region to one clean color and keeps the outline pure black.
# K scales with detail so a genuinely rich illustration is not crushed at "high".
_TRACE_RESOLUTION = 2048
# Auto/--hires supersample target long-edge (line-quality Phase 3): the loop-validated minimum
# that lifts a sub-768px input off the native pixel staircase (subpixel grid + SSIM gain) without
# the heavier cost of the full reference-grid factor. Paired with an uncapped region merge so the
# flat-economy levers can claw the high-res path blow-up back toward reference counts.
_SUPERSAMPLE_AUTO_RES = 1536
_SUPERSAMPLE_REGION_MAX_PX = 20000  # uncap the region merge at high res (native cap is 2000)
_SAME_FILL_MERGE_SSIM_DROP = 0.01  # gate: keep the same-fill merge only within this SSIM cost
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
# At --detail high (max fidelity) legitimately grainy art needs far more tiles; raise the
# cap so grain survives, while still catching a misgated photo (which explodes far past it).
_COVERAGE_MAX_PATHS_HIGH = 14000

# Tier 2 (#67): perceptual-coverage + connected-component min-AREA cleanup as the default
# colour engine. The area filter makes coverage economical and clean on *any* rich-colour
# input — gradients inside hard-edged shapes (#69), ragged dark outlines (#66), and flat
# art alike — so the narrow edge-density gate is no longer needed. SAFE-REVERT: set this
# False to fall back to pure Tier 1 (narrow full-frame-gradient gate, no region cleanup).
_TIER2_REGION_COVERAGE = True

# Dark-outline black snap (#lever-C). On the coverage path the dark linework of an outlined
# cartoon scatters across many near-black tints (the pigeon JPEG: 186 dark colours), bloating
# paths and giving a ragged variable-width outline. We pull every pixel within this ΔE76 ball of
# pure black out of the palette build and paint it ONE clean #000000 region (NO morphology — that
# was verified to regress outline IoU). ``_COVERAGE_BLACK_SNAP_DE`` is the ball radius;
# ``_COVERAGE_BLACK_SNAP_MIN_AREA`` is the near-black share that must be present before it fires,
# so line-free / no-black art (e.g. c_07) is left byte-identical. SSIM-guarded in ``convert``: the
# snapped trace is kept only when it does not cost more than ``_COVERAGE_BLACK_SNAP_SSIM_DROP``
# SSIM against the no-snap trace. SAFE-REVERT: set the ΔE to 0.0 to disable.
_COVERAGE_BLACK_SNAP_DE = 12.0
_COVERAGE_BLACK_SNAP_MIN_AREA = 0.004
_COVERAGE_BLACK_SNAP_SSIM_DROP = 0.01


def _distinct_dark_fills(svg: str) -> int:
    """Count distinct dark (luma < 60) fill colours in an SVG — the dark-layer cleanliness
    proxy. Used to guard the black snap: a sharper black core can make the colour tracer
    carve MORE near-black edge tints on some inputs, so we keep the snap only when it does
    not inflate this count.
    """
    import re

    dark = set()
    for hexc in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg):
        r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
        if 0.299 * r + 0.587 * g + 0.114 * b < 60:
            dark.add(hexc.lower())
    return len(dark)


def _has_black_outline(image: ImageInput) -> bool:
    """Cheap pre-check: does ``image`` carry enough near-pure-black mass to snap an outline?

    Mirrors the per-pixel test inside ``quantize_coverage`` (ΔE76 to the LAB origin) so the
    pipeline can decide up front whether the black-snap will fire — and therefore whether the
    SSIM-guard's extra no-snap render is even needed. No-black art short-circuits to a single
    render with byte-identical output.
    """
    from skimage.color import rgb2lab

    rgb = np.asarray(load_image(image, "RGB"), dtype=np.float64)
    lab = rgb2lab(rgb / 255.0)
    near_black = (np.linalg.norm(lab, axis=2) <= _COVERAGE_BLACK_SNAP_DE).mean()
    return float(near_black) >= _COVERAGE_BLACK_SNAP_MIN_AREA


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
# Wobble-relief escape clause for the smooth gate. Raw-SSIM rewards the antialiased
# pixel staircase, so on the wobbliest inputs the visibly-SHARPER smoothed output can
# score LOWER and get vetoed by the plain SSIM gate above — exactly where smoothing is
# most needed (e.g. a jittery vtracer polyline that smoothing straightens). To recover
# those cases we measure path wobble directly off the SVG geometry (angle sign-flips per
# unit length over the polyline approximation — the same structure SSIM is blind to). If
# smoothing cuts wobble by at least _SMOOTH_WOBBLE_RELIEF (relative) AND SSIM stays within
# the wider _SMOOTH_SSIM_HARD_FLOOR, we keep the smoothed output even when the plain SSIM
# tolerance would veto it. The hard floor still guards against a genuine feature loss
# (which would crater SSIM far past the staircase penalty). Inputs that already clear the
# plain tolerance are unaffected, so default behaviour is unchanged for them.
_SMOOTH_WOBBLE_RELIEF = 0.25
_SMOOTH_SSIM_HARD_FLOOR = 0.10
_MODE_PRESET = {"binary": "logo", "color": "illustration", "pixel": "pixel"}
MODES = ("auto", *_MODE_PRESET)


def _path_wobble(svg: str) -> float:
    """Wobble of an SVG's contours: angle sign-flips per unit length over its polylines.

    Rasterization-free structural measure of polyline jitter — high when a contour
    zig-zags (raw quantized-pixel edges), low when it is a clean curve (after smoothing).
    Mirrors the line-quality jitter metric so the smooth gate can reward genuine wobble
    removal that raw SSIM (which rewards the antialiased staircase) penalises.
    """
    import math
    import re
    import xml.etree.ElementTree as ET

    num = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return 0.0
    flips = 0.0
    length = 0.0
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] != "path":
            continue
        toks = re.findall(r"[MmLlCcQqZzHhVv]|" + num, el.attrib.get("d", ""))
        pts: list[list[float]] = []
        subs: list[np.ndarray] = []
        cur = [0.0, 0.0]
        i = 0
        while i < len(toks):
            t = toks[i]
            if t in "Mm":
                if pts:
                    subs.append(np.array(pts))
                    pts = []
                x, y = float(toks[i + 1]), float(toks[i + 2])
                cur = [x, y] if t == "M" else [cur[0] + x, cur[1] + y]
                pts = [cur[:]]
                i += 3
            elif t in "Ll":
                x, y = float(toks[i + 1]), float(toks[i + 2])
                cur = [x, y] if t == "L" else [cur[0] + x, cur[1] + y]
                pts.append(cur[:])
                i += 3
            elif t in "Cc":
                c = [float(v) for v in toks[i + 1 : i + 7]]
                cur = [c[4], c[5]] if t == "C" else [cur[0] + c[4], cur[1] + c[5]]
                pts.append(cur[:])
                i += 7
            elif t in "Qq":
                c = [float(v) for v in toks[i + 1 : i + 5]]
                cur = [c[2], c[3]] if t == "Q" else [cur[0] + c[2], cur[1] + c[3]]
                pts.append(cur[:])
                i += 5
            else:
                i += 1
        if pts:
            subs.append(np.array(pts))
        for poly in subs:
            if len(poly) < 4:
                continue
            seg = np.diff(poly, axis=0)
            seg_len = np.hypot(seg[:, 0], seg[:, 1])
            ang = np.arctan2(seg[:, 1], seg[:, 0])
            if len(ang) <= 2:
                continue
            da = np.diff(ang)
            da = (da + np.pi) % (2 * np.pi) - np.pi
            small = np.abs(da) < math.radians(40)
            flip = np.sign(da[:-1]) != np.sign(da[1:])
            flips += float(np.sum(small[:-1] & flip))
            length += float(seg_len.sum())
    return 100.0 * flips / max(length, 1.0)


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
    # Illustration-geometry knobs (experimental; tuned by the daily quality loop). Both default
    # OFF, so behaviour is unchanged unless set. They apply ONLY to the outlined low-res
    # illustration class on the coverage path (gated by ``_supersample_candidate``); gradients,
    # photos, and high-res inputs ignore them. ``illustration_supersample`` traces the flat-colour
    # mask at this resolution (e.g. 2048) so feather/scallop boundaries come out round and uniform;
    # ``illustration_dark_thin`` erodes the dark linework by this many steps (at the trace
    # resolution) so thick outline blobs become thin crescents (small dark detail is protected).
    illustration_supersample: int = 0
    illustration_dark_thin: int = 0
    # Resolution lever (line-quality Phase 3): trace a supersampled mask so curves are fit on
    # a fine grid (crisp, high-resolution lines) instead of the native pixel staircase. AUTO for
    # flat low-resolution illustrations (the ``_supersample_candidate`` class — e.g. a 640px JPEG
    # cartoon that traces staircased); ``hires`` FORCES it on any color input (textured/high-res
    # too, where the same-fill economy may not apply so it just costs more paths). Gradients,
    # photos and already-high-res inputs are untouched.
    hires: bool = False
    out: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0, 1], got {self.quality}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if self.illustration_supersample < 0:
            raise ValueError(
                f"illustration_supersample must be >= 0, got {self.illustration_supersample}"
            )
        if self.illustration_dark_thin < 0:
            raise ValueError(
                f"illustration_dark_thin must be >= 0, got {self.illustration_dark_thin}"
            )
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
            # Lossless supersample byte lever: when the mask was traced above native
            # resolution the viewBox is N× the native grid, so default .2f coordinates
            # resolve far finer than the original pixels — pure byte bloat. smooth_svg
            # auto-drops decimals by log10(factor) so granularity tracks the native grid.
            smoothed = smooth_svg(svg, native_long_edge=max(reference.size))
            smoothed_score = score(reference, rasterize(smoothed, reference.size))
            keep = smoothed_score >= result.best_score - _SMOOTH_SSIM_TOLERANCE
            # Wobble-relief escape clause: raw SSIM rewards the antialiased staircase, so a
            # visibly-sharper smooth can dip past the plain tolerance on the wobbliest inputs.
            # Recover those by measuring path wobble directly (SSIM is blind to it): if
            # smoothing genuinely straightens the contours and SSIM stays within the wider
            # hard floor, keep it. The floor still vetoes a true feature loss (huge SSIM drop).
            if not keep and smoothed_score >= result.best_score - _SMOOTH_SSIM_HARD_FLOOR:
                raw_wobble = _path_wobble(svg)
                if raw_wobble > 0.0:
                    relief = (raw_wobble - _path_wobble(smoothed)) / raw_wobble
                    keep = relief >= _SMOOTH_WOBBLE_RELIEF
            if keep:
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
            # Economical-fidelity operating point (loop-validated): a fine perceptual step with
            # almost no global speckle drop preserves subtle colour detail, while the detail-aware
            # region merge (region_noise_de) collapses low-ΔE anti-alias fringes for economy/speed
            # yet KEEPS high-ΔE deliberate marks (dots/stipple/texture). Reaches reference-grade
            # fidelity at ~1.4-3.6x reference bytes and 1-8 s (was 8-13x / 40-218 s without it).
            coverage_step=5.0,
            coverage_min_area=0.00003,
            coverage_region_min_area=0.0006,
            coverage_region_noise_de=COVERAGE_NOISE_DE[opts.detail],
        )
        # --detail high = max fidelity: skip the pre-trace flatten so painterly brush-grain
        # / fine texture survives to the tracer (flatten, even light, smooths grain away).
        # Other levels keep the flatten (the loop-validated economical default).
        if opts.detail == "high":
            cov_pre = replace(cov_pre, flatten=False)
        # Resolution lever (line-quality Phase 3): supersample the flat-colour mask so curves are
        # fit on a fine grid (crisp, high-resolution lines) instead of the native pixel staircase.
        # AUTO for the flat low-res illustration class (``_supersample_candidate`` — e.g. a 640px
        # JPEG cartoon that traces staircased); FORCED by ``--hires``; or set explicitly by the
        # experimental illustration knob. When supersampling, uncap the region merge (the 2000px
        # cap throttles consolidation at high res) so the flat-economy levers can claw the path
        # blow-up back down. Gradients, photos and already-high-res inputs never enter here.
        ss_res = opts.illustration_supersample or (
            _SUPERSAMPLE_AUTO_RES if (_supersample_candidate(image) or opts.hires) else 0
        )
        did_supersample = bool(ss_res)
        if did_supersample or opts.illustration_dark_thin:
            cov_pre = replace(
                cov_pre,
                trace_resolution=ss_res or cov_pre.trace_resolution,
                coverage_dark_thin=opts.illustration_dark_thin,
                coverage_region_max_px=(
                    _SUPERSAMPLE_REGION_MAX_PX
                    if did_supersample
                    else cov_pre.coverage_region_max_px
                ),
            )
        cov_class = classification._replace(preset="continuous")
        # Dark-outline black snap (#lever-C): collapse scattered near-black tints into one clean
        # #000000 outline layer. Only engage when there is genuine near-black mass (so no-black art
        # is untouched and pays no extra render); then SSIM-guard against the no-snap trace so a
        # dark-but-not-black-detail image can never regress.
        snap_black = _COVERAGE_BLACK_SNAP_DE > 0.0 and _has_black_outline(image)
        base_svg, base_sim, base_iters = render(
            cov_pre, cov_class, palette_threshold=0.0, max_iters=1
        )
        if snap_black:
            # Palette-build snap gives the tracer a clean pure-black core; the output-side
            # fill snap then collapses the tints the colour tracer re-derives along the
            # anti-aliased outline edges into one #000000 layer. SSIM-guarded: keep the
            # snapped result only when it does not cost more than the tolerance vs the
            # un-snapped trace, so a dark-but-not-black-detail image can never regress.
            snap_pre = replace(
                cov_pre,
                coverage_black_snap=_COVERAGE_BLACK_SNAP_DE,
                coverage_black_snap_min_area=_COVERAGE_BLACK_SNAP_MIN_AREA,
            )
            snap_svg, snap_sim, snap_iters = render(
                snap_pre, cov_class, palette_threshold=0.0, max_iters=1
            )
            snap_svg = snap_dark_fills(snap_svg, _COVERAGE_BLACK_SNAP_DE)
            snap_sim = score(
                load_image(image, "RGB"), rasterize(snap_svg, load_image(image, "RGB").size)
            )
            # Keep the snap only when it both holds SSIM AND actually cleans the dark layer
            # (never inflates the distinct-dark-fill count) — a sharper black core can make the
            # colour tracer carve MORE edge tints on some inputs, the opposite of the goal.
            cleaner = _distinct_dark_fills(snap_svg) <= _distinct_dark_fills(base_svg)
            if base_sim - snap_sim <= _COVERAGE_BLACK_SNAP_SSIM_DROP and cleaner:
                svg, similarity, iterations = snap_svg, snap_sim, snap_iters
            else:
                svg, similarity, iterations = base_svg, base_sim, base_iters
        else:
            svg, similarity, iterations = base_svg, base_sim, base_iters
        # Flat-region economy (Phase 3): supersampling fragments a clean solid region into many
        # same-fill paths; collapse each colour's fragments into one compound path. SSIM-gated —
        # the hoist reorders paint and is lossless only on flat tiling art, so textured/overlapping
        # content fails the gate and keeps the un-merged trace (where the supersample cost stays).
        if did_supersample:
            merged_svg = global_same_fill_merge(svg)
            if merged_svg != svg:
                ref_img = load_image(image, "RGB")
                merged_sim = score(ref_img, rasterize(merged_svg, ref_img.size))
                if merged_sim >= similarity - _SAME_FILL_MERGE_SSIM_DROP:
                    svg, similarity = merged_svg, merged_sim
        # Path cap = misgated-photo blowup guard. At --detail high the user opted into
        # max fidelity, so a high count is INTENDED for legitimately grainy/painterly art
        # (the reference traces such inputs into thousands of micro-tiles) — raise the cap
        # so the grain-rich coverage output is kept instead of being thrown away and
        # re-traced on the baseline path (which both loses the grain AND doubles the time).
        # A truly misgated photo still explodes far past the high cap and falls back.
        cov_cap = _COVERAGE_MAX_PATHS_HIGH if opts.detail == "high" else _COVERAGE_MAX_PATHS
        if svg.count("<path") > cov_cap:
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
