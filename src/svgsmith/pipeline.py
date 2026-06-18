"""End-to-end conversion pipeline.

Orchestrates classify → preprocess → trace → postprocess → verify and assembles
the canonical :class:`~svgsmith.report.Report`. The CLI is a thin wrapper around
:func:`convert`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from svgsmith.classify import Classification, classify
from svgsmith.engines.base import ImageInput, load_image
from svgsmith.preprocess import PreprocessOptions, preprocess
from svgsmith.report import Report, svg_stats
from svgsmith.verify import run_loop

# Mode → engine label and the preset used when --mode is given explicitly.
_ENGINE = {"binary": "potrace", "color": "vtracer", "pixel": "vtracer"}
_MODE_PRESET = {"binary": "logo", "color": "illustration", "pixel": "pixel"}
MODES = ("auto", *_MODE_PRESET)


@dataclass(frozen=True)
class ConvertOptions:
    """Options for :func:`convert` (mirrors the ``svgsmith convert`` flags)."""

    mode: str = "auto"
    quality: float = 0.9
    max_iters: int = 4
    editable: bool = True
    out: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0, 1], got {self.quality}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")


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
            denoise=False, quantize=True, palette_size=48, remove_background=False
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
    prepared = preprocess(image, _preprocess_opts(classification.mode))

    svg, result = run_loop(
        prepared,
        classification,
        quality=opts.quality,
        max_iters=opts.max_iters,
        editable=opts.editable,
        reference=image,  # score against the true original, not the preprocessed image
    )

    output = _output_path(input_path, opts.out)
    report = Report(
        output=output,
        mode_used=classification.mode,
        engine=_ENGINE[classification.mode],
        preset=classification.preset,
        iterations=result.iterations,
        similarity=result.best_score,
        passed_threshold=result.best_score >= opts.quality,
        svg=svg_stats(svg),
        warnings=list(classification.warnings),
    )
    return svg, report
