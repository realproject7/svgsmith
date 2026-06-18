"""Shared types for the tracing engines.

Defines the :class:`Preset` parameter bundle, the canonical named presets, and
the :class:`Tracer` protocol that every engine adapter implements. The rest of
the pipeline depends only on ``trace(image, preset) -> str`` and never calls an
engine library directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image

ImageInput = str | Path | Image.Image


@dataclass(frozen=True)
class Preset:
    """Engine parameters for one tracing profile.

    A single bundle carries both color (VTracer) and binary (Potrace) knobs so
    a caller can hand the same preset to either adapter; each adapter reads only
    the fields it understands.
    """

    name: str

    # Color (VTracer) parameters.
    color_mode: str = "color"  # "color" | "binary"
    color_precision: int = 6
    layer_difference: int = 16
    filter_speckle: int = 4
    corner_threshold: int = 60
    length_threshold: float = 4.0
    splice_threshold: int = 45
    path_precision: int = 8
    curve_mode: str = "spline"  # "spline" | "polygon" | "none"
    hierarchical: str = "stacked"  # "stacked" | "cutout"

    # Binary (Potrace) parameters.
    turdsize: int = 2  # speckle suppression: drop specks up to N pixels
    alphamax: float = 1.0  # corner threshold
    opttolerance: float = 0.2  # curve-fitting tolerance
    threshold: float = 0.5  # luminance cut for binarization, in [0, 1]


# Canonical preset names. #4 (classifier) and #8 (report `preset`) reference
# exactly this list — do not add variants without updating those tickets.
PRESETS: dict[str, Preset] = {
    "logo": Preset(
        name="logo",
        color_precision=6,
        filter_speckle=4,
        corner_threshold=60,
        turdsize=2,
        alphamax=1.0,
        opttolerance=0.2,
    ),
    "icon": Preset(
        name="icon",
        color_precision=8,
        filter_speckle=2,
        corner_threshold=40,
        turdsize=1,
        alphamax=1.0,
        opttolerance=0.2,
    ),
    "illustration": Preset(
        name="illustration",
        color_precision=8,
        layer_difference=8,
        filter_speckle=4,
        corner_threshold=60,
        turdsize=2,
        alphamax=1.0,
        opttolerance=0.4,
    ),
    "pixel": Preset(
        name="pixel",
        color_precision=8,
        filter_speckle=0,
        curve_mode="none",
        corner_threshold=180,
        turdsize=0,
        alphamax=0.0,
        opttolerance=0.0,
    ),
}


def get_preset(name: str) -> Preset:
    """Return the named canonical preset, or raise ``ValueError``."""
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {known}") from None


def load_image(image: ImageInput, mode: str = "RGBA") -> Image.Image:
    """Open ``image`` (path or PIL image) and convert it to ``mode``."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    return img.convert(mode)


@runtime_checkable
class Tracer(Protocol):
    """Uniform tracing interface implemented by every engine adapter."""

    def trace(self, image: ImageInput, preset: Preset) -> str:
        """Trace ``image`` with ``preset`` and return an SVG document string."""
        ...
