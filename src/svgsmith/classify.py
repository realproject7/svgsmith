"""Input classification for ``--mode auto``.

Deterministic, heuristic-only (no ML) selection of an engine *mode* and a
canonical *preset* for an input image. Downstream the pipeline maps:

- ``binary`` → Potrace (``logo`` / ``icon`` preset)
- ``color``  → VTracer (``illustration`` preset)
- ``pixel``  → VTracer (``pixel`` preset)

Photographic inputs classify as ``color`` and carry a raw warning string; this
module emits the string only — #8 assembles the canonical report, so the report
schema is intentionally not imported here.
"""

from __future__ import annotations

from typing import NamedTuple

from PIL import Image, ImageFilter, ImageOps

from svgsmith.engines.base import ImageInput, load_image

# Heuristic thresholds. The fixture feature values sit well clear of every
# boundary, so classification stays stable across Pillow versions.
PIXEL_MAX_DIM = 64  # pixel art lives on a tiny canvas
PIXEL_MAX_PALETTE = 32  # ...with a small palette
PHOTO_MIN_PALETTE = 64  # rich palette / gradients read as photographic
BINARY_MAX_PALETTE = 3  # monochrome-ish ink-on-paper
BINARY_MIN_EDGE_DENSITY = 0.02  # ...with sharp edges
EDGE_MAGNITUDE_CUTOFF = 40  # grayscale edge strength counted as a "strong" edge

PHOTO_WARNING = "photographic gradients; vectorization may bloat"


class Classification(NamedTuple):
    """Result of :func:`classify`.

    Unpacks as ``(mode, preset, warnings)``; callers that only need the engine
    selection can read ``.mode`` / ``.preset``.
    """

    mode: str  # "binary" | "color" | "pixel"
    preset: str  # canonical preset name from engines.PRESETS
    warnings: tuple[str, ...]


def _palette_size(img: Image.Image) -> int:
    """Distinct colors after a coarse posterize that suppresses codec noise."""
    posterized = ImageOps.posterize(img, 4)
    colors = posterized.getcolors(maxcolors=1 << 16)
    return len(colors) if colors is not None else (1 << 16)


def _edge_density(img: Image.Image) -> float:
    """Fraction of pixels whose grayscale edge magnitude exceeds the cutoff."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    strong = sum(histogram[EDGE_MAGNITUDE_CUTOFF:])
    total = img.width * img.height
    return strong / total if total else 0.0


def classify(image: ImageInput) -> Classification:
    """Classify ``image`` into an engine mode + canonical preset.

    Returns a :class:`Classification`. Photographic inputs classify as ``color``
    with :data:`PHOTO_WARNING` attached.
    """
    img = load_image(image, "RGB")
    max_dim = max(img.size)
    palette = _palette_size(img)

    # Monochrome-ish + sharp edges → line art. Checked before the pixel-art
    # branch so a tiny 2-3 color icon reaches `binary`/`icon` instead of being
    # captured as pixel art (pixel art carries more than a couple of hues).
    if palette <= BINARY_MAX_PALETTE and _edge_density(img) >= BINARY_MIN_EDGE_DENSITY:
        preset = "icon" if max_dim <= PIXEL_MAX_DIM else "logo"
        return Classification("binary", preset, ())

    # Tiny canvas + small (but non-monochrome) palette → pixel art.
    if max_dim <= PIXEL_MAX_DIM and palette <= PIXEL_MAX_PALETTE:
        return Classification("pixel", "pixel", ())

    # Rich palette / gradients → photographic. Still vectorizable as color, but
    # flag the likely bloat for the report to surface.
    if palette >= PHOTO_MIN_PALETTE:
        return Classification("color", "illustration", (PHOTO_WARNING,))

    # Everything else: flat multi-color illustration.
    return Classification("color", "illustration", ())
