"""Tests for the raster preprocessing steps."""

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from svgsmith.preprocess import (
    PreprocessOptions,
    preprocess,
    quantize_colors,
    remove_background,
    upscale_tiny,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOISY = FIXTURES / "noisy.png"
FLAT_BG = FIXTURES / "flat_bg.png"
PIXEL = FIXTURES / "pixel.png"


def _unique_colors(img: Image.Image) -> int:
    rgb = np.asarray(img.convert("RGB")).reshape(-1, 3)
    return len(np.unique(rgb, axis=0))


def _small_speckle_count(img: Image.Image, max_size: int = 4) -> int:
    """Count connected components (4-conn) of non-dominant pixels up to max_size."""
    arr = np.asarray(img.convert("RGB"))
    flat = arr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    dominant = colors[counts.argmax()]
    mask = np.any(arr != dominant, axis=2)

    height, width = mask.shape
    seen = np.zeros_like(mask)
    small = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            seen[y, x] = True
            queue = deque([(y, x)])
            size = 0
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if size <= max_size:
                small += 1
    return small


def test_preprocess_reduces_colors_and_speckles():
    original = Image.open(NOISY).convert("RGB")
    processed = preprocess(NOISY)

    assert _unique_colors(processed) < _unique_colors(original)
    assert _small_speckle_count(processed) < _small_speckle_count(original)


def test_quantize_respects_palette_size():
    processed = quantize_colors(Image.open(NOISY).convert("RGB"), palette_size=8)
    assert _unique_colors(processed) <= 8


def test_background_removal_yields_transparency():
    processed = remove_background(Image.open(FLAT_BG).convert("RGBA"), tolerance=24)
    arr = np.asarray(processed)
    # Every corner becomes transparent; the centered shape stays opaque.
    assert arr[0, 0, 3] == 0
    assert arr[0, -1, 3] == 0
    assert arr[-1, 0, 3] == 0
    assert arr[-1, -1, 3] == 0
    assert arr[arr.shape[0] // 2, arr.shape[1] // 2, 3] == 255


def test_upscale_only_below_min_dimension():
    tiny = Image.open(PIXEL)  # 16x16
    upscaled = upscale_tiny(tiny, min_dimension=64)
    assert min(upscaled.size) >= 64

    big = Image.open(FLAT_BG)  # 128x128, already large enough
    assert upscale_tiny(big, min_dimension=64).size == big.size


def test_steps_are_individually_toggleable():
    original = Image.open(NOISY)
    # With every step disabled, only the RGBA normalization is applied.
    untouched = preprocess(NOISY, PreprocessOptions(
        upscale=False, denoise=False, quantize=False, remove_background=False
    ))
    assert untouched.size == original.size
    assert _unique_colors(untouched) == _unique_colors(original)
    assert untouched.mode == "RGBA"


def test_preprocess_returns_rgba():
    assert preprocess(FLAT_BG).mode == "RGBA"
