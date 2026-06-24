"""Tests for the raster preprocessing steps."""

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from svgsmith.preprocess import (
    PreprocessOptions,
    preprocess,
    quantize_colors,
    quantize_kmeans,
    remove_background,
    solid_background,
    upscale_tiny,
    upscale_to,
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


def test_upscale_to_reaches_target_and_is_capped():
    # Reaches the target long-edge for a normal low-res input.
    assert max(upscale_to(Image.new("RGB", (640, 640)), 2048).size) == 2048
    # Capped at max_factor so a tiny input is not blown up to a huge trace.
    assert max(upscale_to(Image.new("RGB", (100, 100)), 2048, max_factor=4.0).size) == 400
    # No-op when already at/above the target.
    assert upscale_to(Image.new("RGB", (2048, 2048)), 2048).size == (2048, 2048)


def test_quantize_kmeans_cuts_colors_and_anchors_black():
    # A many-color gradient with a bold black band (a stand-in outline).
    grad = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    arr = np.stack([grad, np.full((200, 200), 120, np.uint8), 255 - grad], axis=2)
    arr[90:110, :] = (0, 0, 0)
    out = quantize_kmeans(Image.fromarray(arr, "RGB"), k=5)
    colors = np.unique(np.asarray(out.convert("RGB")).reshape(-1, 3), axis=0)
    assert len(colors) <= 6  # k clusters + the pure-black anchor
    assert any((c == 0).all() for c in colors)  # the outline stays pure #000000


def test_quantize_kmeans_skips_anchor_when_mostly_dark():
    # A near-black image: the anchor must NOT fire, or everything collapses to one
    # black fill; k-means keeps the dark tones distinct instead.
    arr = np.random.default_rng(0).integers(0, 40, (64, 64, 3), dtype=np.uint8)
    out = quantize_kmeans(Image.fromarray(arr, "RGB"), k=4)
    colors = np.unique(np.asarray(out.convert("RGB")).reshape(-1, 3), axis=0)
    assert len(colors) > 1


def _black_fraction(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB")).reshape(-1, 3)
    return float((arr.sum(axis=1) < 30).mean())


def test_quantize_kmeans_linework_snaps_dark_thin_lines_to_black():
    # A green field crossed by a thin DARK line whose luma is above the near-black
    # anchor (so only the black-hat linework pass can catch it). detect_linework
    # turns the line into clean black; without it the line stays a dark-green fill.
    arr = np.full((120, 120, 3), (40, 165, 70), dtype=np.uint8)  # green, luma ~120
    arr[:, 57:63] = (55, 90, 55)  # thin darker-green line, luma ~80 (> anchor 45)
    img = Image.fromarray(arr, "RGB")
    plain = quantize_kmeans(img, k=4, detect_linework=False)
    inked = quantize_kmeans(img, k=4, detect_linework=True)
    assert _black_fraction(inked) > _black_fraction(plain)
    assert _black_fraction(plain) == 0.0  # the line is above the near-black anchor


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


def _pink_ear_fixture() -> Image.Image:
    """Pink wall with a subject blob that matches the wall color but is enclosed
    by a darker outline — the #53 "pink ear on a pink wall" case."""
    pink = (235, 170, 200)
    outline = (40, 40, 40)
    arr = np.full((60, 60, 3), pink, dtype=np.uint8)
    # A ring of dark outline (rows/cols 20..40) with a pink-filled interior.
    arr[20:41, 20:41] = outline
    arr[24:37, 24:37] = pink  # enclosed subject region, same color as the background
    return Image.fromarray(arr, "RGB")


def test_solid_background_keeps_enclosed_same_color_subject():
    img = _pink_ear_fixture()
    out = np.asarray(solid_background(img, tolerance=24, target_color="#FFFFFF"))

    # The true edge-connected background flattens to the target white.
    assert tuple(out[0, 0]) == (255, 255, 255)
    # The enclosed pink region, though it shares the wall color, is NOT
    # edge-connected (the dark outline blocks the flood fill) — it survives.
    assert tuple(out[30, 30]) == (235, 170, 200)
    # The dark outline (subject) is also preserved.
    assert tuple(out[20, 30]) == (40, 40, 40)


def test_solid_background_auto_uses_median_color():
    img = _pink_ear_fixture()
    out = np.asarray(solid_background(img, tolerance=24))
    # Auto mode repaints the bg to the detected (pink) color, enclosed blob kept.
    assert tuple(out[0, 0]) == (235, 170, 200)
    assert tuple(out[30, 30]) == (235, 170, 200)
    assert tuple(out[20, 30]) == (40, 40, 40)


def test_merge_small_regions_noise_de_keeps_distinct_marks():
    """The detail-aware merge (#economical-fidelity): a small region merges into its
    neighbour only when that neighbour is closer than ``noise_de`` (ΔE76). A distinct-
    colour small mark (a dot/stipple speck, large ΔE) is KEPT; a near-colour speck
    (anti-alias/noise, small ΔE) is absorbed. ``noise_de`` == 0 merges everything."""
    from svgsmith.preprocess import _merge_small_regions

    # 6x6 grid: region 0 = background; region 1 = small DISTINCT-colour strip (top-left);
    # region 2 = small NEAR-colour strip (bottom-right). Both are below min_area_px.
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[0, 0:2] = 1  # 2 px, distinct
    labels[5, 4:6] = 2  # 2 px, near-bg
    region_lab = np.array(
        [[50.0, 0.0, 0.0], [50.0, 60.0, 60.0], [52.0, 2.0, 2.0]]  # bg  # far  # near
    )
    min_area_px = 5.0

    # Legacy behaviour (noise_de=0): both small regions merge into the background.
    roots0 = _merge_small_regions(labels, region_lab, min_area_px, 0.0)
    assert roots0[1] == 0
    assert roots0[2] == 0

    # Detail-aware (noise_de=13): the distinct mark is kept (root is itself); the near
    # speckle still merges into the background.
    roots = _merge_small_regions(labels, region_lab, min_area_px, 13.0)
    assert roots[1] == 1  # distinct-colour mark preserved
    assert roots[2] == 0  # near-colour noise absorbed


def test_coverage_noise_de_dial_orders_detail_levels():
    """The coverage detail-aware merge threshold rises as the detail dial flattens:
    'high' keeps the most texture (lowest ΔE), 'poster' collapses the most (highest)."""
    from svgsmith.pipeline import COVERAGE_NOISE_DE

    assert (
        COVERAGE_NOISE_DE["high"]
        < COVERAGE_NOISE_DE["normal"]
        < COVERAGE_NOISE_DE["clean"]
        < COVERAGE_NOISE_DE["poster"]
    )
