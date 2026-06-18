"""Raster preprocessing applied before tracing.

Pure image-to-image transforms (Pillow + numpy, no engine calls) that improve
trace quality and reduce path count. Each step is individually toggleable via
:class:`PreprocessOptions` and defaults to a sensible value. The output feeds
the T2 engines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from svgsmith.engines.base import ImageInput, load_image


@dataclass(frozen=True)
class PreprocessOptions:
    """Toggles and parameters for :func:`preprocess` (all default sensible)."""

    upscale: bool = True
    min_dimension: int = 64  # upscale only when the shorter side is below this

    denoise: bool = True
    median_size: int = 3  # odd window for the median filter

    flatten: bool = False  # edge-preserving color flattening (bilateral)
    flatten_sigma: float = 0.04  # color sigma; higher = flatter regions

    quantize: bool = True
    palette_size: int = 16  # target palette; T3 preset can inform this

    remove_background: bool = True
    bg_tolerance: int = 24  # per-channel tolerance for the background color


def upscale_tiny(img: Image.Image, min_dimension: int) -> Image.Image:
    """Nearest-neighbor upscale so the shorter side reaches ``min_dimension``.

    Returns the image unchanged when it is already large enough. Nearest-neighbor
    keeps pixel-art edges hard.
    """
    width, height = img.size
    shortest = min(width, height)
    if shortest >= min_dimension or shortest == 0:
        return img
    factor = -(-min_dimension // shortest)  # ceil division
    return img.resize((width * factor, height * factor), Image.Resampling.NEAREST)


def denoise(img: Image.Image, median_size: int) -> Image.Image:
    """Median-filter the color channels to drop isolated speckles.

    The alpha channel, if any, is preserved untouched.
    """
    size = median_size if median_size % 2 == 1 else median_size + 1
    filtered = img.convert("RGB").filter(ImageFilter.MedianFilter(size))
    if img.mode == "RGBA":
        filtered = filtered.convert("RGBA")
        filtered.putalpha(img.getchannel("A"))
    return filtered


def flatten_colors(img: Image.Image, sigma_color: float) -> Image.Image:
    """Edge-preserving bilateral smoothing to flatten color variation.

    Softens gradients and texture *within* regions while keeping edges sharp, so
    color-variation-heavy art (painterly, shaded lettering) traces into clean flat
    regions instead of shattering into many small facets — also cutting path count
    and file size. The alpha channel, if any, is preserved untouched.
    """
    from skimage.restoration import denoise_bilateral

    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    smoothed = denoise_bilateral(rgb, sigma_color=sigma_color, sigma_spatial=4, channel_axis=2)
    out = Image.fromarray((smoothed * 255.0).round().astype(np.uint8), "RGB")
    if img.mode == "RGBA":
        out = out.convert("RGBA")
        out.putalpha(img.getchannel("A"))
    return out


def quantize_colors(img: Image.Image, palette_size: int) -> Image.Image:
    """Median-cut quantization to at most ``palette_size`` colors.

    Dithering is disabled so no new colors are introduced; the alpha channel, if
    any, is preserved.
    """
    quantized = (
        img.convert("RGB")
        .quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        .convert("RGB")
    )
    if img.mode == "RGBA":
        quantized = quantized.convert("RGBA")
        quantized.putalpha(img.getchannel("A"))
    return quantized


def remove_background(img: Image.Image, tolerance: int) -> Image.Image:
    """Flood-fill the dominant corner color from the border to transparency.

    Only background pixels connected to the image edge are cleared, so interior
    regions that happen to match the background color are kept opaque.
    """
    rgba = np.array(img.convert("RGBA"))
    height, width = rgba.shape[:2]
    rgb = rgba[:, :, :3].astype(np.int16)

    corners = [
        tuple(rgba[0, 0, :3]),
        tuple(rgba[0, width - 1, :3]),
        tuple(rgba[height - 1, 0, :3]),
        tuple(rgba[height - 1, width - 1, :3]),
    ]
    background = np.array(max(set(corners), key=corners.count), dtype=np.int16)

    close = np.abs(rgb - background).max(axis=2) <= tolerance

    visited = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if close[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if close[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and close[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    rgba[visited, 3] = 0
    return Image.fromarray(rgba, "RGBA")


def preprocess(image: ImageInput, opts: PreprocessOptions | None = None) -> Image.Image:
    """Run the enabled preprocessing steps in order and return an RGBA image.

    Order: tiny-input upscale → denoise → quantize → background removal.
    Background removal runs last so the cleared alpha is not quantized away.
    """
    opts = opts or PreprocessOptions()
    img = load_image(image, "RGBA")

    if opts.upscale:
        img = upscale_tiny(img, opts.min_dimension)
    if opts.denoise:
        img = denoise(img, opts.median_size)
    if opts.flatten:
        img = flatten_colors(img, opts.flatten_sigma)
    if opts.quantize:
        img = quantize_colors(img, opts.palette_size)
    if opts.remove_background:
        img = remove_background(img, opts.bg_tolerance)

    return img.convert("RGBA")
