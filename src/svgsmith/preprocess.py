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
from PIL import Image, ImageColor, ImageFilter

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

    solid_background: bool = False  # replace the background with one clean solid color
    background_tolerance: int = 32  # per-channel tolerance for the edge-flood-fill bg region
    background_color: str | None = None  # exact bg color (#RRGGBB/named); None = auto median

    uniform_outline: bool = False  # force a constant-width outline band (opt-in)
    outline_width: int = 8  # band half-width in px, used when uniform_outline is on

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


def _edge_flood_fill_mask(rgba: np.ndarray, tolerance: int) -> np.ndarray:
    """Boolean mask of background pixels reachable by flood-fill from the borders.

    The background color is the dominant image corner; a pixel counts as background
    when it is within ``tolerance`` (per channel) of that color AND is connected to
    the image edge through other such pixels. Interior regions that merely *match*
    the background color but are enclosed by the subject's outline are therefore
    NOT marked — they stay subject. ``rgba`` is an HxWx(3 or 4) uint8 array; only
    the RGB channels are used. Returns an HxW bool mask (True = background).
    """
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

    return visited


def _parse_color(color: str) -> tuple[int, int, int]:
    """Parse a hex (``#RRGGBB``) or named color into an RGB triple.

    Accepts anything :func:`PIL.ImageColor.getrgb` understands (hex, named colors).
    Raises ``ValueError`` with a clear message on an unrecognized color.
    """
    try:
        return ImageColor.getrgb(color)[:3]
    except ValueError as exc:
        raise ValueError(f"invalid background color {color!r}: {exc}") from exc


def solid_background(
    img: Image.Image, tolerance: int, target_color: str | None = None
) -> Image.Image:
    """Isolate the subject and repaint everything else as one clean solid color.

    The background is the region reachable by edge flood-fill from the image
    borders (see :func:`_edge_flood_fill_mask`); the subject is everything not
    edge-connected, so a subject region that happens to share the background color
    but is enclosed by an outline (a pink ear on a pink wall) is kept, not punched
    into a hole. All background pixels are then flattened to one solid color —
    ``target_color`` when given (an exact ``#RRGGBB`` or named color), otherwise the
    median background color — removing texture, grain, streaks, and stray specks
    while the subject is left untouched.
    """
    rgb = np.array(img.convert("RGB"))
    background_mask = _edge_flood_fill_mask(rgb, tolerance)

    out = rgb.copy()
    if target_color is not None:
        fill = np.array(_parse_color(target_color), dtype=np.uint8)
        out[background_mask] = fill
    else:
        bg_pixels = rgb[background_mask]
        if bg_pixels.size:
            out[background_mask] = np.median(bg_pixels, axis=0).astype(np.uint8)
    return Image.fromarray(out, "RGB").convert(img.mode)


def uniform_outline(img: Image.Image, width: int, bg_tolerance: int = 18) -> Image.Image:
    """Paint a constant-width band of the darkest color around the silhouette.

    Premium cartoon SVGs read with an even outline because the dark "line" is a
    constant geometric inset of one silhouette. A trace builds the outline as the
    byproduct of two independently-traced contours, so its width wobbles. Forcing a
    uniform dark band on the silhouette boundary *before* tracing makes the outer
    outline even by construction.

    Opt-in only: it assumes the art HAS a dark outline around a solid-background
    figure. On line art or outline-free illustrations it would add a wrong border.
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes

    rgba = np.array(img.convert("RGBA"))
    rgb = rgba[:, :, :3].astype(int)
    corners = [
        tuple(rgba[0, 0, :3]),
        tuple(rgba[0, -1, :3]),
        tuple(rgba[-1, 0, :3]),
        tuple(rgba[-1, -1, :3]),
    ]
    background = np.array(max(set(corners), key=corners.count), dtype=int)
    silhouette = binary_fill_holes(np.abs(rgb - background).max(axis=2) > bg_tolerance)
    eroded = binary_erosion(silhouette, iterations=max(1, width))
    band = silhouette & ~eroded
    flat = rgba[:, :, :3].reshape(-1, 3)
    outline_color = flat[int(np.argmin(flat.sum(axis=1)))]
    rgba[band, :3] = outline_color
    rgba[band, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def remove_background(img: Image.Image, tolerance: int) -> Image.Image:
    """Flood-fill the dominant corner color from the border to transparency.

    Only background pixels connected to the image edge are cleared, so interior
    regions that happen to match the background color are kept opaque.
    """
    rgba = np.array(img.convert("RGBA"))
    background_mask = _edge_flood_fill_mask(rgba, tolerance)
    rgba[background_mask, 3] = 0
    return Image.fromarray(rgba, "RGBA")


def preprocess(image: ImageInput, opts: PreprocessOptions | None = None) -> Image.Image:
    """Run the enabled preprocessing steps in order and return an RGBA image.

    Order: solid background → tiny-input upscale → denoise → flatten → quantize →
    outline band → background removal. Solid-background runs first so the subject
    is detected from the original colors before any flattening/quantization.
    """
    opts = opts or PreprocessOptions()
    img = load_image(image, "RGBA")

    if opts.solid_background:
        img = solid_background(img, opts.background_tolerance, opts.background_color)
    if opts.upscale:
        img = upscale_tiny(img, opts.min_dimension)
    if opts.denoise:
        img = denoise(img, opts.median_size)
    if opts.flatten:
        img = flatten_colors(img, opts.flatten_sigma)
    if opts.quantize:
        img = quantize_colors(img, opts.palette_size)
    if opts.uniform_outline:
        img = uniform_outline(img, opts.outline_width)
    if opts.remove_background:
        img = remove_background(img, opts.bg_tolerance)

    return img.convert("RGBA")
