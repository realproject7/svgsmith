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

    # Pre-trace supersampling (#60): bicubic-upscale the working image to this
    # long-edge before quantize+trace, so vtracer fits smooth curves to a finer
    # grid (a 640px JPEG traces staircased contours; 2048 traces smooth ones).
    # 0 disables. Only upscales (never downscales) and skips when already >= target.
    trace_resolution: int = 0

    denoise: bool = True
    median_size: int = 3  # odd window for the median filter

    flatten: bool = False  # edge-preserving color flattening (bilateral)
    flatten_sigma: float = 0.04  # color sigma; higher = flatter regions
    flatten_spatial: int = 4  # bilateral spatial sigma; higher = wider smoothing

    quantize: bool = True
    palette_size: int = 16  # target palette; T3 preset can inform this

    # Fixed-K k-means palette (#41): when True, quantize with perceptual k-means to
    # ``palette_k`` colors with a near-black/outline anchor instead of median-cut.
    # Snaps each flat region to one clean color and keeps the outline pure black —
    # median-cut instead leaves near-duplicate tints and a muddy near-black outline.
    kmeans_palette: bool = False
    palette_k: int = 9  # target color count for the k-means quantizer
    black_anchor_luma: int = 45  # pixels darker than this snap to pure #000000
    detect_linework: bool = False  # also snap dark thin lines/creases to black (black-hat)
    linework_radius: int = 9  # black-hat structuring-element size (px, at trace resolution)
    linework_threshold: float = 28.0  # min black-hat response to count as a dark line

    # Perceptual-coverage palette (#65, Tier 1): when True, quantize by covering the
    # occupied CIELAB colour volume with as many flat colours as the content needs at
    # a fixed perceptual step (``coverage_step``, ΔE76). The palette count is not
    # chosen — it emerges from content, so a smooth gradient keeps the many low-step
    # bands a fixed-K/median-cut path crushes into visible banding. Gated to smooth
    # continuous-tone inputs by the pipeline; flat art never takes this path.
    coverage_palette: bool = False
    coverage_step: float = 4.0  # perceptual step (ΔE76) between adjacent retained colours
    coverage_fraction: float = 0.99  # cover this share of pixel mass before stopping
    coverage_min_area: float = 0.0006  # drop colours below this share of pixels (speckle)
    coverage_max_colors: int = 256  # hard safety cap on emitted colours
    # Spatial region cleanup (#67, Tier 2): after coverage quantization, merge by
    # connected-component *area* rather than by global colour mass. A gradient band is
    # thin-but-long (large area → kept) while an anti-alias sliver / linework fringe is
    # thin-but-short (sub-area → absorbed into its nearest-ΔE neighbour, which collapses
    # ragged dark linework into clean continuous outline regions and keeps flat art
    # economical). This reaches gradients *inside* hard-edged shapes (#69) and fixes
    # over-split outlines (#66) — the cases the narrow Tier 1 gate can't. ``False`` keeps
    # pure Tier 1 behaviour (the safe-revert is this flag alone).
    coverage_region_cleanup: bool = False
    coverage_region_min_area: float = 0.0004  # merge components below this share of pixels
    coverage_region_max_px: int = 2000  # px ceiling on the threshold (protect small marks)

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


def upscale_to(img: Image.Image, target_long_edge: int, max_factor: float = 4.0) -> Image.Image:
    """Bicubic-upscale so the longer side reaches ``target_long_edge``.

    Supersampling before tracing: vtracer fits curves to the pixel grid, so a
    larger grid yields smoother contours and fewer micro-paths. Bicubic (not
    nearest) so anti-aliased edges stay smooth; quantization downstream snaps the
    resulting fringe back onto flat colors. Never downscales and is a no-op when
    the image already reaches the target. The upscale is capped at ``max_factor``
    so a tiny input is not blown up to a multi-megapixel trace (a 64px image would
    otherwise become 2048px = a 32× trace).
    """
    width, height = img.size
    longest = max(width, height)
    if target_long_edge <= 0 or longest >= target_long_edge or longest == 0:
        return img
    scale = min(target_long_edge / longest, max_factor)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return img.resize(new_size, Image.Resampling.BICUBIC)


def _assign_nearest(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for each pixel, one centroid pass at a time.

    Computed centroid-by-centroid (k passes over an Nx1 distance vector) rather
    than materializing an NxKx3 tensor, so it stays cheap on multi-megapixel
    upscaled images.
    """
    best_dist = np.full(len(pixels), np.inf, dtype=np.float32)
    best_idx = np.zeros(len(pixels), dtype=np.int32)
    for i, c in enumerate(centroids):
        dist = ((pixels - c) ** 2).sum(axis=1)
        closer = dist < best_dist
        best_dist = np.where(closer, dist, best_dist)
        best_idx = np.where(closer, i, best_idx)
    return best_idx


def _linework_mask(luma2d: np.ndarray, radius: int, threshold: float) -> np.ndarray:
    """Boolean mask of dark, thin line structures (outlines, internal creases).

    Black-hat = morphological closing minus the image, which highlights dark
    features narrower than the structuring element while leaving large dark regions
    (a slate head) untouched. This separates a feather-crease *line* — dark, thin,
    surrounded by a lighter color — from a dark *region*, so the linework can be
    snapped to one clean black outline layer instead of bleeding into dark-tint
    fills the way a plain color trace does.
    """
    from scipy.ndimage import grey_closing

    blackhat = grey_closing(luma2d, size=(radius, radius)) - luma2d
    return blackhat > threshold


def quantize_kmeans(
    img: Image.Image,
    k: int,
    black_anchor_luma: int = 45,
    *,
    detect_linework: bool = False,
    linework_radius: int = 9,
    linework_threshold: float = 28.0,
    seed: int = 7,
    iters: int = 12,
    sample: int = 60000,
) -> Image.Image:
    """Fixed-K palette via k-means in RGB with a near-black/outline anchor.

    Pixels darker than ``black_anchor_luma`` (luma) are forced to pure ``#000000``
    and **excluded** from clustering, so the line-art outline collapses to one
    clean pure-black layer and small accent colors (a red beak, a pink highlight)
    are not absorbed into a muddy dark cluster. With ``detect_linework`` the anchor
    also captures dark *thin* structures (outlines and internal feather creases)
    via a black-hat transform, so they become clean black lines instead of
    dark-tint fills. The remaining pixels are clustered to ``k`` centroids
    (k-means++ seeding on a random sample, then Lloyd iterations) and every pixel
    is snapped to its centroid. The anchor is skipped when the black mask dominates
    the image (a genuinely dark picture, not an outline), so dark art is not crushed.

    Dithering is never introduced and the alpha channel, if any, is preserved.
    """
    rng = np.random.default_rng(seed)
    rgba = np.array(img.convert("RGBA"))
    height, width = rgba.shape[:2]
    flat = rgba[:, :, :3].astype(np.float32).reshape(-1, 3)
    luma = flat @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    black = luma < black_anchor_luma
    if detect_linework:
        ink = _linework_mask(luma.reshape(height, width), linework_radius, linework_threshold)
        black = black | ink.reshape(-1)
    # Only treat near-black as a separate outline layer when it is a minority of
    # the image; on a mostly-dark picture, cluster it like any other color.
    black_fraction = float(black.mean())
    anchor = 0.0 < black_fraction < 0.35
    cluster_mask = ~black if anchor else np.ones(len(flat), dtype=bool)
    subject = flat[cluster_mask]

    out = flat.copy()
    if len(subject) and k > 0:
        pool = (
            subject
            if len(subject) <= sample
            else subject[rng.choice(len(subject), sample, replace=False)]
        )
        # k-means++-style spread seeding.
        centroids = pool[rng.integers(len(pool))][None, :]
        while len(centroids) < min(k, len(pool)):
            dist = _nearest_dist(pool, centroids)
            centroids = np.vstack([centroids, pool[dist.argmax()]])
        for _ in range(iters):
            labels = _assign_nearest(pool, centroids)
            for c in range(len(centroids)):
                members = labels == c
                if members.any():
                    centroids[c] = pool[members].mean(axis=0)
        out[cluster_mask] = centroids[_assign_nearest(subject, centroids)]
    if anchor:
        out[black] = 0.0

    quantized = np.clip(out, 0, 255).astype(np.uint8).reshape(height, width, 3)
    rgba[:, :, :3] = quantized
    return Image.fromarray(rgba, "RGBA").convert(img.mode)


def _nearest_dist(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared distance from each pixel to its nearest centroid (for ++ seeding)."""
    best = np.full(len(pixels), np.inf, dtype=np.float32)
    for c in centroids:
        best = np.minimum(best, ((pixels - c) ** 2).sum(axis=1))
    return best


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


def flatten_colors(img: Image.Image, sigma_color: float, sigma_spatial: int = 4) -> Image.Image:
    """Edge-preserving bilateral smoothing to flatten color variation.

    Softens gradients and texture *within* regions while keeping edges sharp, so
    color-variation-heavy art (painterly, shaded lettering) traces into clean flat
    regions instead of shattering into many small facets — also cutting path count
    and file size. The alpha channel, if any, is preserved untouched.
    """
    from skimage.restoration import denoise_bilateral

    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    smoothed = denoise_bilateral(
        rgb, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=2
    )
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


def _connected_regions(label_img: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components (4-connectivity) within equal-value runs of ``label_img``.

    Returns a contiguous ``region_label`` map (0..n-1) and the region count ``n``.
    """
    from scipy import ndimage

    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    out = np.zeros(label_img.shape, dtype=np.int64)
    next_id = 0
    for value in np.unique(label_img):
        comp, count = ndimage.label(label_img == value, structure=struct)
        if count:
            mask = comp > 0
            out[mask] = comp[mask] + next_id  # 1-based within this colour
            next_id += count
    # ``out`` now holds contiguous ids 1..next_id; shift to 0-based 0..next_id-1.
    return out - 1, next_id


def _adjacent_region_pairs(region_label: np.ndarray) -> np.ndarray:
    """Unique unordered adjacent region-id pairs (4-connectivity), vectorized."""
    horiz = np.stack([region_label[:, :-1].ravel(), region_label[:, 1:].ravel()], axis=1)
    vert = np.stack([region_label[:-1, :].ravel(), region_label[1:, :].ravel()], axis=1)
    pairs = np.concatenate([horiz, vert], axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return pairs
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)


def _merge_small_regions(
    region_label: np.ndarray, region_lab: np.ndarray, min_area_px: float
) -> np.ndarray:
    """Absorb every region below ``min_area_px`` into its lowest-ΔE live neighbour.

    Smallest-first (a region that grows past the threshold is kept). Returns ``roots``:
    the surviving region id for each original region id (apply as ``roots[region_label]``).
    The survivor keeps its own colour — merging only *repaints* the small region, so no
    new colours are introduced.
    """
    import heapq

    n = int(region_label.max()) + 1
    sizes = np.bincount(region_label.ravel(), minlength=n).astype(np.float64)
    parent = np.arange(n)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in _adjacent_region_pairs(region_label):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    lab = [region_lab[i].astype(np.float64).copy() for i in range(n)]
    heap = [(sizes[i], i) for i in range(n) if sizes[i] < min_area_px]
    heapq.heapify(heap)
    while heap:
        _, region = heapq.heappop(heap)
        root = find(region)
        if root != region or sizes[root] >= min_area_px:
            continue  # already merged, or grew past the threshold
        neighbours = {find(nb) for nb in adj[root]} - {root}
        if not neighbours:
            continue
        best = min(neighbours, key=lambda nb: float(np.linalg.norm(lab[root] - lab[nb])))
        parent[root] = best
        wa, wb = sizes[best], sizes[root]
        lab[best] = (lab[best] * wa + lab[root] * wb) / (wa + wb)
        sizes[best] += sizes[root]
        sizes[root] = 0.0
        adj[best] |= adj[root]
        adj[best].discard(best)
        if sizes[best] < min_area_px:
            heapq.heappush(heap, (sizes[best], best))

    return np.array([find(i) for i in range(n)], dtype=np.int64)


def quantize_coverage(
    img: Image.Image,
    step: float = 3.0,
    *,
    coverage: float = 0.99,
    min_area: float = 0.0006,
    max_colors: int = 256,
    region_cleanup: bool = False,
    region_min_area: float = 0.0004,
    region_max_px: int = 2000,
) -> Image.Image:
    """Perceptual-coverage quantization in CIELAB (#65).

    Greedy max-coverage: repeatedly take the most popular still-uncovered colour as a
    new palette entry and absorb every colour within ``step`` (ΔE76) of it, until
    ``coverage`` of the pixel mass is covered (or ``max_colors`` is hit). The palette
    size is therefore *emergent* — a flat region collapses to a couple of entries while
    a smooth gradient, being a long thin manifold in colour space, keeps many low-step
    bands. Each palette colour is the mean RGB of the pixels it absorbs (real colours,
    not synthetic centroids); entries below ``min_area`` of the image are merged into
    their nearest survivor to drop speckle. Dithering is never introduced; the alpha
    channel, if any, is preserved.
    """
    from skimage.color import rgb2lab

    rgba = np.array(img.convert("RGBA"))
    height, width = rgba.shape[:2]
    rgb = rgba[:, :, :3].reshape(-1, 3).astype(np.float64)
    lab = rgb2lab(rgba[:, :, :3] / 255.0).reshape(-1, 3)

    # Weighted histogram over 1-unit LAB bins (cheap, resolution-independent work set).
    qlab = np.round(lab).astype(np.int64)
    bins, inverse, counts = np.unique(qlab, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    binf = bins.astype(np.float64)
    total = float(counts.sum())
    # Mean RGB per bin, so a palette entry resolves to a real average colour.
    bin_rgb_sum = np.zeros((len(bins), 3))
    np.add.at(bin_rgb_sum, inverse, rgb)

    assigned = np.full(len(bins), -1, dtype=np.int64)
    centers_lab: list[np.ndarray] = []
    remaining = counts.astype(np.float64).copy()
    covered = 0.0
    while covered < coverage * total and len(centers_lab) < max_colors:
        idx = int(np.argmax(remaining))
        if remaining[idx] <= 0:
            break
        center = binf[idx]
        within = (np.linalg.norm(binf - center, axis=1) <= step) & (assigned < 0)
        ci = len(centers_lab)
        assigned[within] = ci
        centers_lab.append(center)
        covered += counts[within].sum()
        remaining[within] = 0

    if not centers_lab:  # degenerate (single colour) — nothing to do
        return img
    centers = np.array(centers_lab)
    # Tail colours beyond the coverage budget snap to their nearest palette entry.
    leftover = np.where(assigned < 0)[0]
    for bi in leftover:
        assigned[bi] = int(np.argmin(np.linalg.norm(centers - binf[bi], axis=1)))

    # Representative RGB per palette entry = mean RGB of its pixels.
    n = len(centers)
    rep_sum = np.zeros((n, 3))
    rep_cnt = np.zeros(n)
    np.add.at(rep_sum, assigned, bin_rgb_sum)
    np.add.at(rep_cnt, assigned, counts)
    rep_rgb = rep_sum / np.maximum(rep_cnt[:, None], 1.0)

    # Drop speckle entries (below min_area of the image) into their nearest survivor.
    keep = rep_cnt >= max(1.0, min_area * total)
    if not keep.all() and keep.any():
        survivors = np.where(keep)[0]
        surv_lab = centers[survivors]
        remap = np.arange(n)
        for ci in np.where(~keep)[0]:
            remap[ci] = survivors[int(np.argmin(np.linalg.norm(surv_lab - centers[ci], axis=1)))]
        assigned = remap[assigned]
        rep_sum = np.zeros((n, 3))
        rep_cnt = np.zeros(n)
        np.add.at(rep_sum, assigned, bin_rgb_sum)
        np.add.at(rep_cnt, assigned, counts)
        rep_rgb = np.divide(rep_sum, np.maximum(rep_cnt[:, None], 1.0))

    pixel_palette = assigned[inverse]  # palette index per pixel (flat)

    # Tier 2 (#67): merge by connected-component *area*, not global colour mass. This
    # reaches gradients inside hard-edged shapes and collapses ragged dark linework into
    # clean continuous outlines, while keeping flat art economical.
    if region_cleanup:
        region_label, n_regions = _connected_regions(pixel_palette.reshape(height, width))
        if n_regions > 1:
            # Palette index per region (regions are constant-colour, so any member works).
            region_pal = np.zeros(n_regions, dtype=np.int64)
            region_pal[region_label.ravel()] = pixel_palette
            region_lab = centers[region_pal]  # ΔE merge distance uses the palette colour
            min_area_px = min(float(region_min_area) * total, float(region_max_px))
            min_area_px = max(min_area_px, 1.0)
            roots = _merge_small_regions(region_label, region_lab, min_area_px)
            pixel_palette = region_pal[roots[region_label]].ravel()

    out_rgb = rep_rgb[pixel_palette].round().clip(0, 255).astype(np.uint8)
    rgba[:, :, :3] = out_rgb.reshape(height, width, 3)
    return Image.fromarray(rgba, "RGBA").convert(img.mode)


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

    Order: solid background → tiny-input upscale → trace-resolution supersample →
    denoise → flatten → quantize (median-cut or fixed-K k-means) → outline band →
    background removal. Solid-background runs first so the subject is detected from
    the original colors before any flattening/quantization.
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
        img = flatten_colors(img, opts.flatten_sigma, opts.flatten_spatial)
    # Supersample AFTER denoise/flatten: the bilateral filter is O(pixels) and must
    # run on the native grid (running it on the 2048 upscale is ~70x slower for no
    # benefit), then quantize on the upscaled image so vtracer traces a clean grid.
    if opts.trace_resolution:
        img = upscale_to(img, opts.trace_resolution)
    if opts.quantize:
        if opts.coverage_palette:
            img = quantize_coverage(
                img,
                opts.coverage_step,
                coverage=opts.coverage_fraction,
                min_area=opts.coverage_min_area,
                max_colors=opts.coverage_max_colors,
                region_cleanup=opts.coverage_region_cleanup,
                region_min_area=opts.coverage_region_min_area,
                region_max_px=opts.coverage_region_max_px,
            )
        elif opts.kmeans_palette:
            img = quantize_kmeans(
                img,
                opts.palette_k,
                opts.black_anchor_luma,
                detect_linework=opts.detect_linework,
                linework_radius=opts.linework_radius,
                linework_threshold=opts.linework_threshold,
            )
        else:
            img = quantize_colors(img, opts.palette_size)
    if opts.uniform_outline:
        img = uniform_outline(img, opts.outline_width)
    if opts.remove_background:
        img = remove_background(img, opts.bg_tolerance)

    return img.convert("RGBA")
