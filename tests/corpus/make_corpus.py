"""Generate the quality-gate corpus (programmatically, no external images).

Run with ``python tests/corpus/make_corpus.py`` to (re)build the corpus under
``tests/corpus/<category>/*.png``. Every image is drawn here, so there are no
image-license questions and the set is fully deterministic.

Categories and the engine mode each is expected to classify into:
    logo, icon       → binary   (monochrome line art)
    illustration     → color    (flat multi-color regions)
    photo            → color     (smooth gradients; carries the photo warning)
    pixel            → pixel     (tiny, low-palette, hard grid)

This ticket (T8) owns ``tests/corpus/``; it is the single corpus source.
``tests/fixtures/`` (owned by T2/T3/T4) is separate and not used here.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# logo — monochrome line art, large canvas (→ binary / logo)
# --------------------------------------------------------------------------- #


def logo_ring() -> Image.Image:
    img = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(img)
    draw.ellipse((18, 18, 110, 110), outline=0, width=12)
    draw.polygon([(64, 40), (90, 88), (38, 88)], fill=0)
    return img.convert("RGB")


def logo_bars() -> Image.Image:
    img = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(img)
    for i, x in enumerate(range(20, 110, 22)):
        draw.rectangle((x, 24 + i * 6, x + 12, 104), fill=0)
    draw.rectangle((16, 104, 112, 112), fill=0)
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# icon — small monochrome glyph (→ binary / icon)
# --------------------------------------------------------------------------- #


def icon_cross() -> Image.Image:
    img = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((6, 6, 25, 25), outline=0, width=3)
    draw.line((6, 6, 25, 25), fill=0, width=3)
    return img.convert("RGB")


def icon_arrow() -> Image.Image:
    img = Image.new("L", (40, 40), 255)
    draw = ImageDraw.Draw(img)
    draw.line((8, 20, 30, 20), fill=0, width=4)
    draw.polygon([(26, 12), (34, 20), (26, 28)], fill=0)
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# illustration — flat multi-color regions (→ color)
# --------------------------------------------------------------------------- #


def illustration_scene() -> Image.Image:
    img = Image.new("RGB", (128, 128), (86, 180, 233))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 88, 128, 128), fill=(0, 158, 115))
    draw.ellipse((72, 12, 116, 56), fill=(240, 228, 66))
    draw.polygon([(40, 88), (16, 88), (28, 44)], fill=(213, 94, 0))
    draw.rectangle((52, 60, 80, 88), fill=(204, 121, 167))
    return img


def illustration_blocks() -> Image.Image:
    img = Image.new("RGB", (128, 128), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    palette = [(231, 76, 60), (46, 134, 193), (241, 196, 15), (39, 174, 96)]
    for i, color in enumerate(palette):
        x = 8 + (i % 2) * 60
        y = 8 + (i // 2) * 60
        draw.rectangle((x, y, x + 52, y + 52), fill=color)
    return img


# --------------------------------------------------------------------------- #
# photo — smooth gradients, many colors (→ color + photo warning)
# --------------------------------------------------------------------------- #


def photo_radial() -> Image.Image:
    size = 128
    img = Image.new("RGB", (size, size))
    cx, cy = size / 2, size / 2
    pixels = []
    for y in range(size):
        for x in range(size):
            r = int(40 + 215 * x / (size - 1))
            g = int(40 + 215 * y / (size - 1))
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            b = int(255 * max(0.0, 1.0 - dist / (size / 2)))
            pixels.append((r, g, b))
    img.putdata(pixels)
    return img


def photo_linear() -> Image.Image:
    size = 128
    img = Image.new("RGB", (size, size))
    pixels = []
    for y in range(size):
        for x in range(size):
            r = int(20 + 235 * x / (size - 1))
            g = int(120 + 100 * y / (size - 1))
            b = int(200 - 150 * x / (size - 1))
            pixels.append((r, g, max(0, min(255, b))))
    img.putdata(pixels)
    return img


# --------------------------------------------------------------------------- #
# pixel — tiny, low palette, hard grid (→ pixel)
# --------------------------------------------------------------------------- #


def _from_grid(grid: list[list[tuple[int, int, int]]]) -> Image.Image:
    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGB", (w, h))
    img.putdata([grid[y][x] for y in range(h) for x in range(w)])
    return img


def pixel_face() -> Image.Image:
    # Light background: svgsmith removes the background to transparency and the
    # verify loop scores over a white matte, so a light ground compares cleanly.
    bg = (245, 245, 245)
    body = (240, 200, 40)
    eye = (30, 30, 30)
    mouth = (200, 60, 60)
    grid = [[bg] * 16 for _ in range(16)]
    for y in range(3, 13):
        for x in range(3, 13):
            grid[y][x] = body
    grid[6][6] = grid[6][9] = eye
    for x in range(6, 10):
        grid[10][x] = mouth
    return _from_grid(grid)


def pixel_heart() -> Image.Image:
    # Four colors keep this in the pixel branch (palette > 3 avoids `binary`);
    # light background for clean scoring over the white matte.
    bg = (245, 245, 245)
    red = (220, 40, 60)
    hi = (255, 150, 160)
    edge = (90, 0, 20)
    rows = [
        "..ee..ee..",
        ".errreerr.",
        "erhhrrrrre",
        "erhhrrrrre",
        "errrrrrrre",
        ".errrrrre.",
        "..errrre..",
        "...erre...",
        "....ee....",
        "..........",
    ]
    palette = {".": bg, "r": red, "h": hi, "e": edge}
    grid = [[palette[c] for c in row] for row in rows]
    return _from_grid(grid)


CORPUS = {
    "logo": {"logo_ring.png": logo_ring, "logo_bars.png": logo_bars},
    "icon": {"icon_cross.png": icon_cross, "icon_arrow.png": icon_arrow},
    "illustration": {
        "illustration_scene.png": illustration_scene,
        "illustration_blocks.png": illustration_blocks,
    },
    "photo": {"photo_radial.png": photo_radial, "photo_linear.png": photo_linear},
    "pixel": {"pixel_face.png": pixel_face, "pixel_heart.png": pixel_heart},
}


def main() -> None:
    for category, fixtures in CORPUS.items():
        out_dir = HERE / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename, builder in fixtures.items():
            builder().save(out_dir / filename)


if __name__ == "__main__":
    main()
