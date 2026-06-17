"""Regenerate the engine + classifier fixtures.

Run with ``python tests/fixtures/make_fixtures.py`` to deterministically rebuild:
- ``logo.png`` — monochrome line art (engine smoke + classifier ``binary``/``logo``)
- ``illustration.png`` — flat multi-color (engine smoke + classifier ``color``)
- ``icon.png`` — tiny 2-color glyph (classifier ``binary``/``icon``)
- ``pixel.png`` — tiny low-palette pixel art (classifier ``pixel``)
- ``photo.png`` — smooth gradients (classifier ``color`` + photo warning)
- ``noisy.png`` — flat bg + shape + speckle noise (preprocess quantize/denoise)
- ``flat_bg.png`` — solid bg + centered shape (preprocess background removal)

``logo.png`` / ``illustration.png`` are owned by T2; ``pixel.png`` / ``photo.png``
were added by T3; ``noisy.png`` / ``flat_bg.png`` were added by T4. All live here
and are reused downstream — do not mint duplicates elsewhere.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZE = 128


def make_logo() -> Image.Image:
    """A black-on-white mark: a ring plus a triangle (monochrome line art)."""
    img = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 16, 112, 112), outline=0, width=12)
    draw.polygon([(64, 36), (92, 92), (36, 92)], fill=0)
    return img.convert("RGB")


def make_illustration() -> Image.Image:
    """Flat solid color regions, no gradients (good for the color tracer)."""
    img = Image.new("RGB", (SIZE, SIZE), (86, 180, 233))  # sky blue background
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 88, SIZE, SIZE), fill=(0, 158, 115))  # green ground
    draw.ellipse((72, 12, 116, 56), fill=(240, 228, 66))  # yellow sun
    draw.polygon([(40, 88), (16, 88), (28, 44)], fill=(213, 94, 0))  # orange peak
    draw.rectangle((52, 60, 80, 88), fill=(204, 121, 167))  # pink block
    return img


def make_icon() -> Image.Image:
    """A tiny 32x32 black-on-white glyph: monochrome line art at icon scale."""
    img = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((6, 6, 25, 25), outline=0, width=3)
    draw.line((6, 6, 25, 25), fill=0, width=3)
    return img.convert("RGB")


def make_pixel() -> Image.Image:
    """A 16x16 pixel-art mark with a tiny flat palette and hard pixel grid."""
    bg = (40, 40, 60)
    body = (240, 200, 40)
    eye = (30, 30, 30)
    mouth = (200, 60, 60)
    grid = [[bg] * 16 for _ in range(16)]
    for y in range(3, 13):
        for x in range(3, 13):
            grid[y][x] = body
    for ex, ey in ((6, 6), (9, 6)):
        grid[ey][ex] = eye
    for mx in range(6, 10):
        grid[10][mx] = mouth
    img = Image.new("RGB", (16, 16))
    img.putdata([grid[y][x] for y in range(16) for x in range(16)])
    return img


def make_photo() -> Image.Image:
    """Smooth two-axis gradient: many unique colors, soft edges (photo-like)."""
    img = Image.new("RGB", (SIZE, SIZE))
    cx, cy = SIZE / 2, SIZE / 2
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            r = int(40 + 215 * x / (SIZE - 1))
            g = int(40 + 215 * y / (SIZE - 1))
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            b = int(255 * max(0.0, 1.0 - dist / (SIZE / 2)))
            pixels.append((r, g, b))
    img.putdata(pixels)
    return img


def make_noisy() -> Image.Image:
    """Flat gray bg + a solid shape + seeded salt-and-pepper speckle noise."""
    img = Image.new("RGB", (SIZE, SIZE), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 88, 88), fill=(40, 40, 160))
    rng = random.Random(20240617)  # seeded → deterministic fixture
    for _ in range(320):
        x = rng.randrange(SIZE)
        y = rng.randrange(SIZE)
        color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        img.putpixel((x, y), color)
    return img


def make_flat_bg() -> Image.Image:
    """Solid white background with a centered opaque red disc."""
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 88, 88), fill=(220, 30, 30))
    return img


def main() -> None:
    make_logo().save(HERE / "logo.png")
    make_illustration().save(HERE / "illustration.png")
    make_icon().save(HERE / "icon.png")
    make_pixel().save(HERE / "pixel.png")
    make_photo().save(HERE / "photo.png")
    make_noisy().save(HERE / "noisy.png")
    make_flat_bg().save(HERE / "flat_bg.png")


if __name__ == "__main__":
    main()
