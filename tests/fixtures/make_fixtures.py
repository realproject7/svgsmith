"""Regenerate the engine smoke fixtures.

Run with ``python tests/fixtures/make_fixtures.py`` to deterministically rebuild
``logo.png`` (monochrome line art) and ``illustration.png`` (flat multi-color).
These two images are owned by T2 and reused by #4 and #7; do not mint duplicates
elsewhere.
"""

from __future__ import annotations

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


def main() -> None:
    make_logo().save(HERE / "logo.png")
    make_illustration().save(HERE / "illustration.png")


if __name__ == "__main__":
    main()
