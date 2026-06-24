"""Quality comparison harness for the svgsmith daily quality loop.

Given a folder laid out as::

    <folder>/inputs/     source images (.png/.jpg/.webp)
    <folder>/ref-svg/    reference-target SVGs, same basenames (optional)

run svgsmith on each input with the given illustration-geometry knobs, write the SVGs to
``<folder>/ours/``, build a per-image ``input | ours | reference target`` montage in
``<folder>/montage/``, compute metrics, and write ``<folder>/metrics.md``.

The reference is referred to generically as "reference target"; no product/brand names appear
here or in the output. This is the reusable tool the daily-loop ticket invokes; it is not part
of the published package (``tools/`` is outside ``src/``).

Usage::

    python tools/quality_compare.py <folder> [--supersample 2048] [--dark-thin 2]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

from svgsmith.pipeline import ConvertOptions, convert
from svgsmith.verify import rasterize

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_FILL = re.compile(r'fill\s*[:=]\s*["\']?(#[0-9a-fA-F]{3,6})')


def _palette(svg: str) -> int:
    return len({m.lower() for m in _FILL.findall(svg)})


def _to_rgb(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB").resize(size, Image.LANCZOS)


def _ssim(a: Image.Image, b: Image.Image) -> float:
    return float(ssim(np.asarray(a), np.asarray(b), channel_axis=2))


def _montage(cells: list[tuple[Image.Image, str]], width: int, path: Path) -> None:
    imgs = [im.resize((width, int(im.height * width / im.width))) for im, _ in cells]
    height = max(i.height for i in imgs)
    canvas = Image.new("RGB", (width * len(imgs) + 10 * (len(imgs) + 1), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, (_, label)) in enumerate(zip(imgs, cells, strict=True)):
        canvas.paste(im, (10 + i * (width + 10), 24))
        draw.text((10 + i * (width + 10) + 2, 7), label, fill="black")
    canvas.save(path)


def _write_metrics(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    lines = [
        f"# metrics (supersample={args.supersample}, dark_thin={args.dark_thin})",
        "",
        "| image | SSIM→ref | ours pal/paths/bytes | ref pal/paths/bytes |",
        "|---|---|---|---|",
    ]
    for r in rows:
        sr = f"{r['ssim_ref']:.3f}" if r["ssim_ref"] is not None else "-"
        ours = f"{r['pal']}/{r['paths']}/{r['bytes']}"
        ref = (
            f"{r['ref_pal']}/{r['ref_paths']}/{r['ref_bytes']}"
            if r["ref_pal"] is not None
            else "-"
        )
        lines.append(f"| {r['name']} | {sr} | {ours} | {ref} |")
    vals = [r["ssim_ref"] for r in rows if r["ssim_ref"] is not None]
    if vals:
        lines += ["", f"mean SSIM→ref: {sum(vals) / len(vals):.3f} over {len(vals)} images"]
    path.write_text("\n".join(lines) + "\n")


def run(folder: Path, args: argparse.Namespace) -> list[dict]:
    inputs, refs = folder / "inputs", folder / "ref-svg"
    out, mont = folder / "ours", folder / "montage"
    out.mkdir(parents=True, exist_ok=True)
    mont.mkdir(parents=True, exist_ok=True)
    opts = ConvertOptions(
        max_iters=1,
        illustration_supersample=args.supersample,
        illustration_dark_thin=args.dark_thin,
    )
    rows: list[dict] = []
    for src in sorted(inputs.iterdir()):
        if src.suffix.lower() not in IMG_EXT:
            continue
        name = src.stem
        svg, _ = convert(str(src), opts)
        (out / f"{name}.svg").write_text(svg)
        orig = Image.open(src).convert("RGB")
        size = orig.size
        ours_r = _to_rgb(rasterize(svg, size), size)
        row = dict(
            name=name, pal=_palette(svg), paths=svg.count("<path"), bytes=len(svg),
            ssim_ref=None, ref_pal=None, ref_paths=None, ref_bytes=None,
        )
        cells = [(_to_rgb(orig, size), "input"), (ours_r, "ours")]
        ref_path = refs / f"{name}.svg"
        if ref_path.exists():
            ref_svg = ref_path.read_text()
            ref_r = _to_rgb(rasterize(ref_svg, size), size)
            row.update(
                ssim_ref=_ssim(ours_r, ref_r), ref_pal=_palette(ref_svg),
                ref_paths=ref_svg.count("<path"), ref_bytes=len(ref_svg),
            )
            cells.append((ref_r, "reference target"))
        _montage(cells, args.width, mont / f"{name}.png")
        rows.append(row)
    _write_metrics(folder / "metrics.md", rows, args)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="svgsmith quality comparison harness")
    ap.add_argument("folder", type=Path, help="daily folder with inputs/ and ref-svg/")
    ap.add_argument("--supersample", type=int, default=0, help="illustration_supersample knob")
    ap.add_argument("--dark-thin", type=int, default=0, help="illustration_dark_thin knob")
    ap.add_argument("--width", type=int, default=320, help="montage cell width (px)")
    args = ap.parse_args()
    rows = run(args.folder, args)
    print(
        f"done: {len(rows)} images -> {args.folder / 'ours'}, "
        f"montages -> {args.folder / 'montage'}, metrics -> {args.folder / 'metrics.md'}"
    )


if __name__ == "__main__":
    main()
