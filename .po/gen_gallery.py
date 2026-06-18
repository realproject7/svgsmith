import subprocess, sys, os
from PIL import Image
import cairosvg

V = ".po/venv/bin/svgsmith"
items = [
    ("logo",  "tests/corpus/logo/logo_ring.png",                 ["--mode","binary"]),
    ("icon",  "tests/corpus/icon/icon_arrow.png",                ["--mode","binary"]),
    ("illustration", "tests/corpus/illustration/illustration_scene.png", ["--mode","color"]),
    ("pixel", "tests/corpus/pixel/pixel_heart.png",              ["--mode","pixel"]),
]
DISP = 256  # display square
os.makedirs("docs/gallery", exist_ok=True)
for name, src, flags in items:
    out_svg = f".po/out/gal_{name}.svg"
    subprocess.run([V,"convert",src,*flags,"--out",out_svg,"--report","json"], check=False)
    # input -> upscaled PNG (nearest to keep crisp pixels)
    im = Image.open(src).convert("RGBA")
    im = im.resize((DISP,DISP), Image.NEAREST)
    bg = Image.new("RGBA",(DISP,DISP),(255,255,255,255)); bg.alpha_composite(im)
    bg.convert("RGB").save(f"docs/gallery/{name}_in.png")
    # output svg -> PNG at same size
    cairosvg.svg2png(url=out_svg, write_to=f"docs/gallery/{name}_out.png",
                     output_width=DISP, output_height=DISP, background_color="white")
    print(f"{name}: {src} -> {out_svg} -> gallery PNGs")
