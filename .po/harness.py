#!/usr/bin/env python3
"""Repeatable svgsmith quality harness over the real test images.
Usage: .po/venv/bin/python .po/harness.py [extra svgsmith convert args...]
Outputs: metrics table (stdout) + hi-res side-by-side per image + a grid.
"""
import os, sys, json, subprocess
from PIL import Image, ImageDraw
import cairosvg

SRC = "/Users/cho/Projects/z-design/SVGSmith_Convert"
OUT = os.path.join(SRC, "output")
SVG = ".po/venv/bin/svgsmith"
os.makedirs(OUT, exist_ok=True)

IMAGES = {
 "doodle_cat":     "runyu_minimal_doodle-style_cat_avatar_sunglasses_cat_coding_o_b0508f54-aafc-42aa-b2c0-2ad879b1a9e6_3.png",
 "shiba_mascot":   "project7_A_cute_chubby_white_Shiba_Inu_mascot_named_SVGSmith__9fc0d444-ec20-4fd5-a11f-fb0be744bd4f_2.png",
 "vector":         "project7_vector_--profile_i538g88_--v_8.1_74c97b3d-1003-446f-84bb-1215c16a53c5_2.png",
 "snowman":        "crystalkohri_cute_happy_snowman_crouching_slightly_while_roll_b1c95e84-f616-4448-a2ba-cda6c185acfb_3.png",
 "watercolor_bee": "project7_flat_watercolor_bee_and_cicada_--profile_i538g88_--v_50599cc1-8d2f-4397-8a70-7044158ea680_2.png",
 "lettering":      "project7_SVGSmith_in_huge_bold_custom_lettering_--profile_i53_f22bbff1-f5d4-4a21-aeaa-b92200bf2c42_3.png",
}

def run(extra):
    rows = []
    S = 640
    for name, fn in IMAGES.items():
        src = os.path.join(SRC, fn)
        out_svg = os.path.join(OUT, f"{name}.svg")
        cmd = [SVG, "convert", src, "--mode", "auto", "--out", out_svg, "--report", "json", *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            d = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            rows.append((name, "ERR", "", "", "", "", "", ""));
            print(f"[{name}] ERROR: {p.stdout[-200:]} {p.stderr[-200:]}");
            continue
        s = d["svg"]
        rows.append((name, d["mode_used"], d["engine"], d["iterations"],
                     f"{d['similarity']:.3f}", d["passed_threshold"], s["paths"], s["colors"], s["bytes"]))
        # hi-res side-by-side
        inp = Image.open(src).convert("RGB"); inp.thumbnail((S, S))
        rpng = os.path.join(OUT, f"{name}_render.png")
        cairosvg.svg2png(url=out_svg, write_to=rpng, output_width=S, output_height=S, background_color="white")
        out = Image.open(rpng).convert("RGB")
        pad, lab = 16, 26
        W = inp.width + out.width + pad*3
        H = max(inp.height, out.height) + pad*2 + lab
        c = Image.new("RGB", (W, H), (245,245,245)); d2 = ImageDraw.Draw(c)
        c.paste(inp, (pad, pad+lab)); c.paste(out, (pad*2+inp.width, pad+lab))
        d2.text((pad, 4), "ORIGINAL", fill=(60,60,60))
        d2.text((pad*2+inp.width, 4), f"svgsmith  sim={rows[-1][4]} colors={s['colors']} paths={s['paths']}", fill=(60,60,60))
        c.save(os.path.join(OUT, f"cmp_{name}.png"))
    # table
    hdr = ("image","mode","engine","iters","sim","pass","paths","colors","bytes")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"]*len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")

if __name__ == "__main__":
    run(sys.argv[1:])
