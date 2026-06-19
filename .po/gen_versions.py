import os, sys
import cairosvg
from PIL import Image
from svgsmith.pipeline import convert, ConvertOptions
SRC="/Users/cho/Projects/z-design/SVGSmith_Convert"; OUT=os.path.join(SRC,"output")
IMAGES={
 "doodle_cat":"runyu_minimal_doodle-style_cat_avatar_sunglasses_cat_coding_o_b0508f54-aafc-42aa-b2c0-2ad879b1a9e6_3.png",
 "shiba_mascot":"project7_A_cute_chubby_white_Shiba_Inu_mascot_named_SVGSmith__9fc0d444-ec20-4fd5-a11f-fb0be744bd4f_2.png",
 "snowman":"crystalkohri_cute_happy_snowman_crouching_slightly_while_roll_b1c95e84-f616-4448-a2ba-cda6c185acfb_3.png",
 "vector":"project7_vector_--profile_i538g88_--v_8.1_74c97b3d-1003-446f-84bb-1215c16a53c5_2.png",
 "lettering":"project7_SVGSmith_in_huge_bold_custom_lettering_--profile_i53_f22bbff1-f5d4-4a21-aeaa-b92200bf2c42_3.png",
 "watercolor_bee":"project7_flat_watercolor_bee_and_cicada_--profile_i538g88_--v_50599cc1-8d2f-4397-8a70-7044158ea680_2.png",
}
def gen(version, smooth):
    vdir=os.path.join(OUT, version); os.makedirs(vdir, exist_ok=True)
    print(f"\n[{version}] smooth={smooth}")
    for name,fn in IMAGES.items():
        out_svg=os.path.join(vdir, f"{name}.svg")
        _svg, rep = convert(os.path.join(SRC,fn), ConvertOptions(smooth=smooth, out=out_svg, max_iters=4))
        open(out_svg,"w").write(_svg)
        print(f"  {name:14} sim={rep.similarity:.3f} bytes={rep.svg.bytes:7} paths={rep.svg.paths}")
gen("v1.0", smooth=False)
gen("v1.1", smooth=True)
