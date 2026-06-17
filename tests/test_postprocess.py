"""Tests for postprocessing tracer output into editable SVG."""

import xml.etree.ElementTree as ET
from pathlib import Path

from svgsmith.postprocess import (
    PostprocessOptions,
    count_path_points,
    postprocess,
    svg_bbox,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRACE_SVG = (FIXTURES / "trace_illustration.svg").read_text()

SVG_NS = "http://www.w3.org/2000/svg"


def _groups(svg: str) -> list[ET.Element]:
    root = ET.fromstring(svg)
    return [el for el in root.iter() if el.tag == f"{{{SVG_NS}}}g"]


def _fill_colors(svg: str) -> set[str]:
    root = ET.fromstring(svg)
    colors = set()
    for el in root.iter():
        fill = el.get("fill")
        if fill and fill != "none":
            colors.add(fill.lower())
    return colors


def test_output_is_valid_svg_with_groups():
    out = postprocess(TRACE_SVG)
    root = ET.fromstring(out)  # raises if invalid
    assert root.tag == f"{{{SVG_NS}}}svg"
    groups = _groups(out)
    # At least one <g> layer per distinct region/color.
    assert len(groups) >= 1
    assert len(groups) >= len(_fill_colors(out))
    # Every group carries a stable, readable id.
    assert all(g.get("id", "").startswith("layer-") for g in groups)


def test_point_count_strictly_reduced_at_default():
    out = postprocess(TRACE_SVG)
    assert count_path_points(out) < count_path_points(TRACE_SVG)


def test_color_count_not_increased():
    out = postprocess(TRACE_SVG)
    assert len(_fill_colors(out)) <= len(_fill_colors(TRACE_SVG))


def test_bounding_box_preserved_within_tolerance():
    out = postprocess(TRACE_SVG)
    before = svg_bbox(TRACE_SVG)
    after = svg_bbox(out)
    assert before is not None and after is not None
    width = before[2] - before[0]
    height = before[3] - before[1]
    tolerance = 0.03 * max(width, height)  # conservative default stays well under
    assert all(abs(a - b) <= tolerance for a, b in zip(before, after, strict=True))


def test_viewbox_or_dimensions_preserved():
    out = postprocess(TRACE_SVG)
    src_root = ET.fromstring(TRACE_SVG)
    out_root = ET.fromstring(out)
    assert out_root.get("width") == src_root.get("width")
    assert out_root.get("height") == src_root.get("height")


def test_simplify_level_zero_keeps_points():
    out = postprocess(TRACE_SVG, PostprocessOptions(simplify_level=0))
    assert count_path_points(out) == count_path_points(TRACE_SVG)


def test_higher_level_simplifies_at_least_as_much():
    conservative = count_path_points(postprocess(TRACE_SVG, PostprocessOptions(simplify_level=1)))
    aggressive = count_path_points(postprocess(TRACE_SVG, PostprocessOptions(simplify_level=6)))
    assert aggressive <= conservative


def test_palette_consolidation_merges_near_identical_fills():
    svg = (
        f'<svg xmlns="{SVG_NS}" width="10" height="10">'
        '<path d="M0 0 L1 0 L1 1 Z" fill="#101010"/>'
        '<path d="M2 2 L3 2 L3 3 Z" fill="#121212"/>'
        '<path d="M5 5 L6 5 L6 6 Z" fill="#a0c0e0"/>'
        "</svg>"
    )
    assert len(_fill_colors(svg)) == 3
    out = postprocess(svg, PostprocessOptions(palette_threshold=12.0))
    # The two near-identical dark fills collapse into one.
    assert len(_fill_colors(out)) == 2


def test_consolidation_can_be_disabled():
    svg = (
        f'<svg xmlns="{SVG_NS}" width="10" height="10">'
        '<path d="M0 0 L1 0 L1 1 Z" fill="#101010"/>'
        '<path d="M2 2 L3 2 L3 3 Z" fill="#121212"/>'
        "</svg>"
    )
    out = postprocess(svg, PostprocessOptions(consolidate_palette=False, simplify_level=0))
    assert len(_fill_colors(out)) == 2
