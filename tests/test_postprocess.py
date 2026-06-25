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


def _curve_command_count(svg: str) -> int:
    root = ET.fromstring(svg)
    return sum(
        el.get("d", "").count("C") + el.get("d", "").count("Q")
        for el in root.iter()
        if el.tag == f"{{{SVG_NS}}}path"
    )


def test_curves_survive_default_simplification():
    # The input is curve-based; postprocess must not polygonalize it into a
    # coarser, all-straight-line result at the conservative default.
    assert _curve_command_count(TRACE_SVG) > 0
    out = postprocess(TRACE_SVG)
    assert _curve_command_count(out) > 0


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


def test_output_is_responsive_with_viewbox():
    # Output must be a scalable, responsive root: a viewBox is present and there are
    # no fixed pixel width/height (which would make a browser render at raw size and
    # overflow/scroll). A viewBox is derived from the source dimensions when absent.
    out_root = ET.fromstring(postprocess(TRACE_SVG))
    assert out_root.get("viewBox") is not None
    assert out_root.get("width") is None
    assert out_root.get("height") is None
    assert "100%" in (out_root.get("style") or "")


def test_existing_viewbox_is_kept():
    svg = (
        f'<svg xmlns="{SVG_NS}" width="200" height="100" viewBox="0 0 20 10">'
        '<path d="M0 0 L20 0 L20 10 L0 10 Z" fill="#123456"/>'
        "</svg>"
    )
    out_root = ET.fromstring(postprocess(svg))
    assert out_root.get("viewBox") == "0 0 20 10"
    assert out_root.get("width") is None and out_root.get("height") is None


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


def test_palette_threshold_zero_merges_nothing():
    """palette_threshold <= 0 is an identity (no merge): near-identical fills both survive.
    The short-circuit also skips the O(n^2) pairwise ΔE scan (the coverage path uses 0)."""
    svg = (
        f'<svg xmlns="{SVG_NS}" width="10" height="10">'
        '<path d="M0 0 L1 0 L1 1 Z" fill="#101010"/>'
        '<path d="M2 2 L3 2 L3 3 Z" fill="#121212"/>'  # near-identical, but not merged at 0
        "</svg>"
    )
    out = postprocess(svg, PostprocessOptions(palette_threshold=0.0, simplify_level=0))
    assert len(_fill_colors(out)) == 2


def test_merge_fill_runs_collapses_consecutive_translate_paths():
    # Two consecutive same-fill translate-only paths collapse into ONE <path>
    # (subpaths combined, translate baked in); a different fill stays separate.
    svg = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10">'
        '<path fill="#ff0000" transform="translate(1,1)" d="M0 0 L2 0 L2 2 Z"/>'
        '<path fill="#ff0000" transform="translate(3,3)" d="M0 0 L1 0 L1 1 Z"/>'
        '<path fill="#0000ff" d="M5 5 L6 5 L6 6 Z"/>'
        "</svg>"
    )
    root = ET.fromstring(postprocess(svg))
    reds = [g for g in root if g.get("fill") == "#ff0000"]
    assert len(reds) == 1
    assert len(reds[0]) == 1  # one merged path, not two
    assert reds[0][0].get("d").count("M") == 2  # both shapes' subpaths kept
    assert reds[0][0].get("transform") is None  # translate baked into geometry


def test_merge_fill_runs_skips_scaled_paths():
    # Scale (Potrace/binary) transforms must NOT be baked — those paths stay
    # separate so the scale is preserved rather than silently dropped.
    svg = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10">'
        '<path fill="#ff0000" transform="scale(2)" d="M0 0 L1 0 L1 1 Z"/>'
        '<path fill="#ff0000" transform="scale(2)" d="M2 2 L3 2 L3 3 Z"/>'
        "</svg>"
    )
    reds = [g for g in ET.fromstring(postprocess(svg)) if g.get("fill") == "#ff0000"]
    assert len(reds[0]) == 2  # not merged
    assert all(p.get("transform") == "scale(2)" for p in reds[0])


def test_merge_fill_runs_can_be_disabled():
    svg = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10">'
        '<path fill="#ff0000" transform="translate(1,1)" d="M0 0 L2 0 L2 2 Z"/>'
        '<path fill="#ff0000" transform="translate(3,3)" d="M0 0 L1 0 L1 1 Z"/>'
        "</svg>"
    )
    out = postprocess(svg, PostprocessOptions(merge_fill_runs=False))
    reds = [g for g in ET.fromstring(out) if g.get("fill") == "#ff0000"]
    assert len(reds[0]) == 2  # left as separate paths


def _all_path_ds(svg: str) -> list[str]:
    return [p.get("d") for p in ET.fromstring(svg).iter(f"{{{SVG_NS}}}path")]


def test_snap_background_layer_rewrites_full_canvas_path_as_rect():
    from svgsmith.postprocess import snap_background_layer

    # A wobbly near-full-canvas background path (overshoots the canvas) under a
    # small foreground shape. The background becomes the exact canvas rectangle.
    svg = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">'
        '<path fill="#eeee00" d="M-2 -2 L101 0 L100 102 L1 100 Z"/>'
        '<path fill="#ff0000" d="M40 40 L60 40 L60 60 Z"/>'
        "</svg>"
    )
    ds = _all_path_ds(snap_background_layer(svg))
    assert ds[0] == "M0 0L100 0L100 100L0 100Z"  # bottom path snapped to the rect


def test_snap_background_layer_is_noop_without_a_full_canvas_background():
    from svgsmith.postprocess import snap_background_layer

    # Line-art / no dominant background: nothing covers the canvas, so no snap.
    svg = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">'
        '<path fill="#ff0000" d="M10 10 L20 10 L20 20 Z"/>'
        "</svg>"
    )
    ds = _all_path_ds(snap_background_layer(svg))
    assert all(not d.startswith("M0 0L100 0") for d in ds)


def test_snap_dark_fills_collapses_near_black_to_one_layer():
    """#lever-C output side: every fill within ΔE of pure black collapses to one #000000;
    non-dark fills and (de<=0) are untouched — so it is a no-op on art with no near-black."""
    from svgsmith.postprocess import snap_dark_fills

    svg = (
        f'<svg xmlns="{SVG_NS}" width="10" height="10">'
        '<path d="M0 0 L1 0 L1 1 Z" fill="#0a0a0a"/>'  # near-black tint
        '<path d="M2 2 L3 2 L3 3 Z" fill="#050505"/>'  # near-black tint
        '<path d="M5 5 L6 5 L6 6 Z" fill="#ff0000"/>'  # bright, must survive
        "</svg>"
    )
    out = snap_dark_fills(svg, de=12.0)
    fills = _fill_colors(out)
    assert "#000000" in fills
    assert "#ff0000" in fills
    assert "#0a0a0a" not in fills and "#050505" not in fills  # both darks merged
    # de<=0 disables the pass entirely (byte-identical)
    assert snap_dark_fills(svg, de=0.0) == svg


def test_global_same_fill_merge_collapses_fragmented_fills():
    """Phase 3 flat-economy: same-fill fragments hoist into one <path> per fill (fewer
    paths, same pixels); a non-repeating palette is returned unchanged (nothing to merge)."""
    from svgsmith.postprocess import global_same_fill_merge

    svg = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10">'
        '<path fill="#ff0000" d="M0 0 L1 0 L1 1 Z"/>'
        '<path fill="#ff0000" d="M2 2 L3 2 L3 3 Z"/>'  # same fill, fragmented
        '<path fill="#00ff00" d="M5 5 L6 5 L6 6 Z"/>'
        "</svg>"
    )
    out = global_same_fill_merge(svg)
    root = ET.fromstring(out)
    paths = [p for p in root.iter(f"{{{SVG_NS}}}path")]
    fills = [p.get("fill") for p in paths]
    assert len(paths) == 2  # the two reds collapsed into one, green stays
    assert fills.count("#ff0000") == 1 and "#00ff00" in fills
    red = next(p for p in paths if p.get("fill") == "#ff0000")
    assert red.get("d").count("M") == 2  # both red subpaths preserved in the merged path

    # All-unique palette: nothing to merge → returned unchanged.
    uniq = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10">'
        '<path fill="#ff0000" d="M0 0 L1 0 L1 1 Z"/>'
        '<path fill="#00ff00" d="M2 2 L3 2 L3 3 Z"/>'
        "</svg>"
    )
    assert global_same_fill_merge(uniq) == uniq
