"""Tests for the canonical Report schema and SVG statistics."""

import json

from svgsmith.report import Report, SvgStats, svg_stats

SVG_NS = "http://www.w3.org/2000/svg"

GROUPED_SVG = (
    f'<svg xmlns="{SVG_NS}" width="10" height="10">'
    '<g fill="#ff0000"><path d="M0 0 L1 0 L1 1 Z"/><path d="M2 2 L3 2 L3 3 Z"/></g>'
    '<g fill="#00ff00"><path d="M4 4 L5 4 L5 5 Z"/></g>'
    "</svg>"
)


def test_svg_stats_counts():
    stats = svg_stats(GROUPED_SVG)
    assert stats.paths == 3
    assert stats.groups == 2
    assert stats.colors == 2
    assert stats.bytes == len(GROUPED_SVG.encode("utf-8"))


def test_report_serializes_exact_contract_fields():
    report = Report(
        output="out.svg",
        mode_used="color",
        engine="vtracer",
        preset="illustration",
        iterations=2,
        similarity=0.93,
        passed_threshold=True,
        svg=SvgStats(paths=3, groups=2, colors=2, bytes=128),
        warnings=["photographic gradients; vectorization may bloat"],
    )
    data = json.loads(report.to_json())

    assert list(data.keys()) == [
        "output",
        "mode_used",
        "engine",
        "preset",
        "iterations",
        "similarity",
        "passed_threshold",
        "svg",
        "color_error",
        "warnings",
    ]
    assert list(data["svg"].keys()) == ["paths", "groups", "colors", "bytes"]
    assert data["warnings"] == ["photographic gradients; vectorization may bloat"]


def test_report_to_json_is_parseable():
    report = Report(
        output="a.svg",
        mode_used="binary",
        engine="potrace",
        preset="logo",
        iterations=1,
        similarity=0.99,
        passed_threshold=True,
        svg=SvgStats(paths=1, groups=1, colors=1, bytes=64),
    )
    assert json.loads(report.to_json())["engine"] == "potrace"
    assert report.warnings == []


def test_color_error_metric_catches_color_shift_ssim_misses():
    """#37: mean ΔE flags a pure color shift; identical images score ~0. (Reported channel —
    the loop's similarity/pass logic is unchanged.)"""
    import numpy as np
    from PIL import Image

    from svgsmith.verify import color_error

    base = np.zeros((120, 120, 3), np.uint8)
    base[:] = (200, 40, 40)  # red card
    red = Image.fromarray(base, "RGB")
    shifted = Image.fromarray(np.roll(base, 1, axis=2), "RGB")  # same structure, blue-ish

    assert color_error(red, red.copy()) < 0.5
    shift_de = color_error(red, shifted)
    assert shift_de > 20  # violent hue shift = large ΔE


def test_report_carries_color_error(tmp_path):
    from PIL import Image

    from svgsmith.pipeline import ConvertOptions, convert

    src = tmp_path / "flat.png"
    Image.new("RGB", (200, 200), (30, 120, 200)).save(src)
    _, report = convert(str(src), ConvertOptions(max_iters=1))
    assert report.color_error is not None and report.color_error < 6.0
    assert "color_error" in report.to_json()
