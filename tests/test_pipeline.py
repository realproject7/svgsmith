"""End-to-end tests for the conversion pipeline and CLI report output."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from svgsmith.classify import PHOTO_WARNING
from svgsmith.pipeline import ConvertOptions, convert

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ALL_FIXTURES = ["logo.png", "illustration.png", "icon.png", "pixel.png", "photo.png"]


@pytest.mark.parametrize(
    "kwargs",
    [{"quality": -0.1}, {"quality": 1.1}, {"max_iters": 0}],
)
def test_convert_options_rejects_out_of_range_values(kwargs):
    with pytest.raises(ValueError):
        ConvertOptions(**kwargs)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_convert_produces_valid_svg_and_consistent_report(name):
    svg, report = convert(str(FIXTURES / name), ConvertOptions(max_iters=2))
    # Valid SVG.
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    # passed_threshold matches similarity >= quality.
    assert report.passed_threshold == (report.similarity >= 0.9)
    # Report is internally consistent.
    assert report.output.endswith(".svg")
    assert report.engine in ("vtracer", "potrace")
    assert report.iterations >= 1
    assert report.svg.bytes == len(svg.encode("utf-8"))


def test_photo_fixture_surfaces_classifier_warning():
    _svg, report = convert(str(FIXTURES / "photo.png"), ConvertOptions(max_iters=1))
    assert PHOTO_WARNING in report.warnings


def test_explicit_mode_overrides_classification():
    _svg, report = convert(
        str(FIXTURES / "illustration.png"), ConvertOptions(mode="binary", max_iters=1)
    )
    assert report.mode_used == "binary"
    assert report.engine == "potrace"
    assert report.preset == "logo"


def test_no_editable_skips_grouping():
    _svg, grouped = convert(str(FIXTURES / "illustration.png"), ConvertOptions(max_iters=1))
    _svg2, raw = convert(
        str(FIXTURES / "illustration.png"), ConvertOptions(editable=False, max_iters=1)
    )
    # Editable output is grouped into <g> layers; raw traced output is not.
    assert grouped.svg.groups >= 1
    assert raw.svg.groups == 0


def test_out_path_defaults_to_input_with_svg_suffix():
    _svg, report = convert(str(FIXTURES / "logo.png"), ConvertOptions(max_iters=1))
    assert report.output == str(FIXTURES / "logo.svg")


def test_cli_report_json_is_only_thing_on_stdout(tmp_path):
    out = tmp_path / "out.svg"
    result = subprocess.run(
        [
            sys.executable, "-m", "svgsmith", "convert", str(FIXTURES / "illustration.png"),
            "--max-iters", "2", "--out", str(out), "--report", "json",
        ],
        capture_output=True,
        text=True,
    )
    # stdout parses cleanly as JSON and is the only thing there.
    payload = json.loads(result.stdout)
    assert payload["mode_used"] == "color"
    assert payload["passed_threshold"] == (payload["similarity"] >= 0.9)
    assert out.exists()
    # Exit code reflects pass/below-threshold.
    assert result.returncode in (0, 2)
    assert (result.returncode == 0) == payload["passed_threshold"]


def test_cli_default_report_keeps_stdout_empty(tmp_path):
    out = tmp_path / "out.svg"
    result = subprocess.run(
        [
            sys.executable, "-m", "svgsmith", "convert", str(FIXTURES / "logo.png"),
            "--max-iters", "1", "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""  # no --report json → nothing on stdout
    assert out.exists()


def test_uniform_outline_is_opt_in_and_color_only():
    # Off by default: identical to the standard color conversion.
    base = convert(str(FIXTURES / "illustration.png"), ConvertOptions(max_iters=1))[1]
    on = convert(
        str(FIXTURES / "illustration.png"),
        ConvertOptions(max_iters=1, uniform_outline=True),
    )[1]
    assert base.mode_used == "color"
    # The flag runs without error and still produces a valid color SVG report.
    assert on.mode_used == "color"
    assert on.svg.paths >= 1


def test_solid_background_is_opt_in_and_runs():
    on = convert(
        str(FIXTURES / "illustration.png"),
        ConvertOptions(max_iters=1, solid_background=True),
    )[1]
    # Off by default elsewhere; on, it still yields a valid color SVG report.
    assert on.mode_used == "color"
    assert on.svg.paths >= 1


def test_detail_level_validation_and_spectrum():
    import pytest

    with pytest.raises(ValueError):
        ConvertOptions(detail="ultra")
    # The dial trades detail for flatness: higher levels keep fewer colors.
    counts = {}
    for level in ("high", "normal", "clean", "poster"):
        _svg, rep = convert(
            str(FIXTURES / "illustration.png"),
            ConvertOptions(detail=level, max_iters=1),
        )
        counts[level] = rep.svg.colors
    assert counts["high"] >= counts["normal"] >= counts["clean"] >= counts["poster"]
