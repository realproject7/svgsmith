"""Tests for SVG -> PNG rasterization (the `rasterize` subcommand)."""

from pathlib import Path

from PIL import Image

from svgsmith.render import _viewbox_size, rasterize

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20" '
    'style="width:100%;height:100%"><path d="M0 0 L40 0 L40 20 L0 20 Z" fill="#3366cc"/></svg>'
)


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "in.svg"
    p.write_text(_SVG, encoding="utf-8")
    return p


def test_viewbox_size_from_regex():
    assert _viewbox_size(_SVG) == (40, 20)


def test_rasterize_defaults_to_viewbox_size(tmp_path):
    out = rasterize(str(_write(tmp_path)), str(tmp_path / "out.png"))
    assert Image.open(out).size == (40, 20)


def test_rasterize_explicit_width(tmp_path):
    out = rasterize(str(_write(tmp_path)), str(tmp_path / "out.png"), width=200)
    assert Image.open(out).size[0] == 200


def test_rasterize_does_not_full_parse_untrusted_svg():
    # _viewbox_size must not choke on (or expand) entity/DTD content — it only
    # regex-scrapes the root tag, never a full XML parse.
    hostile = '<!DOCTYPE svg [<!ENTITY x "y">]>' + _SVG
    assert _viewbox_size(hostile) == (40, 20)
