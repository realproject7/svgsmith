"""Tests for the engine adapters (VTracer color + Potrace binary)."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from svgsmith.engines import (
    PRESETS,
    BinaryTracer,
    ColorTracer,
    Preset,
    Tracer,
    get_preset,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOGO = FIXTURES / "logo.png"
ILLUSTRATION = FIXTURES / "illustration.png"

_FILL_RE = re.compile(r"fill\s*:?=?\s*[\"']?\s*(#[0-9a-fA-F]{3,6})")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_svg(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert _local(root.tag) == "svg", f"root is <{_local(root.tag)}>, expected <svg>"
    return root


def _fill_colors(svg: str) -> set[str]:
    return {m.group(1).lower() for m in _FILL_RE.finditer(svg)}


def _path_count(root: ET.Element) -> int:
    return sum(1 for el in root.iter() if _local(el.tag) == "path")


def test_canonical_presets_present():
    assert set(PRESETS) == {"logo", "icon", "illustration", "pixel"}
    for name, preset in PRESETS.items():
        assert isinstance(preset, Preset)
        assert preset.name == name


def test_get_preset_roundtrip_and_error():
    assert get_preset("logo").name == "logo"
    with pytest.raises(ValueError):
        get_preset("nope")


def test_adapters_satisfy_tracer_protocol():
    assert isinstance(ColorTracer(), Tracer)
    assert isinstance(BinaryTracer(), Tracer)


def test_color_tracer_returns_multicolor_svg():
    svg = ColorTracer().trace(ILLUSTRATION, get_preset("illustration"))
    root = _parse_svg(svg)
    assert _path_count(root) >= 1
    # The flat-color illustration has several distinct regions.
    assert len(_fill_colors(svg)) >= 2


def test_binary_tracer_returns_monochrome_svg():
    svg = BinaryTracer().trace(LOGO, get_preset("logo"))
    root = _parse_svg(svg)
    assert _path_count(root) >= 1
    # Potrace emits a single ink color (monochrome).
    assert len(_fill_colors(svg)) <= 1


def test_trace_accepts_path_string():
    svg = ColorTracer().trace(str(ILLUSTRATION), get_preset("illustration"))
    _parse_svg(svg)
