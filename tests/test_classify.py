"""Tests for the deterministic input classifier (``--mode auto``)."""

from pathlib import Path

import pytest

from svgsmith.classify import PHOTO_WARNING, Classification, classify

FIXTURES = Path(__file__).resolve().parent / "fixtures"

CASES = [
    ("logo.png", "binary", "logo"),
    ("illustration.png", "color", "illustration"),
    ("pixel.png", "pixel", "pixel"),
    ("photo.png", "color", "illustration"),
]


@pytest.mark.parametrize(("filename", "mode", "preset"), CASES)
def test_classify_fixture_modes(filename, mode, preset):
    result = classify(FIXTURES / filename)
    assert isinstance(result, Classification)
    assert result.mode == mode
    assert result.preset == preset


def test_photo_emits_warning_others_do_not():
    assert PHOTO_WARNING in classify(FIXTURES / "photo.png").warnings
    assert classify(FIXTURES / "logo.png").warnings == ()
    assert classify(FIXTURES / "illustration.png").warnings == ()
    assert classify(FIXTURES / "pixel.png").warnings == ()


@pytest.mark.parametrize("filename", [c[0] for c in CASES])
def test_classify_is_deterministic(filename):
    first = classify(FIXTURES / filename)
    second = classify(FIXTURES / filename)
    assert first == second


def test_classification_unpacks_as_tuple():
    mode, preset, warnings = classify(FIXTURES / "photo.png")
    assert (mode, preset) == ("color", "illustration")
    assert PHOTO_WARNING in warnings
