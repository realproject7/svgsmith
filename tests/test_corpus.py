"""Golden quality gate: run the pipeline over the corpus and assert thresholds.

The corpus under ``tests/corpus/<category>/`` plus ``expected.json`` is the
headless replacement for a human eyeballing output. Each fixture must classify
into the expected mode, score at or above its similarity floor, and stay within
its path/byte budget. Floors come from observed-good runs with a small margin;
a regression that lowers pipeline quality drops a fixture below its floor and
fails the suite (see ``test_capping_colors_breaks_the_gate``).
"""

import json
from pathlib import Path

import pytest

import svgsmith.verify as verify_module
from svgsmith.pipeline import ConvertOptions, convert

CORPUS = Path(__file__).resolve().parent / "corpus"
EXPECTED = json.loads((CORPUS / "expected.json").read_text())
MAX_ITERS = 4


def _convert(rel_path: str):
    return convert(str(CORPUS / rel_path), ConvertOptions(max_iters=MAX_ITERS))


@pytest.mark.parametrize("rel_path", sorted(EXPECTED))
def test_corpus_fixture_meets_golden_thresholds(rel_path):
    expected = EXPECTED[rel_path]
    _svg, report = _convert(rel_path)

    assert report.mode_used == expected["mode"], (
        f"{rel_path}: mode {report.mode_used} != expected {expected['mode']}"
    )
    assert report.similarity >= expected["min_similarity"], (
        f"{rel_path}: similarity {report.similarity:.4f} below floor "
        f"{expected['min_similarity']}"
    )
    assert report.svg.paths <= expected["max_paths"], (
        f"{rel_path}: paths {report.svg.paths} over budget {expected['max_paths']}"
    )
    assert report.svg.bytes <= expected["max_bytes"], (
        f"{rel_path}: bytes {report.svg.bytes} over budget {expected['max_bytes']}"
    )


def test_every_corpus_image_is_listed_in_expected():
    on_disk = {
        f"{png.parent.name}/{png.name}"
        for png in CORPUS.rglob("*.png")
    }
    assert on_disk == set(EXPECTED), "corpus images and expected.json are out of sync"


def test_capping_colors_breaks_the_gate(monkeypatch):
    """Proves the suite is a real gate: degrade quality and a floor must fail."""
    from svgsmith.engines.base import Preset

    # Force every iteration to a single, color-starved preset (caps quality).
    def _starved(base: Preset, _color_level: int) -> Preset:
        from dataclasses import replace

        return replace(base, color_precision=1, filter_speckle=12)

    monkeypatch.setattr(verify_module, "_tune_preset", _starved)

    failures = 0
    for rel_path, expected in EXPECTED.items():
        _svg, report = _convert(rel_path)
        if report.similarity < expected["min_similarity"]:
            failures += 1
    assert failures > 0, "capping colors should drop at least one fixture below its floor"
