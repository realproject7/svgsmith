"""Tests for the self-verify loop (rasterize → SSIM → re-tune)."""

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from svgsmith.classify import classify
from svgsmith.verify import VerifyResult, rasterize, run_loop, score

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ILLUSTRATION = FIXTURES / "illustration.png"
LOGO = FIXTURES / "logo.png"

SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">'
    '<rect width="32" height="32" fill="#ffffff"/>'
    '<rect x="8" y="8" width="16" height="16" fill="#cc0000"/>'
    "</svg>"
)


def test_rasterize_returns_rgb_of_requested_size():
    img = rasterize(SIMPLE_SVG, (40, 24))
    assert img.mode == "RGB"
    assert img.size == (40, 24)


def test_score_identical_is_one_and_different_is_less():
    original = Image.open(ILLUSTRATION).convert("RGB")
    assert score(original, original) >= 0.999
    other = Image.new("RGB", original.size, (0, 0, 0))
    assert score(original, other) < score(original, original)


def test_first_pass_full_fidelity_converges():
    cls = classify(ILLUSTRATION)
    _svg, result = run_loop(ILLUSTRATION, cls, quality=0.8, max_iters=4)
    # The first pass now traces at full color fidelity (no color-starved ramp),
    # so a reasonable target is met and the best result clears it.
    assert result.best_score >= 0.8


def test_loop_engages_and_keeps_best_when_target_unreachable():
    cls = classify(ILLUSTRATION)
    # An unreachable target forces the loop to spend its whole budget, varying
    # parameters each pass; it must keep the strictly best-scoring result.
    _svg, result = run_loop(ILLUSTRATION, cls, quality=0.999, max_iters=4)
    assert result.iterations > 1
    assert result.best_score == max(result.scores)


def test_result_reports_scores_params_and_iterations():
    cls = classify(ILLUSTRATION)
    _svg, result = run_loop(ILLUSTRATION, cls, quality=0.8, max_iters=4)
    assert isinstance(result, VerifyResult)
    assert len(result.scores) == result.iterations
    assert result.best_score in result.scores
    for key in ("mode", "preset", "color_precision", "filter_speckle", "simplify_level"):
        assert key in result.params


def test_returns_best_scoring_result():
    cls = classify(ILLUSTRATION)
    _svg, result = run_loop(ILLUSTRATION, cls, quality=0.8, max_iters=4)
    # The returned result must be the highest-scoring iteration, not merely the
    # last one that stayed above target.
    assert result.best_score == max(result.scores)


def test_default_renderer_is_cairosvg_even_if_resvg_present(monkeypatch):
    import svgsmith.verify as verify_module

    # Pretend resvg is installed; the default path must still NOT call it.
    monkeypatch.setattr(verify_module.shutil, "which", lambda _name: "/usr/bin/resvg")

    def _fail(*_args, **_kwargs):
        raise AssertionError("resvg should not be used by the default renderer")

    monkeypatch.setattr(verify_module, "_rasterize_resvg", _fail)
    img = rasterize(SIMPLE_SVG, (32, 32))  # renderer=None default
    assert img.size == (32, 32)


def test_iteration_count_never_exceeds_max_iters():
    cls = classify(ILLUSTRATION)
    for max_iters in (1, 2, 3):
        _svg, result = run_loop(ILLUSTRATION, cls, quality=0.999, max_iters=max_iters)
        assert result.iterations <= max_iters


def test_cost_discipline_first_pass_above_threshold_stops_early():
    cls = classify(ILLUSTRATION)
    # A low target is met on the first pass, so the loop should not iterate.
    _svg, result = run_loop(ILLUSTRATION, cls, quality=0.5, max_iters=4)
    assert result.iterations == 1


def test_run_loop_returns_valid_svg():
    cls = classify(ILLUSTRATION)
    svg, _result = run_loop(ILLUSTRATION, cls, quality=0.8, max_iters=3)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
