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


def test_flatten_shading_reduces_facets_on_gradients(tmp_path):
    """--flatten-shading collapses smooth shading, so a noisy gradient traces into
    fewer paths than the default."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    ramp = np.add.outer(np.linspace(0, 200, 96), np.linspace(0, 40, 96))
    base = np.clip(ramp[..., None] + rng.normal(0, 6, (96, 96, 3)), 0, 255)
    Image.fromarray(base.astype(np.uint8), "RGB").save(tmp_path / "gradient.png")
    src = str(tmp_path / "gradient.png")

    plain = convert(src, ConvertOptions(mode="color", max_iters=2))[0]
    flat = convert(src, ConvertOptions(mode="color", max_iters=2, flatten_shading=True))[0]
    assert flat.count("<path") < plain.count("<path")


def test_transparent_background_removes_bg_keeps_subject(tmp_path):
    """--transparent-background drops the edge-connected background, leaving the
    subject on transparency (region-based, so subject detail is preserved)."""
    import numpy as np
    from PIL import Image, ImageDraw

    from svgsmith.render import rasterize

    # A blue subject on a uniform red background.
    image = Image.new("RGB", (96, 96), (230, 80, 80))
    ImageDraw.Draw(image).ellipse([28, 28, 68, 68], fill=(40, 60, 200))
    src = tmp_path / "subject.png"
    image.save(src)

    opts = ConvertOptions(mode="color", transparent_background=True, max_iters=2)
    svg, _ = convert(str(src), opts)
    svg_path = tmp_path / "out.svg"
    svg_path.write_text(svg, encoding="utf-8")
    png_path = tmp_path / "out.png"
    rasterize(str(svg_path), str(png_path), width=96)  # no background → transparent

    alpha = np.asarray(Image.open(png_path).convert("RGBA"))[:, :, 3]
    assert alpha[3, 3] == 0  # corner background removed
    assert alpha[-4, -4] == 0
    assert alpha[48, 48] > 0  # centered subject kept


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


def test_background_color_repaints_to_exact_color():
    import numpy as np

    from svgsmith.preprocess import PreprocessOptions, preprocess

    # The CLI/pipeline threads --background "#FFFFFF" into preprocess as an exact
    # target color: every edge-connected background pixel becomes pure white while
    # the subject is preserved. Assert on the preprocessed raster (the trace is
    # lossy) so the exact-color contract is checked deterministically.
    prepared = preprocess(
        str(FIXTURES / "illustration.png"),
        PreprocessOptions(solid_background=True, background_color="#FFFFFF"),
    )
    arr = np.asarray(prepared.convert("RGB"))
    assert tuple(arr[0, 0]) == (255, 255, 255)
    assert tuple(arr[0, -1]) == (255, 255, 255)
    # The subject (image center) is not flattened to white.
    assert tuple(arr[arr.shape[0] // 2, arr.shape[1] // 2]) != (255, 255, 255)
    # The whole pipeline accepts the flag and still produces a valid SVG.
    report = convert(
        str(FIXTURES / "illustration.png"),
        ConvertOptions(max_iters=1, background="#FFFFFF"),
    )[1]
    assert report.svg.paths >= 1


def test_background_invalid_color_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        ConvertOptions(background="not-a-color")


def _gradient_with_black_bars(size: int = 300):
    """Low-res flat cartoon stand-in: a many-color gradient (so there is something
    to consolidate) crossed by bold black bars (hard outline edges)."""
    import numpy as np
    from PIL import Image, ImageDraw

    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    arr = np.stack([grad, np.full((size, size), 120, np.uint8), 255 - grad], axis=2)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    for i in range(4):
        draw.rectangle([20 + i * 70, 40, 50 + i * 70, size - 40], fill=(0, 0, 0))
    return img


def test_supersample_gate_targets_low_res_flat_cartoons_only():
    """The supersample + k-means path triggers only for low-res, raster-degraded
    flat cartoon art — never large, synthetic-flat, gradient, or hatched inputs."""
    import numpy as np
    from PIL import Image, ImageDraw

    from svgsmith.pipeline import _supersample_candidate

    assert _supersample_candidate(_gradient_with_black_bars()) is True
    # Already-large art (resolution gate).
    assert _supersample_candidate(Image.new("RGB", (1100, 1100), (200, 50, 50))) is False
    # Synthetic clean flat: a handful of exact colors, nothing to consolidate.
    flat = Image.new("RGB", (200, 200), (0, 0, 0))
    ImageDraw.Draw(flat).rectangle([0, 0, 100, 200], fill=(200, 30, 30))
    assert _supersample_candidate(flat) is False
    # Smooth gradient / photo: many colors but no hard edges (below the floor).
    gx = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    gradient = Image.fromarray(np.stack([gx, gx, 255 - gx], axis=2), "RGB")
    assert _supersample_candidate(gradient) is False
    # Hatched / textured: many colours AND edges everywhere — excluded by the
    # edge-density CEILING (upscaling it would explode the trace).
    noise = np.random.default_rng(0).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    assert _supersample_candidate(Image.fromarray(noise, "RGB")) is False


def test_supersample_falls_back_when_the_trace_explodes(monkeypatch, tmp_path):
    """If a non-flat input slips past the cheap gate, the output-complexity guard
    trips and convert falls back to the baseline path — never a bloated SVG."""
    import svgsmith.pipeline as pipeline_module

    src = tmp_path / "cartoon.png"
    _gradient_with_black_bars().save(src)

    # Coverage (Tier 2) is the default for rich-colour inputs; disable it so the
    # supersample path (its fallback) is the one under test here.
    monkeypatch.setattr(pipeline_module, "_coverage_candidate", lambda image: False)

    # Baseline: gate forced off, so the standard median-cut path runs.
    monkeypatch.setattr(pipeline_module, "_supersample_candidate", lambda image: False)
    baseline_paths = convert(str(src), ConvertOptions(max_iters=1))[1].svg.paths

    # Gate forced on but the path cap set to 0, so the supersampled trace always
    # trips the guard and convert must fall back to exactly the baseline result.
    monkeypatch.setattr(pipeline_module, "_supersample_candidate", lambda image: True)
    monkeypatch.setattr(pipeline_module, "_SUPERSAMPLE_MAX_PATHS", 0)
    fellback_paths = convert(str(src), ConvertOptions(max_iters=1))[1].svg.paths
    assert fellback_paths == baseline_paths


def test_low_res_flat_cartoon_is_supersampled(monkeypatch, tmp_path):
    """End to end: with the coverage engine disabled, a triggering low-res input is
    traced at a larger internal resolution (supersample path), so the output viewBox
    exceeds the native size."""
    import svgsmith.pipeline as pipeline_module

    # Coverage (Tier 2) is the default; disable it to exercise the supersample path.
    monkeypatch.setattr(pipeline_module, "_coverage_candidate", lambda image: False)
    src = tmp_path / "cartoon.png"
    _gradient_with_black_bars().save(src)
    svg, report = convert(str(src), ConvertOptions(max_iters=1))
    root = ET.fromstring(svg)
    view_box = root.get("viewBox")
    assert view_box is not None
    assert max(float(v) for v in view_box.split()) > 300  # upscaled past native 300px
    assert report.mode_used == "color"


def test_low_res_rich_color_uses_coverage_engine_by_default(tmp_path):
    """Tier 2 (#67): a low-res, rich-colour input (gradient + hard bars) traces via the
    perceptual-coverage + region-cleanup engine by default — a clean, economical,
    faithful SVG at native resolution (the coverage path supersedes supersampling for
    rich-colour inputs)."""
    src = tmp_path / "cartoon.png"
    _gradient_with_black_bars().save(src)
    svg, report = convert(str(src), ConvertOptions(max_iters=1))
    assert report.mode_used == "color"
    assert report.similarity > 0.85  # faithful
    assert svg.count("<path") < 200  # economical (region cleanup), not bloated


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
