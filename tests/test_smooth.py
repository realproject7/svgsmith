"""Tests for the curve-smoothing post-pass (Schneider fit + axis-snap)."""

import math
import xml.etree.ElementTree as ET

import numpy as np

from svgsmith.smooth import _axis_snap, _resolution_precision, smooth_svg


def _root(view_box: str) -> ET.Element:
    return ET.fromstring(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}"></svg>'
    )


def test_resolution_precision_drops_decimals_for_supersampled_viewbox():
    # 640px native traced at 1536 (factor 2.4): .2f resolves ~2.4x finer than the
    # native pixel grid, so the (one-guard-decimal) lossless decimal count is .1f.
    assert _resolution_precision(_root("0 0 1536 1536"), 640, 2) == 1
    # 1024px native at 4096 (factor 4): still .1f with the guard decimal kept.
    assert _resolution_precision(_root("0 0 4096 4096"), 1024, 2) == 1
    # Extreme 20x supersample (tiny pixel art): safe to drop to integer coords.
    assert _resolution_precision(_root("0 0 2000 2000"), 100, 2) == 0


def test_resolution_precision_noops_without_native_or_supersample():
    # Unknown native size -> keep the requested precision untouched.
    assert _resolution_precision(_root("0 0 1536 1536"), 0, 2) == 2
    # Native-size trace (no supersample) -> no decimals dropped.
    assert _resolution_precision(_root("0 0 640 640"), 640, 2) == 2


def test_resolution_precision_never_increases_precision():
    # The lever only ever saves bytes: it must not raise precision above the request.
    assert _resolution_precision(_root("0 0 100 100"), 1000, 1) == 1


def test_smooth_svg_native_long_edge_reduces_coordinate_decimals():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1536 1536">'
        '<path d="M100.12 200.34 L300.56 400.78 L100.12 200.34 Z"/></svg>'
    )
    out = smooth_svg(svg, native_long_edge=640)
    # No coordinate should carry two decimals once auto-precision picks .1f.
    assert ".12" not in out and ".34" not in out


def test_axis_snap_snaps_near_horizontal_and_keeps_length():
    # A segment ~3deg off horizontal snaps to exact horizontal, length preserved.
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.5])  # ~2.9deg tilt
    snapped, drift = _axis_snap(start, end, snap_deg=8.0, max_drift=100.0)
    assert abs(snapped[1] - start[1]) < 1e-6  # now exactly horizontal
    assert math.isclose(np.hypot(*(snapped - start)), np.hypot(*(end - start)), rel_tol=1e-9)
    assert drift > 0.0


def test_axis_snap_leaves_true_diagonal_untouched():
    # A genuine 30deg diagonal is outside the snap band → returned unchanged.
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 5.77])  # ~30deg
    snapped, drift = _axis_snap(start, end, snap_deg=8.0, max_drift=100.0)
    assert np.allclose(snapped, end)
    assert drift == 0.0


def test_axis_snap_refuses_when_drift_exceeds_budget():
    # Near-axis but a tiny drift budget forbids the move (no accumulation).
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.5])
    snapped, drift = _axis_snap(start, end, snap_deg=8.0, max_drift=0.01)
    assert np.allclose(snapped, end)  # snap refused
