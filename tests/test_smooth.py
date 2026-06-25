"""Tests for the curve-smoothing post-pass (Schneider fit + axis-snap)."""

import math

import numpy as np

from svgsmith.smooth import _axis_snap


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
