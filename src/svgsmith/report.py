"""The canonical svgsmith JSON report — the agent contract.

This ticket (T7) owns the report schema. Upstream stages emit raw signals
(SSIM scores from the verify loop, warning strings from the classifier); they
are assembled here. The field names below are the stable contract the T9 skill
reads — do not rename them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from defusedxml import ElementTree as DefusedET

SVG_NS = "http://www.w3.org/2000/svg"


@dataclass
class SvgStats:
    """Structural summary of the produced SVG."""

    paths: int
    groups: int
    colors: int
    bytes: int


@dataclass
class Report:
    """Structured result of a conversion (serializes to the contract JSON)."""

    output: str
    mode_used: str
    engine: str
    preset: str
    iterations: int
    similarity: float
    passed_threshold: bool
    svg: SvgStats
    # Mean perceptual color distance (ΔE Lab) vs the original (#37) — the color-fidelity
    # channel SSIM is blind to. Reported for gates/inspection; not yet part of pass/fail.
    color_error: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def svg_stats(svg: str) -> SvgStats:
    """Compute path/group/color counts and byte size for an SVG string."""
    root = DefusedET.fromstring(svg)
    paths = 0
    groups = 0
    colors: set[str] = set()
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "path":
            paths += 1
        elif tag == "g":
            groups += 1
        fill = element.get("fill")
        if fill and fill != "none":
            colors.add(fill.lower())
    return SvgStats(
        paths=paths,
        groups=groups,
        colors=len(colors),
        bytes=len(svg.encode("utf-8")),
    )
