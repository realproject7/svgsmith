"""Uniform tracing interface over the bundled engines.

Import the adapters and preset helpers from here; downstream code (T3 routing,
T5 postprocess, T6 verify) depends only on ``trace(image, preset) -> str``.
"""

from .base import (
    PRESETS,
    ImageInput,
    Preset,
    Tracer,
    get_preset,
    load_image,
)
from .binary import BinaryTracer, PotraceNotFoundError
from .color import ColorTracer

__all__ = [
    "PRESETS",
    "BinaryTracer",
    "ColorTracer",
    "ImageInput",
    "PotraceNotFoundError",
    "Preset",
    "Tracer",
    "get_preset",
    "load_image",
]
