"""VTracer-backed adapter for full-color raster images.

Wraps the ``vtracer`` PyPI package and exposes the uniform
``trace(image, preset) -> str`` interface. Produces layered, multi-color SVG.
"""

from __future__ import annotations

import vtracer

from .base import ImageInput, Preset, load_image


class ColorTracer:
    """Trace color images into multi-color SVG via VTracer."""

    def trace(self, image: ImageInput, preset: Preset) -> str:
        img = load_image(image, "RGBA")
        # get_flattened_data() is the forward-compatible accessor; fall back to
        # getdata() on older Pillow releases.
        accessor = getattr(img, "get_flattened_data", img.getdata)
        pixels = list(accessor())
        return vtracer.convert_pixels_to_svg(
            pixels,
            img.size,
            colormode=preset.color_mode,
            hierarchical=preset.hierarchical,
            mode=preset.curve_mode,
            filter_speckle=preset.filter_speckle,
            color_precision=preset.color_precision,
            layer_difference=preset.layer_difference,
            corner_threshold=preset.corner_threshold,
            length_threshold=preset.length_threshold,
            splice_threshold=preset.splice_threshold,
            path_precision=preset.path_precision,
        )
