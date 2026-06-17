"""Potrace-backed adapter for line-art / monochrome images.

Shells out to the system ``potrace`` binary (installed via apt in CI) rather
than binding to pypotrace, and exposes the uniform ``trace(image, preset) -> str``
interface. The image is binarized to 1-bit, fed to ``potrace`` as a PBM bitmap
over stdin, and the SVG is read back from stdout. See README for the system
dependency.
"""

from __future__ import annotations

import io
import shutil
import subprocess

from .base import ImageInput, Preset, load_image

POTRACE_BINARY = "potrace"


class PotraceNotFoundError(RuntimeError):
    """Raised when the system ``potrace`` binary cannot be located."""


class BinaryTracer:
    """Trace monochrome / line-art images into SVG via the potrace binary."""

    def __init__(self, potrace_path: str | None = None) -> None:
        self._potrace = potrace_path or shutil.which(POTRACE_BINARY)

    def _binarize(self, image: ImageInput, threshold: float):
        gray = load_image(image, "L")
        cut = round(max(0.0, min(1.0, threshold)) * 255)
        # Pixels brighter than the cut become white (255); darker pixels become
        # black (0), which is what potrace traces.
        return gray.point(lambda p, c=cut: 255 if p > c else 0, mode="1")

    def trace(self, image: ImageInput, preset: Preset) -> str:
        if self._potrace is None:
            raise PotraceNotFoundError(
                f"the {POTRACE_BINARY!r} binary was not found on PATH; install it "
                "(e.g. `apt-get install potrace`)"
            )

        bitmap = self._binarize(image, preset.threshold)
        buffer = io.BytesIO()
        bitmap.save(buffer, format="PPM")  # 1-bit mode is written as binary PBM

        command = [
            self._potrace,
            "--svg",
            "--output",
            "-",
            "--turdsize",
            str(preset.turdsize),
            "--alphamax",
            str(preset.alphamax),
            "--opttolerance",
            str(preset.opttolerance),
            "-",
        ]
        result = subprocess.run(
            command,
            input=buffer.getvalue(),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"potrace failed (exit {result.returncode}): {detail}")
        return result.stdout.decode("utf-8")
