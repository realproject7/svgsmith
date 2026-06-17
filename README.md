# svgsmith

> Agent-native, self-verifying raster→SVG vectorizer.

`svgsmith` turns PNG/JPG images into **editable** SVG. It is built to be driven by an
AI agent without a human in the loop: it picks the right tracing engine for the input,
post-processes the result into clean editable layers, and **verifies its own output** by
re-rasterizing the SVG and comparing it to the original — re-tuning until a quality
threshold is met. Every run returns a structured JSON report so a calling agent can
decide whether to accept, retry, or escalate.

It does **not** reinvent tracing. It wraps proven engines
([VTracer](https://github.com/visioncortex/vtracer) for color,
[Potrace](https://potrace.sourceforge.net/) for line art) and adds the layer that is
missing for agent use: routing, editable output, and a self-verification loop.

> **Status:** early development. See the [EPIC](../../issues) for the build plan.

## System dependencies

The line-art engine shells out to the [**Potrace**](https://potrace.sourceforge.net/)
binary (svgsmith does not bundle a Potrace Python binding). Install it from your
package manager before use:

```bash
# Debian / Ubuntu
sudo apt-get install -y potrace
# macOS (Homebrew)
brew install potrace
```

The color engine ([VTracer](https://github.com/visioncortex/vtracer)) ships as a
pinned PyPI wheel and needs no system package.

## What makes it different

- **Auto-routing** — classifies the input (logo/icon vs illustration vs pixel art) and
  selects the engine + preset automatically. No tracer-flag expertise required.
- **Editable output** — instead of one monolithic `<path>`, output is grouped into
  `<g>` layers with simplified paths and a consolidated color palette.
- **Self-verifying** — converts, re-rasterizes, diffs against the original (SSIM), and
  re-tunes parameters until it converges on a quality target.
- **Structured report** — emits JSON (mode, engine, iterations, similarity score,
  warnings) so agents can branch programmatically.
- **Local & private** — runs fully offline; images never leave the machine.

## Planned usage

```bash
svgsmith convert input.png \
  --mode auto \         # auto | binary | color | pixel
  --quality 0.9 \       # target similarity (0–1), drives the verify loop
  --max-iters 4 \
  --editable \          # editable layered output (default on)
  --out output.svg \
  --report json
```

```json
{
  "output": "output.svg",
  "mode_used": "color",
  "engine": "vtracer",
  "iterations": 2,
  "similarity": 0.93,
  "passed_threshold": true,
  "svg": { "paths": 84, "groups": 6, "colors": 12, "bytes": 14820 },
  "warnings": []
}
```

## License

MIT — see [LICENSE](LICENSE).
