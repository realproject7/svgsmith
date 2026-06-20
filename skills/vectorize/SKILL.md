---
name: vectorize
description: Convert a raster image (PNG/JPG) into clean, editable SVG using the svgsmith CLI. Use when the user wants to vectorize, trace, or turn a bitmap/logo/icon/illustration/pixel-art image into SVG. Picks the engine by intent, self-verifies the output quality, and retries before reporting.
---

# vectorize

Turn a raster image into an **editable** SVG by driving the local `svgsmith` CLI, then
verify the result and decide whether to accept, retry, or report a limitation — without
asking the user to tune tracer flags.

## Prerequisite

`svgsmith` must be installed and on `PATH` (`svgsmith --help` should work). If it is not,
tell the user to install it (`pip install svgsmith`, plus the Potrace system binary for
line-art mode: `brew install potrace` / `apt-get install -y potrace`) and stop.

## Step 1 — Map intent to flags

Choose `--mode` and `--quality` from what the user asked for. When unsure, use `auto`.

| User intent | Flags |
|---|---|
| "clean logo", "crisp icon", line art, black & white | `--mode binary --quality 0.95` |
| "trace this illustration", colorful flat art, graphic | `--mode color` |
| "pixel art", sprite, low-res blocky image | `--mode pixel` |
| anything unspecified / "vectorize this" | `--mode auto` |

Then add **refinement flags** based on what the user wants out of it (these compose):

| User wants | Add flag |
|---|---|
| "clean / flat / solid background", "remove the background texture", "just the subject on a plain color", "isolate the cat/logo/person" | `--solid-background` |
| "make the background white / `<color>`", "change / swap the background color" | `--background white` |
| "cut it out", "transparent background", "remove the background entirely", "just the subject, no background" | `--transparent-background` |
| "maximum detail", "keep every detail / texture / shading" | `--detail high` |
| "cleaner / tidier", "less noise / grain", "smooth it out a bit" | `--detail clean` |
| "poster / flat / bold graphic", "simple flat colors", "minimalist" | `--detail poster` |
| "it looks scratchy / shattered / broken", "glossy or shiny art isn't clean", "flatten the shading" | `--flatten-shading` |
| "even / consistent outline", "uniform line weight" (only for art that already has a dark outline) | `--uniform-outline` |
| "keep the raw / rough / hand-drawn look", "don't smooth it" | `--no-smooth` |

`--detail` is the dial between fidelity and a clean/flat look; default `normal` is balanced.
Flags compose, e.g. *"a detailed cat on a clean solid background"* → `--detail high --solid-background`.

Always pass `--report json` and a sensible `--out` (default: input path with `.svg`).

## Step 2 — Run the CLI

```bash
svgsmith convert <input> --mode <mode> [--quality <q>] --out <output.svg> --report json
```

`stdout` is a single JSON object (the only thing on stdout). Parse it. Exit codes:
`0` = success (`similarity >= quality`), `2` = SVG produced but below the quality target,
`1` = hard error.

The report fields: `output, mode_used, engine, preset, iterations, similarity,
passed_threshold, svg:{paths,groups,colors,bytes}, warnings[]`.

## Step 3 — Decide: accept / retry / report

- **`passed_threshold: true` (exit 0)** → **accept.** Report the output path and the
  similarity score to the user.
- **Below threshold (exit 2), first attempt** → **retry once**, escalating effort:
  - if `mode_used` was `auto`, re-run with the explicit mode that fits the image
    (binary for sharp/flat, color for many colors);
  - otherwise raise `--max-iters` (e.g. to 8) and/or lower `--quality` toward a realistic
    target for the content. If the report `warnings` mention photographic gradients, do
    **not** chase a high score — photos vectorize poorly by nature.
- **Still below threshold after the retry** → **report honestly.** Give the user the SVG
  anyway (it is still produced), the achieved `similarity`, and the reason from `warnings`.
  Do not silently present a poor result as good.
- **Exit 1 (hard error)** → surface the stderr message; check the prerequisite (missing
  `potrace` binary is the most common cause).

## Step 4 — Visual self-check

Before declaring success, **look at the output**: open/render the produced SVG and compare
it to the source image. Confirm the shapes, colors, and proportions match; note any obvious
defects (lost detail, wrong colors, broken paths). If the SVG looks wrong despite a passing
score, say so — the score is a guide, not a guarantee.

## Notes

- `--editable` is on by default (grouped `<g>` layers, simplified paths, consolidated
  palette). Only pass `--no-editable` if the user explicitly wants the raw traced output.
- Everything runs locally; the image never leaves the machine.
