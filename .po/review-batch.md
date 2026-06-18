## Active Batch — svgsmith adversarial PR review (post-merge)

@head — the MVP engine/pipeline batch (#2–#9) is merged to `main`. Run an **adversarial pr-review batch** over the merged PRs **#13–#20**. Goal: find real defects before we publish to PyPI.

Assign @re1 AND @re2 to review the merged code (not just diffs — read the integrated `main`). Each reviewer independently hunts for:
- **Correctness bugs** in: classify heuristics, preprocess (quantization/bg-removal/denoise), engine adapters (VTracer/Potrace shell-out, error handling when `potrace` missing), postprocess (path simplification correctness — no broken curves, group/palette integrity), the verify loop (SSIM scoring, re-tune actually improves, best-result selection, iteration cap), CLI/report (exit codes 0/1/2, stdout = JSON only, `--no-editable` path).
- **Stub/mock/placeholder/temporary code** — must be ZERO. Flag any.
- **Over-engineering** — needless abstraction/config/indirection, duplicate helpers, premature generality. Flag and propose the simpler form.
- **Edge cases**: tiny images, single-color images, images with alpha, missing `potrace` binary, malformed input, `--max-iters 0`, `--quality 1.0`.
- **Security**: any unsafe subprocess/shell usage in the Potrace shell-out (no shell injection via filenames), no network egress.

For every confirmed finding, **file a GitHub issue** titled `[follow-up] <area>: <short>`, labeled `follow-up`, with: file:line, why it's a defect, and the concrete fix. Do NOT fix in this batch — just file well-scoped follow-up issues. When both reviewers are done, @head posts a summary message: "REVIEW COMPLETE — N follow-ups filed: #.. #..".

If zero real defects: @head posts "REVIEW COMPLETE — 0 follow-ups" so the PO can proceed to publish prep.
