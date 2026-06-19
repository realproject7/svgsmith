## Active Batch — svgsmith follow-up fixes (pre-PyPI hygiene)

@head — fix the confirmed adversarial-review follow-ups before publish. Three issues, each
self-contained with file:line and the concrete fix. Assign @dev; require @re1 AND @re2 review;
merge on dual approval. Branches `task/<issue>-<slug>`, PR `Closes #<n>` (one keyword per issue).

1. **#22 packaging: modernize license metadata for PyPI** — in `pyproject.toml`: replace
   `license = { text = "MIT" }` with SPDX `license = "MIT"`, remove the redundant
   `License :: OSI Approved :: MIT License` classifier, add `license-files = ["LICENSE"]`.
   Ensure `python -m build` produces a clean wheel/sdist with no license deprecation warning.

2. **#23 packaging: single-source the version** — version is duplicated in `pyproject.toml`
   and `src/svgsmith/__init__.py`. Single-source it (e.g. `[tool.setuptools.dynamic]
   version = {attr = "svgsmith.__version__"}` with `dynamic = ["version"]`). Confirm
   `svgsmith --version` and the built wheel metadata agree. Add a test if practical.

3. **#25 cli: validate --quality and --max-iters bounds** — reject `--quality` outside
   `[0,1]` and `--max-iters < 1` with a clear stderr error and exit 1, before any work
   (confirmed bugs: `--quality -1` exits 0/pass; `--quality 1.5` runs; `--max-iters 0` hits
   an unhelpful `conversion failed:` path). Add tests for low/high quality and zero iters.

Acceptance for the batch: all three issues closed, `pytest` + `ruff` green, CI green on 3.11/3.12,
`python -m build` clean. No stub/mock; concise changes only. When done, @head posts
"FIX BATCH COMPLETE — #22 #23 #25 merged" so the PO can do final publish prep.
