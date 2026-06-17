# Security Policy

## Reporting a Vulnerability

svgsmith processes local image files and does not transmit them anywhere.
If you discover a security issue (e.g. a crafted input causing unsafe behavior
in a bundled tracer or in SVG post-processing), please open a private security
advisory via GitHub's "Report a vulnerability" feature rather than a public issue.

## Scope

- svgsmith runs fully locally; no network egress, no telemetry.
- Bundled engines (VTracer, Potrace) and their versions are pinned; report
  upstream CVEs that affect the pinned versions here so we can bump them.
