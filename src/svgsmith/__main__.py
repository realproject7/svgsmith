"""Allow ``python -m svgsmith`` to invoke the CLI."""

from svgsmith.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
