"""Tests for the svgsmith command-line interface."""

import shutil
import subprocess
import sys
from importlib.metadata import version

import pytest

from svgsmith import __version__
from svgsmith.cli import EXIT_ERROR, EXIT_OK, main


def test_root_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == EXIT_OK


def test_convert_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["convert", "--help"])
    assert excinfo.value.code == EXIT_OK


def test_no_command_prints_help_and_returns_ok():
    assert main([]) == EXIT_OK


def test_convert_parses_full_flag_contract():
    # The full flag contract must parse; a missing input file is a hard error
    # (exit 1), not a crash.
    code = main(
        [
            "convert",
            "does-not-exist.png",
            "--mode",
            "auto",
            "--quality",
            "0.9",
            "--max-iters",
            "4",
            "--no-editable",
            "--out",
            "out.svg",
            "--report",
            "json",
        ]
    )
    assert code == EXIT_ERROR


def test_convert_without_input_is_hard_error():
    assert main(["convert"]) == EXIT_ERROR


def test_invalid_mode_is_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["convert", "x.png", "--mode", "bogus"])
    assert excinfo.value.code == 2  # argparse usage error


def test_console_entrypoint_help_exits_zero():
    executable = shutil.which("svgsmith")
    if executable is None:
        pytest.skip("svgsmith console script is not installed on PATH")
    result = subprocess.run([executable, "--help"], capture_output=True)
    assert result.returncode == EXIT_OK


def test_module_invocation_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "svgsmith", "--help"], capture_output=True
    )
    assert result.returncode == EXIT_OK


def test_version_is_single_sourced():
    # The distribution version is derived from svgsmith.__version__ (dynamic), so
    # the package metadata and the module must never drift apart.
    assert version("svgsmith") == __version__


def test_cli_version_matches_package_version():
    result = subprocess.run(
        [sys.executable, "-m", "svgsmith", "--version"], capture_output=True, text=True
    )
    assert result.returncode == EXIT_OK
    assert __version__ in result.stdout
