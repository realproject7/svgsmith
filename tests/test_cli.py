"""Tests for the svgsmith command-line interface."""

import shutil
import subprocess
import sys

import pytest

from svgsmith.cli import EXIT_OK, main


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


def test_convert_parses_flag_contract():
    # The flag names are the contract for later tickets; parsing must succeed
    # and an actual conversion must raise NotImplementedError (not fake output).
    with pytest.raises(NotImplementedError):
        main(
            [
                "convert",
                "input.png",
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
