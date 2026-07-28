"""Argument parsing and help rendering. The TUI itself is never launched here."""

import subprocess
import sys

import pytest

from jaxplorer.cli import build_parser, main
from jaxplorer.protocol import ALL_STAGES
from jaxplorer.session import DEFAULT_TIMEOUT


def test_defaults():
    args = build_parser().parse_args([])

    assert args.file is None
    assert args.watch is False
    assert args.platform is None
    assert args.x64 is False
    assert args.passes is False
    assert args.stages == list(ALL_STAGES)
    assert args.timeout == DEFAULT_TIMEOUT


def test_full_invocation(tmp_path):
    path = tmp_path / "snippet.py"
    path.touch()
    args = build_parser().parse_args(
        [str(path), "--watch", "--platform", "cpu", "--x64", "--passes", "--timeout", "5"]
    )

    assert args.file == path
    assert args.watch and args.x64 and args.passes
    assert args.platform == "cpu"
    assert args.timeout == 5.0


def test_stages_are_returned_in_pipeline_order_however_they_are_typed():
    args = build_parser().parse_args(["--stages", "analysis,jaxpr"])

    assert args.stages == ["jaxpr", "analysis"]


def test_an_unknown_stage_is_rejected(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--stages", "jaxpr,mlir"])

    assert "unknown stage(s) mlir" in capsys.readouterr().err


def test_an_unknown_platform_is_rejected(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--platform", "fpga"])

    assert "invalid choice" in capsys.readouterr().err


def test_watch_without_a_file_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["--watch"])

    assert "--watch needs a FILE" in capsys.readouterr().err


def test_missing_file_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["does-not-exist.py"])

    assert "no such file" in capsys.readouterr().err


def test_help_uses_spiky_metavars_and_stays_plain_when_piped():
    help_text = build_parser().format_help()

    assert "<stages>" in help_text
    assert "<timeout>" in help_text
    assert "TIMEOUT" not in help_text
    # A flag with choices shows them instead, which is more useful than any metavar.
    assert "{cpu,gpu,tpu}" in help_text
    # Captured stdout is not a TTY, so no escape codes may leak into pipes or CI logs.
    assert "\x1b[" not in help_text


def test_the_console_script_runs_and_prints_help():
    # Nothing else exercises the installed entry point end to end, so a break in the
    # wiring between cli.main and the TUI would otherwise pass every other test.
    done = subprocess.run(
        [sys.executable, "-m", "jaxplorer", "--help"], capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 0
    assert "compiler explorer TUI for JAX" in done.stdout
    assert "--stages" in done.stdout
    assert "\x1b[" not in done.stdout


def test_the_console_script_rejects_a_bad_file():
    done = subprocess.run(
        [sys.executable, "-m", "jaxplorer", "nope.py"], capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 2
    assert "no such file" in done.stderr
