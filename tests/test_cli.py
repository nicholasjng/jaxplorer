"""Argument parsing and help rendering. The TUI itself is never launched here."""

import subprocess
import sys

import pytest

from jaxplorer import __version__, examples
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

    error = capsys.readouterr().err
    assert "no such file" in error
    # A bare name is a valid way in, so say which ones exist.
    assert "bundled examples" in error


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_the_bundled_examples_are_reachable_from_the_package():
    # Asserted through importlib.resources, not a repo-relative path: an install has no
    # examples/ directory beside it.
    assert "mlp" in examples.names()
    assert examples.load("mlp") == examples.load("mlp.py")
    source = examples.load("mlp")
    assert source is not None
    assert "args" in source
    assert examples.load("nope") is None


def test_a_bundled_example_name_is_accepted_in_place_of_a_path(monkeypatch, tmp_path, capsys):
    # From a directory with no examples/ in it, as any installed user is.
    monkeypatch.chdir(tmp_path)

    assert main(["mlp", "--print", "jaxpr", "--platform", "cpu"]) == 0

    assert "dot_general" in capsys.readouterr().out


def test_watching_a_bundled_example_is_rejected(capsys):
    # It lives in site-packages; there is nothing sensible to watch or save over.
    with pytest.raises(SystemExit):
        main(["mlp", "--watch"])

    assert "bundled example" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("pane", "needle"),
    [
        ("jaxpr", "dot_general"),
        ("stablehlo", "stablehlo"),
        ("optimized_hlo", "HloModule"),
        ("analysis", "Cost:"),
        ("passes", "pipeline order"),
    ],
)
def test_print_writes_one_pane_to_stdout(capsys, pane, needle):
    assert main(["mlp", "--print", pane, "--platform", "cpu"]) == 0

    assert needle in capsys.readouterr().out


def test_print_implies_the_stage_it_needs(capsys):
    # --stages stops the chain, so printing a pane outside it must widen the request.
    assert main(["mlp", "--print", "optimized_hlo", "--stages", "jaxpr", "--platform", "cpu"]) == 0

    assert "HloModule" in capsys.readouterr().out


def test_print_reports_a_failing_snippet_on_stderr(capsys, tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("def f(x):\n    return x.nope()\n\nargs = (1.0,)\n")

    assert main([str(path), "--print", "jaxpr", "--platform", "cpu"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nope" in captured.err


def test_print_passes_can_use_the_structural_diff(capsys):
    assert main(["scan", "--print", "passes", "--structural-diff", "--platform", "cpu"]) == 0

    out = capsys.readouterr().out
    assert "pipeline order" in out
    # The structural renderer's shape, which the text diff never emits.
    assert "@@" not in out


def test_examples_can_be_listed(capsys):
    assert main(["--examples"]) == 0

    listed = capsys.readouterr().out.split()
    assert listed == list(examples.names())


def test_watching_and_printing_together_is_rejected(capsys, tmp_path):
    path = tmp_path / "snippet.py"
    path.touch()

    with pytest.raises(SystemExit):
        main([str(path), "--watch", "--print", "jaxpr"])

    assert "pick one" in capsys.readouterr().err


def test_listing_the_examples_costs_no_filesystem_walk_per_call():
    # The names go into --help, which must not wait on a directory scan.
    assert examples.names() is examples.names()


def test_an_unknown_print_pane_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["mlp", "--print", "nonsense"])

    assert "invalid choice" in capsys.readouterr().err


def test_help_uses_spiky_metavars_and_stays_plain_when_piped():
    help_text = build_parser().format_help()

    assert "<stages>" in help_text
    assert "<timeout>" in help_text
    assert "TIMEOUT" not in help_text
    assert "(cpu|gpu|tpu)" in help_text
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


def test_python_flag_is_parsed():
    args = build_parser().parse_args(["--python", sys.executable])

    assert args.python == sys.executable


def test_a_non_executable_interpreter_is_rejected(capsys, tmp_path):
    not_exec = tmp_path / "notpython"
    not_exec.write_text("")

    with pytest.raises(SystemExit):
        main(["--python", str(not_exec)])

    assert "not an executable interpreter" in capsys.readouterr().err
