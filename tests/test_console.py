"""The dependency-free console helpers ported from mew."""

import io

from jaxplorer._console import color_enabled, die, sgr


def test_sgr_wraps_and_resets():
    assert sgr("hi", "bold") == "\x1b[1mhi\x1b[0m"
    assert sgr("hi", "bold", "cyan") == "\x1b[1;36mhi\x1b[0m"


def test_sgr_is_a_no_op_without_style_or_text_or_when_disabled():
    assert sgr("hi") == "hi"
    assert sgr("", "bold") == ""
    assert sgr("hi", "bold", enabled=False) == "hi"


def test_no_color_beats_a_tty(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled(Tty()) is False


def test_force_color_beats_a_pipe(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert color_enabled(io.StringIO()) is True


def test_a_pipe_gets_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert color_enabled(io.StringIO()) is False


def test_die_prefixes_the_program_name(capsys):
    try:
        die("something broke")
    except SystemExit as exit_:
        assert exit_.code == 1
    assert capsys.readouterr().err == "jaxplorer: something broke\n"
