"""Minimal terminal rendering, so the CLI needs no third-party dependency.

Scope is only what jaxplorer prints outside the TUI: colorized ``--help`` and the odd
diagnostic. There is no markup language, so text containing ``f32[8,16]`` needs no
escaping.

Ported from mew's ``_console`` to keep both projects' CLIs behaving the same.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

# Style name -> ANSI SGR code (only the styles jaxplorer uses).
_SGR = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}


def sgr(text: str, *styles: str, enabled: bool = True) -> str:
    """Wrap ``text`` in ANSI codes.

    Parameters
    ----------
    text : str
        Text to style.
    *styles : str
        Style names, each a key of the module's SGR table.
    enabled : bool, optional
        Pass ``False`` to return ``text`` untouched, so callers can gate color once rather
        than at every call site.

    Returns
    -------
    str
        The styled text, or ``text`` unchanged when disabled, unstyled, or empty.
    """
    names = [s for s in styles if s]
    if not enabled or not names or not text:
        return text
    codes = ";".join(_SGR[s] for s in names)
    return f"\x1b[{codes}m{text}\x1b[0m"


def terminal_width(default: int = 80) -> int:
    """Return the terminal's column count, or ``default`` if it has none."""
    import shutil

    return shutil.get_terminal_size((default, 24)).columns


def color_enabled(stream: TextIO) -> bool:
    """Whether to emit ANSI on ``stream``.

    Parameters
    ----------
    stream : TextIO
        Sink that would be written to.

    Returns
    -------
    bool
        ``NO_COLOR`` wins, then ``FORCE_COLOR``, then whether the stream is a TTY.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def die(message: str, code: int = 1) -> None:
    """Report a fatal error as ``jaxplorer: message`` on stderr and exit.

    Parameters
    ----------
    message : str
        What went wrong, lowercase and without a trailing period.
    code : int, optional
        Exit status.

    Raises
    ------
    SystemExit
        Always.
    """
    print(f"jaxplorer: {message}", file=sys.stderr)
    raise SystemExit(code)
