"""argparse CLI for the jaxplorer TUI.

stdlib argparse, no third-party CLI dependency. Help is colorized with a small ANSI
helper (:mod:`jaxplorer._console`), following mew's conventions so both CLIs read the same.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from jaxplorer import __version__
from jaxplorer.protocol import ALL_STAGES
from jaxplorer.session import DEFAULT_TIMEOUT

PLATFORMS = ("cpu", "gpu", "tpu")


class _HelpFormatter(argparse.HelpFormatter):
    """Help formatter with ``<spiky-brace>`` metavars and light ANSI color.

    Styling wraps regex matches *after* argparse has laid the text out, so column alignment
    is untouched. It falls back to plain text off a TTY, keeping pipes and CI logs clean.
    """

    def _metavar(self, action: argparse.Action) -> str:
        return f"<{action.dest.replace('_', '-')}>"

    def _get_default_metavar_for_optional(self, action: argparse.Action) -> str:
        return self._metavar(action)

    def _get_default_metavar_for_positional(self, action: argparse.Action) -> str:
        return self._metavar(action)

    def format_help(self) -> str:
        """Return the help text, colorized when stdout is an interactive terminal."""
        text = super().format_help()
        # TTY check at format time, not import: anything capturing stdout gets plain text.
        if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
            return text
        from jaxplorer._console import sgr

        # The patterns cannot overlap, and ANSI codes carry no -, --, or <>, so the
        # substitutions never nest.
        for pattern, style in (
            (r"(?m)^[A-Za-z][A-Za-z ]*:", "bold"),
            (r"(?<![\w-])--[A-Za-z][\w-]*", "cyan"),
            (r"(?<![\w-])-[A-Za-z](?![\w-])", "green"),
            (r"<[\w-]+>", "yellow"),
        ):
            text = re.sub(pattern, lambda m, s=style: sgr(m.group(), s), text)
        return text


def _stage_list(value: str) -> list[str]:
    """argparse type for --stages: ``'jaxpr,stablehlo'`` to ``['jaxpr', 'stablehlo']``."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected at least one stage")
    unknown = [name for name in names if name not in ALL_STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {', '.join(unknown)}; choose from {', '.join(ALL_STAGES)}"
        )
    # Pipeline order, not the order they were typed: a later stage cannot run without
    # the ones it consumes anyway.
    return [name for name in ALL_STAGES if name in names]


def build_parser() -> argparse.ArgumentParser:
    """Build jaxplorer's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured with :class:`_HelpFormatter`, so ``--help`` is colorized on a TTY.
    """
    parser = argparse.ArgumentParser(
        prog="jaxplorer",
        formatter_class=_HelpFormatter,
        description="A compiler explorer TUI for JAX: Inspect the jaxpr, StableHLO and "
        "optimized HLO of a jitted function as you edit it.",
        epilog="A snippet must define a callable `f` and a tuple `args` of example "
        "inputs (concrete arrays or jax.ShapeDtypeStruct).",
    )
    parser.add_argument(
        "file", nargs="?", type=Path, help="Snippet to open; omit for a scratch buffer."
    )
    parser.add_argument("--version", action="version", version=f"jaxplorer {__version__}")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Reload FILE when it changes on disk and keep the buffer read-only, "
        "so you can edit in your own editor.",
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        metavar="(cpu|gpu|tpu)",
        help="JAX platform to compile for; defaults to JAX's own choice.",
    )
    parser.add_argument(
        "--python",
        metavar="<path>",
        help="Interpreter to compile with, e.g. a project's .venv/bin/python. Lets one "
        "installed jaxplorer inspect whichever jax that environment has, instead of the "
        "one it was installed with.",
    )
    parser.add_argument("--x64", action="store_true", help="enable 64-bit values")
    parser.add_argument(
        "--stages",
        type=_stage_list,
        default=list(ALL_STAGES),
        help=f"Comma-separated subset of {', '.join(ALL_STAGES)}. Stopping before "
        "optimized_hlo skips XLA entirely, which is much faster on a large model.",
    )
    parser.add_argument(
        "--passes",
        action="store_true",
        help="Also collect per-pass HLO snapshots and LLVM IR, via XLA's dump flags.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Give up on a compile after this long, in seconds (default: {DEFAULT_TIMEOUT:g}s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the TUI until the user quits.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    SystemExit
        On a usage error, via argparse, with status 2.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.watch and args.file is None:
        parser.error("--watch needs a FILE to watch")
    if args.file is not None and not args.file.is_file():
        parser.error(f"no such file: {args.file}")
    if args.python is not None and not os.access(args.python, os.X_OK):
        parser.error(f"not an executable interpreter: {args.python}")

    # Imported late so that --help stays instant.
    from jaxplorer.app import JaxplorerApp

    JaxplorerApp(
        path=args.file,
        watch=args.watch,
        platform=args.platform,
        x64=args.x64,
        stages=args.stages,
        passes=args.passes,
        timeout=args.timeout,
        executable=args.python,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
