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

from jaxplorer import __version__, examples
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
        from jaxplorer._console import color_enabled, sgr

        # Checked at format time, not import: anything capturing stdout gets plain text.
        if not color_enabled(sys.stdout):
            return text

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
        "file",
        nargs="?",
        type=Path,
        help="Snippet to open; omit for a scratch buffer. A name that is not a path is "
        f"looked up among the bundled examples ({', '.join(examples.names())}).",
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
        help=f"Comma-separated subset of {', '.join(ALL_STAGES)}. The chain stops after the "
        "last one asked for, so leaving out optimized_hlo skips XLA, which is most of a "
        "compile on a large model.",
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
    parser.add_argument(
        "--print",
        dest="print_pane",
        metavar="<pane>",
        choices=(*ALL_STAGES, "passes", "llvm_ir"),
        help="Print one pane to stdout and exit instead of starting the TUI, for piping "
        f"and scripting. One of {', '.join((*ALL_STAGES, 'passes', 'llvm_ir'))}.",
    )
    parser.add_argument(
        "--structural-diff",
        action="store_true",
        help="Compare pass snapshots as graphs rather than as text, which is what f4 "
        "toggles in the TUI.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="List the bundled examples and exit.",
    )
    return parser


def _print_pane(args: argparse.Namespace, *, path: Path | None, source: str | None) -> int:
    """Compile once, write one pane to stdout, and return an exit status.

    Deliberately does not touch :mod:`jaxplorer.app`, so ``--print`` needs no terminal and
    does not import textual. The compile still happens in the worker subprocess, which is
    where jax lives.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, including ``print_pane``.
    path : Path or None
        File the snippet came from, for tracebacks and click-to-source metadata.
    source : str or None
        Snippet text, when it did not come from ``path``.

    Returns
    -------
    int
        0 when the pane was printed, 1 when the compile or that stage failed.
    """
    import asyncio

    from jaxplorer._console import color_enabled, sgr
    from jaxplorer.hlo import pass_report
    from jaxplorer.session import WorkerSession

    pane = args.print_pane
    if source is None:
        source = path.read_text(encoding="utf-8") if path is not None else ""

    # Ask for what is being printed, whatever --stages says: the chain now stops after the
    # last requested stage, so a narrow --stages would otherwise leave the pane empty.
    wanted = "optimized_hlo" if pane in ("passes", "llvm_ir") else pane
    stages = [stage for stage in ALL_STAGES if stage in args.stages or stage == wanted]
    passes = args.passes or pane in ("passes", "llvm_ir")

    async def run() -> int:
        session = WorkerSession(
            platform=args.platform,
            x64=args.x64,
            timeout=args.timeout,
            executable=args.python,
        )
        try:
            result = await session.compile(
                source, str(path) if path else "<snippet>", stages, passes
            )
        finally:
            await session.close()

        # color_enabled, not a bare isatty: NO_COLOR and FORCE_COLOR have to be honoured on
        # this path too, and it is the same helper --help styling goes through.
        color = color_enabled(sys.stderr)
        if result is None:
            # Only reachable if a newer request superseded this one, and there is no newer
            # request here. Say so rather than exiting non-zero in silence.
            print(
                sgr("the compile was superseded before it finished", "red", enabled=color),
                file=sys.stderr,
            )
            return 1
        if result.fatal:
            print(sgr(result.fatal, "red", enabled=color), file=sys.stderr)
            return 1
        if pane == "passes":
            print(pass_report(result.passes, structural=args.structural_diff))
            return 0
        if pane == "llvm_ir":
            if not result.llvm_ir:
                print(
                    sgr("this backend emitted no LLVM IR", "yellow", enabled=color),
                    file=sys.stderr,
                )
                return 1
            print(result.llvm_ir)
            return 0
        outcome = result.stages.get(pane)
        if outcome is None or outcome.error or outcome.text is None:
            message = outcome.error if outcome and outcome.error else f"{pane} produced nothing"
            print(sgr(message, "red", enabled=color), file=sys.stderr)
            return 1
        print(outcome.text)
        return 0

    return asyncio.run(run())


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the TUI until the user quits, or print one pane and exit.

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

    if args.examples:
        for name in examples.names():
            print(name)
        return 0
    if args.watch and args.file is None:
        parser.error("--watch needs a FILE to watch")
    if args.watch and args.print_pane is not None:
        # --print compiles once and exits, so there is nothing for a reload to update.
        parser.error("--watch and --print do opposite things; pick one")
    if args.python is not None and not os.access(args.python, os.X_OK):
        parser.error(f"not an executable interpreter: {args.python}")

    # A name that is not a path may be a bundled example. Those are read out of the wheel, so
    # they open as a scratch buffer with no path: saving over a file in site-packages would be
    # a surprising thing for `jaxplorer mlp` to do.
    path: Path | None = args.file
    source: str | None = None
    if args.file is not None and not args.file.is_file():
        source = examples.load(args.file.name)
        if source is None:
            parser.error(
                f"no such file: {args.file}\nbundled examples: {', '.join(examples.names())}"
            )
        if args.watch:
            parser.error(f"--watch needs a file on disk; {args.file.name} is a bundled example")
        path = None

    if args.print_pane is not None:
        return _print_pane(args, path=path, source=source)

    # Imported late so that --help stays instant.
    from jaxplorer.app import JaxplorerApp

    JaxplorerApp(
        path=path,
        source=source,
        watch=args.watch,
        platform=args.platform,
        x64=args.x64,
        stages=args.stages,
        passes=args.passes,
        timeout=args.timeout,
        executable=args.python,
        structural_diff=args.structural_diff,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
