"""Compile worker: runs a snippet and reports its IR at every stage.

Runs as ``python -m jaxplorer.worker`` in a subprocess owned by :mod:`jaxplorer.session`, so
that the seconds-long JAX import is paid once, an XLA abort cannot take the TUI down, and the
platform and x64 flags can be fixed before JAX is imported.

Must not import textual.
"""

from __future__ import annotations

import json
import linecache
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jaxplorer.protocol import (
    ALL_STAGES,
    CompileRequest,
    CompileResult,
    PassSnapshot,
    Stage,
    StageResult,
    encode_frame,
)

# Snippet globals that configure the jit wrapper rather than describing the call.
JIT_OPTION_NAMES = ("static_argnums", "static_argnames", "donate_argnums")


class ContractError(Exception):
    """The snippet ran, but does not describe something jaxplorer can compile.

    Distinct from any other failure because the fix is to edit ``f`` or ``args``, so the
    message is guidance rather than a traceback.
    """


@contextmanager
def _protocol_channel():
    """Yield a private binary channel for protocol frames.

    A snippet may ``print`` and XLA logs straight to fd 1, either of which would corrupt
    the stream, so fd 1 is pointed at stderr for the worker's life and frames go out
    on a duplicate of the original.
    """
    saved = os.dup(1)
    os.dup2(2, 1)
    channel = os.fdopen(saved, "wb", buffering=0)
    try:
        yield channel
    finally:
        channel.close()


def _format_error(exc: BaseException, filename: str) -> str:
    """Format ``exc`` keeping only stack frames from ``filename``.

    Worker and JAX-internal frames are noise; the line of the user's own code that blew up
    is not.
    """
    te = traceback.TracebackException.from_exception(exc, lookup_lines=True)
    user_frames = [f for f in te.stack if f.filename == filename]

    parts: list[str] = []
    if user_frames:
        parts.append("Traceback (most recent call last):")
        parts.extend(
            line.rstrip("\n") for line in traceback.StackSummary.from_list(user_frames).format()
        )
    message = "".join(te.format_exception_only())
    # jaxplorer already trimmed the stack, so JAX's note about having hidden its own frames
    # would only confuse.
    message = message.split("\n--------------------\nFor simplicity, JAX")[0]
    parts.extend(line.rstrip("\n") for line in message.rstrip("\n").split("\n"))

    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    if cause is not None:
        parts.append("")
        parts.append("caused by:")
        parts.extend(line.rstrip("\n") for line in traceback.format_exception_only(cause))
    return "\n".join(parts)


def _exec_snippet(source: str, filename: str) -> dict[str, Any]:
    """Execute ``source`` in a namespace that is fresh on every request.

    Reusing one would let an edit see state left by the previous version of itself.

    Raises
    ------
    SyntaxError
        If ``source`` does not parse.
    BaseException
        Whatever the snippet raises at module level.
    """
    # Without this, tracebacks cannot show source lines for a buffer that has no file.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    code = compile(source, filename, "exec")
    namespace: dict[str, Any] = {"__name__": "__jaxplorer__", "__file__": filename}
    exec(code, namespace)
    return namespace


def _entry_from(namespace: dict[str, Any]) -> tuple[Any, tuple, dict, dict]:
    """Pull ``f``, ``args``, ``kwargs`` and jit options out of a snippet namespace."""
    if "f" not in namespace:
        raise ContractError(
            "snippet defines no `f`: jaxplorer compiles the callable named `f`.\n"
            "Add e.g.\n"
            "    def f(x):\n"
            "        return jnp.sin(x)"
        )
    f = namespace["f"]
    if not callable(f):
        raise ContractError(f"`f` must be callable, got {type(f).__name__}")

    if "args" not in namespace:
        raise ContractError(
            "snippet defines no `args`: jaxplorer needs example inputs to trace `f`.\n"
            "Add e.g.\n"
            "    args = (jax.ShapeDtypeStruct((8, 16), jnp.float32),)"
        )
    args = namespace["args"]
    # Spelling a lone argument as a 1-tuple is easy to forget.
    if not isinstance(args, tuple):
        args = (args,)

    kwargs = namespace.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ContractError(f"`kwargs` must be a dict, got {type(kwargs).__name__}")

    jit_opts = {name: namespace[name] for name in JIT_OPTION_NAMES if name in namespace}
    return f, args, kwargs, jit_opts


def _jit(f: Any, jit_opts: dict[str, Any]) -> Any:
    """Return a jitted view of ``f``.

    An already-jitted ``f`` is left alone, since re-jitting would drop options jaxplorer never
    saw. Snippet-level jit options still win, because ignoring those would be worse.
    """
    import jax

    already_jitted = hasattr(f, "trace") and hasattr(f, "lower")
    if already_jitted:
        if not jit_opts:
            return f
        inner = getattr(f, "__wrapped__", None)
        if inner is None:
            return f
        f = inner
    return jax.jit(f, **jit_opts)


# e.g. module_0000.jit_f.0005.simplification.after_pipeline-start.before_algsimp.txt
# The `before_` half is absent on the ad-hoc dump points inside copy-insertion.
_PASS_FILE = re.compile(
    r"\.(?P<index>\d{4})\.(?P<pipeline>[^.]+)"
    r"\.after_(?P<after>.+?)(?:\.before_(?P<before>.+))?\.txt$"
)


@contextmanager
def _dump_dir():
    """Yield a scratch directory for one compile's XLA dumps, then delete it.

    Per request rather than per session, because the dumps of a 60 layer model are tens of
    MB and only the newest compile is ever displayed.
    """
    directory = Path(tempfile.mkdtemp(prefix="jaxplorer-dump-"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _dump_options(directory: Path) -> dict[str, Any]:
    """Compiler options that make XLA write its per-pass snapshots.

    Passed to ``compile()`` rather than set in ``XLA_FLAGS``, since the environment is only
    read when the backend boots and this has to be switchable per request.
    """
    return {"xla_dump_to": str(directory), "xla_dump_hlo_pass_re": ".*"}


def _collect_passes(directory: Path) -> list[PassSnapshot]:
    snapshots = []
    for path in sorted(directory.iterdir()):
        match = _PASS_FILE.search(path.name)
        if match is None:
            continue
        snapshots.append(
            PassSnapshot(
                index=int(match["index"]),
                pipeline=match["pipeline"],
                after=match["after"],
                before=match["before"] or "",
                text=path.read_text(errors="replace"),
            )
        )
    snapshots.sort(key=lambda snapshot: snapshot.index)
    return snapshots


def _collect_llvm_ir(directory: Path) -> str | None:
    """The optimized LLVM IR the CPU backend emitted, if this backend emits any."""
    for suffix in ("ir-with-opt.ll", "ir-no-opt.ll"):
        found = sorted(p for p in directory.iterdir() if p.name.endswith(suffix))
        if found:
            return "\n\n".join(
                f"; ==== {path.name} ====\n{path.read_text(errors='replace')}" for path in found
            )
    return None


def _render_analysis(compiled: Any) -> str:
    """Render cost and memory analysis as aligned tables.

    Both are backend-dependent, so each is reported on its own and a failure in one does not
    hide the other.
    """
    sections: list[str] = []

    try:
        cost = compiled.cost_analysis()
    except Exception as exc:  # noqa: BLE001 - backend-dependent, never fatal
        sections.append(f"cost analysis unavailable: {exc}")
    else:
        sections.append(_table("Cost", _as_mapping(cost)))

    try:
        memory = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001 - backend-dependent, never fatal
        sections.append(f"memory analysis unavailable: {exc}")
    else:
        sections.append(_table("Memory", _as_mapping(memory)))

    return "\n\n".join(s for s in sections if s)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce an analysis result to a flat mapping.

    ``cost_analysis`` returns a dict, or a list of them for a multi-executable program;
    ``memory_analysis`` returns a ``CompiledMemoryStats`` object.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        merged: dict[str, Any] = {}
        for item in value:
            merged.update(_as_mapping(item))
        return merged
    fields = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        attr = getattr(value, name, None)
        # The buffer assignment proto alone is kilobytes of bytes in a table meant to be
        # read at a glance.
        if callable(attr) or isinstance(attr, (bytes, bytearray, memoryview)):
            continue
        fields[name] = attr
    return fields


def _table(title: str, mapping: dict[str, Any]) -> str:
    if not mapping:
        return f"{title}: (empty)"
    width = max(len(str(k)) for k in mapping)
    lines = [f"{title}:"]
    lines.extend(
        f"  {str(key).ljust(width)}  {_number(mapping[key])}" for key in sorted(mapping, key=str)
    )
    return "\n".join(lines)


def _number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def handle(request: CompileRequest) -> CompileResult:
    """Run one compile request end to end.

    Parameters
    ----------
    request : CompileRequest
        What to compile, and how much of it to report.

    Returns
    -------
    CompileResult
        Never raises: a snippet that fails to run comes back with ``fatal`` set, and a
        stage that fails carries its own error while earlier stages keep their output.
    """
    with _dump_dir() as dumps:
        return _handle(request, dumps if request.passes else None)


def _handle(request: CompileRequest, dumps: Path | None) -> CompileResult:
    started = time.perf_counter()
    result = CompileResult(id=request.id)
    wanted = [s for s in ALL_STAGES if s in request.stages]

    try:
        namespace = _exec_snippet(request.source, request.filename)
        f, args, kwargs, jit_opts = _entry_from(namespace)
    except ContractError as exc:
        result.fatal = str(exc)
        result.total_ms = (time.perf_counter() - started) * 1000
        return result
    except BaseException as exc:  # noqa: BLE001 - a snippet may raise anything
        result.fatal = _format_error(exc, request.filename)
        result.total_ms = (time.perf_counter() - started) * 1000
        return result

    # Each stage consumes the previous one's output, so the chain always runs in order and
    # `wanted` only decides what is reported. A failure keeps earlier results: a valid jaxpr
    # beside a lowering error is the signal the user is after.
    def do_jaxpr(_prev: Any) -> tuple[Any, str]:
        traced = _jit(f, jit_opts).trace(*args, **kwargs)
        return traced, str(traced.jaxpr)

    def do_stablehlo(traced: Any) -> tuple[Any, str]:
        lowered = traced.lower()
        return lowered, lowered.as_text()

    def do_optimized_hlo(lowered: Any) -> tuple[Any, str]:
        options = _dump_options(dumps) if dumps is not None else None
        compiled = lowered.compile(compiler_options=options) if options else lowered.compile()
        if dumps is not None:
            result.passes = _collect_passes(dumps)
            result.llvm_ir = _collect_llvm_ir(dumps)
        text = compiled.as_text()
        if text is None:
            text = "(this backend does not expose an optimized HLO module)"
        return compiled, text

    def do_analysis(compiled: Any) -> tuple[Any, str]:
        return compiled, _render_analysis(compiled)

    chain: list[tuple[Stage, Any]] = [
        ("jaxpr", do_jaxpr),
        ("stablehlo", do_stablehlo),
        ("optimized_hlo", do_optimized_hlo),
        ("analysis", do_analysis),
    ]

    state: Any = None
    broken = False
    for stage, run in chain:
        if broken:
            if stage in wanted:
                result.stages[stage] = StageResult(skipped=True)
            continue
        begin = time.perf_counter()
        try:
            state, text = run(state)
        except BaseException as exc:  # noqa: BLE001 - tracing can raise anything
            broken = True
            # Reported even when unasked-for, or a subset request would show unexplained
            # skips downstream.
            result.stages[stage] = StageResult(
                error=_format_error(exc, request.filename),
                elapsed_ms=(time.perf_counter() - begin) * 1000,
            )
            continue
        if stage in wanted:
            result.stages[stage] = StageResult(
                text=text, elapsed_ms=(time.perf_counter() - begin) * 1000
            )

    result.total_ms = (time.perf_counter() - started) * 1000
    return result


# Below this, jax emits no HLO stack-frame tables and click-to-source silently finds
# nothing. Checked here rather than in the UI because only the worker imports jax, and with
# --python that jax is not the one jaxplorer was installed with.
MIN_JAX = (0, 9)


def _version_warning(version: str) -> str | None:
    try:
        parts = tuple(int(p) for p in version.split(".")[:2])
    except ValueError:
        return None
    if parts >= MIN_JAX:
        return None
    floor = ".".join(str(p) for p in MIN_JAX)
    return (
        f"jax {version} is older than the oldest fully supported version {floor}: "
        "click-to-source will not work, since this version emits no HLO stack-frame tables."
    )


def _ready_message() -> str:
    import jax

    return json.dumps(
        {
            "ready": True,
            "jax_version": jax.__version__,
            "platform": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "warning": _version_warning(jax.__version__),
        }
    )


def main() -> int:
    """Serve compile requests off stdin until it closes.

    Returns
    -------
    int
        ``0`` on a clean shutdown, ``1`` if the backend never came up.
    """
    with _protocol_channel() as channel:
        # Only once the backend is up, so a broken environment reads as a startup
        # failure rather than a mystery hang.
        try:
            ready = _ready_message()
        except ModuleNotFoundError as exc:
            # The likeliest first run failure now that jax is an extra, so name the
            # interpreter that lacks it and both ways out.
            missing = (
                f"{exc.name} is not installed in {sys.executable}.\n\n"
                "Point --python at an environment that has jax, run jaxplorer from a "
                "virtual env that has it, or give jaxplorer its own copy with "
                "`uv tool install jaxplorer[jax]`."
                if exc.name in ("jax", "jaxlib")
                else str(exc)
            )
            channel.write(encode_frame(json.dumps({"ready": False, "error": missing})))
            return 1
        except Exception as exc:  # noqa: BLE001 - report and give up cleanly
            channel.write(encode_frame(json.dumps({"ready": False, "error": str(exc)})))
            return 1
        channel.write(encode_frame(ready))

        while True:
            line = sys.stdin.readline()
            if not line:  # the TUI closed our stdin
                break
            if not line.strip():
                continue
            try:
                request = CompileRequest.from_dict(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - malformed frame, keep serving
                channel.write(encode_frame(json.dumps({"id": -1, "fatal": f"bad request: {exc}"})))
                continue
            channel.write(encode_frame(handle(request).to_json()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
