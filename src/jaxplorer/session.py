"""Ownership of the compile worker subprocess.

:class:`WorkerSession` hides three realities from the UI: the worker takes seconds to boot,
XLA aborts the process outright on some inputs, and the user types faster than JAX compiles.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from jaxplorer.protocol import CompileRequest, CompileResult, Stage

# Bootstrap for the worker. `append`, never `insert`, so that a foreign interpreter resolves
# jax and everything else from its own environment and only falls back to ours for jaxplorer
# itself.
_BOOTSTRAP = (
    "import sys; sys.path.append({path!r});"
    " from jaxplorer.worker import main; raise SystemExit(main())"
)

DEFAULT_TIMEOUT = 20.0
WORKER_INTERPRETER = "python.exe" if sys.platform == "win32" else "python"
# Booting JAX and warming a backend is far slower than any single compile.
STARTUP_TIMEOUT = 90.0
STDERR_TAIL_LINES = 40
# Only stderr is read by line; a single XLA log line can still be long.
STREAM_LIMIT = 4 * 1024 * 1024


def resolve_interpreter(explicit: str | None = None) -> str:
    """Pick the interpreter the worker should run under.

    ``--python`` wins, then an active virtualenv, then the interpreter running jaxplorer.
    The middle rule is what makes a tool install useful: `uv run` and a plain `activate`
    both export ``VIRTUAL_ENV``, so a jaxplorer installed with `uv tool install` inspects
    the project you are standing in rather than the jax it shipped with. When jaxplorer is
    itself installed in that virtualenv the two agree, and this changes nothing.

    Parameters
    ----------
    explicit : str, optional
        Value of ``--python``, used as-is when given.

    Returns
    -------
    str
        Path to a Python interpreter.
    """
    if explicit:
        return explicit
    active = os.environ.get("VIRTUAL_ENV")
    if active:
        candidate = Path(active) / ("Scripts" if sys.platform == "win32" else "bin")
        candidate /= WORKER_INTERPRETER
        # Ignore a stale or half-built VIRTUAL_ENV rather than failing to start at all.
        if candidate.is_file():
            return str(candidate)
    return sys.executable


class ProtocolError(RuntimeError):
    """The worker sent something that is not a length-prefixed frame."""


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """What a worker reported about itself at startup.

    Attributes
    ----------
    jax_version : str
        The worker's ``jax.__version__``.
    platform : str
        Backend JAX actually chose, which need not be the one that was requested.
    devices : tuple of str
        Devices that backend exposes.
    warning : str or None
        Something about the environment worth surfacing, e.g. a jax too old for
        click-to-source.
    """

    jax_version: str
    platform: str
    devices: tuple[str, ...]
    warning: str | None = None

    def summary(self) -> str:
        """Return a one-line description for the status bar."""
        return f"{self.platform} · jax {self.jax_version}"


class WorkerStartupError(RuntimeError):
    """The worker process could not be brought up, e.g. no such backend."""


class WorkerSession:
    """Owns one compile worker, respawning it whenever it dies or has to change.

    Parameters
    ----------
    platform : str, optional
        Value for ``JAX_PLATFORMS``. ``None`` leaves the choice to JAX.
    x64 : bool, optional
        Whether to set ``JAX_ENABLE_X64``.
    timeout : float, optional
        Seconds to wait for one compile before killing the worker.
    executable : str, optional
        Interpreter to run the worker under. Defaults to :func:`resolve_interpreter`, which
        prefers an active virtualenv over jaxplorer's own environment.

    Attributes
    ----------
    info : WorkerInfo or None
        The running worker's handshake, or ``None`` while no worker is up.
    """

    def __init__(
        self,
        *,
        platform: str | None = None,
        x64: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        executable: str | None = None,
    ) -> None:
        self.platform = platform
        self.x64 = x64
        self.timeout = timeout
        self.executable = resolve_interpreter(executable)
        self.info: WorkerInfo | None = None

        self._process: asyncio.subprocess.Process | None = None
        self._stderr: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._stderr_task: asyncio.Task[None] | None = None
        # One request in flight at a time: the worker is serial anyway, and the lock is
        # what lets queued requests notice they have been superseded.
        self._lock = asyncio.Lock()
        self._seq = 0
        self._latest = 0

    # -- process lifecycle ------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["JAX_ENABLE_X64"] = "1" if self.x64 else "0"
        if self.platform:
            env["JAX_PLATFORMS"] = self.platform
        else:
            env.pop("JAX_PLATFORMS", None)
        return env

    async def start(self) -> WorkerInfo:
        """Spawn the worker if needed and wait for its readiness handshake.

        Returns
        -------
        WorkerInfo
            The running worker's handshake, reused when one is already up.

        Raises
        ------
        WorkerStartupError
            If the worker exits, hangs, or reports that it is not ready.
        """
        process = self._process
        if process is not None:
            if process.returncode is None and self.info is not None:
                return self.info
            # Either dead, or half-started because a previous start() was cancelled
            # mid-handshake. Neither can serve a request.
            await self._kill()

        self._stderr.clear()
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "-c",
            _BOOTSTRAP.format(path=str(Path(__file__).parent.parent)),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env(),
            limit=STREAM_LIMIT,
        )
        self._process = process
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))

        try:
            frame = await self._read_frame(process, STARTUP_TIMEOUT)
        except TimeoutError as exc:
            await self._kill()
            raise WorkerStartupError(
                f"worker did not become ready within {STARTUP_TIMEOUT:.0f}s"
            ) from exc
        except ProtocolError as exc:
            await self._kill()
            raise WorkerStartupError(
                f"unreadable worker handshake: {exc}\n{self.stderr_tail()}"
            ) from exc
        except asyncio.CancelledError:
            # Leaving a half-started worker behind would wedge every later request.
            await self._kill()
            raise

        if frame is None:
            await self._kill()
            raise WorkerStartupError("worker exited during startup:\n" + self.stderr_tail())

        try:
            handshake = json.loads(frame)
        except json.JSONDecodeError as exc:
            await self._kill()
            raise WorkerStartupError(
                f"unreadable worker handshake {frame[:200]!r}:\n{self.stderr_tail()}"
            ) from exc

        if not handshake.get("ready"):
            await self._kill()
            raise WorkerStartupError(handshake.get("error") or "worker reported it is not ready")

        self.info = WorkerInfo(
            jax_version=str(handshake.get("jax_version", "?")),
            platform=str(handshake.get("platform", "?")),
            devices=tuple(handshake.get("devices") or ()),
            warning=handshake.get("warning"),
        )
        return self.info

    @staticmethod
    async def _read_frame(process: asyncio.subprocess.Process, timeout: float) -> bytes | None:
        """Read one length-prefixed frame, or ``None`` at end of stream.

        Raises
        ------
        ProtocolError
            If the length header is unreadable or the frame is truncated.
        TimeoutError
            If nothing arrives within ``timeout``.
        """
        stream = process.stdout
        assert stream is not None  # always a pipe, see start()
        header = await asyncio.wait_for(stream.readline(), timeout=timeout)
        if not header:
            return None
        try:
            size = int(header)
        except ValueError as exc:
            raise ProtocolError(f"expected a frame length, got {header[:80]!r}") from exc
        try:
            return await asyncio.wait_for(stream.readexactly(size), timeout=timeout)
        except asyncio.IncompleteReadError as exc:
            raise ProtocolError(f"frame truncated at {len(exc.partial)} of {size} bytes") from exc

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """Keep the tail of the worker's stderr for crash diagnostics.

        fd 1 is redirected to stderr in the worker, so this also picks up snippet
        ``print`` output and XLA's logging, which is what is worth showing on a crash.
        """
        stream = process.stderr
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # A line longer than STREAM_LIMIT. Dropping it is fine; giving up on the
                # rest of stderr is not.
                self._stderr.append("<jaxplorer: overlong stderr line dropped>")
                continue
            except asyncio.CancelledError:
                return
            if not line:
                return
            self._stderr.append(line.decode(errors="replace").rstrip("\n"))

    def stderr_tail(self) -> str:
        """Return the worker's most recent stderr lines, for crash diagnostics."""
        return "\n".join(self._stderr)

    async def _kill(self) -> None:
        process, self._process = self._process, None
        self.info = None
        task, self._stderr_task = self._stderr_task, None
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            # A killed process that will not be reaped is not worth blocking the UI for.
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5.0)
        if task is not None:
            task.cancel()

    async def restart(self) -> WorkerInfo:
        """Replace the worker, which is how a platform or x64 change takes effect.

        Returns
        -------
        WorkerInfo
            The new worker's handshake.

        Raises
        ------
        WorkerStartupError
            If the new worker cannot be brought up.
        """
        await self._kill()
        return await self.start()

    async def close(self) -> None:
        """Shut the worker down, closing its stdin first so it can exit on its own."""
        process = self._process
        if process is not None and process.returncode is None and process.stdin:
            with suppress(BrokenPipeError, ConnectionResetError):
                process.stdin.close()
        await self._kill()

    # -- requests ---------------------------------------------------------------

    async def compile(
        self,
        source: str,
        filename: str = "<buffer>",
        stages: list[Stage] | None = None,
        passes: bool = False,
    ) -> CompileResult | None:
        """Compile ``source``, or return ``None`` if a newer request superseded it.

        Requests queue behind each other, and when the queue drains everything but the
        newest is dropped, so a burst of keystrokes costs one compile rather than one per
        stroke.

        Parameters
        ----------
        source : str
            Snippet source.
        filename : str, optional
            Name to compile it under, which tracebacks and HLO debug tables report.
        stages : list of Stage, optional
            Stages to report back. Defaults to all of them.
        passes : bool, optional
            Whether to collect per-pass HLO snapshots and LLVM IR.

        Returns
        -------
        CompileResult or None
            ``None`` when a newer request arrived first. Failures come back as a result
            with ``fatal`` set rather than as an exception, so the UI has something to
            show either way.
        """
        self._seq += 1
        request_id = self._seq
        self._latest = request_id

        async with self._lock:
            if request_id != self._latest:
                return None
            try:
                await self.start()
            except WorkerStartupError as exc:
                return CompileResult(id=request_id, fatal=str(exc))
            request = CompileRequest(
                id=request_id,
                source=source,
                filename=filename,
                passes=passes,
                **({"stages": list(stages)} if stages else {}),
            )
            return await self._roundtrip(request)

    async def _roundtrip(self, request: CompileRequest) -> CompileResult:
        process = self._process
        assert process is not None and process.stdin is not None

        try:
            process.stdin.write((request.to_json() + "\n").encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            await self._kill()
            return CompileResult(
                id=request.id,
                fatal="worker died before it could read the request:\n" + self.stderr_tail(),
            )

        while True:
            try:
                frame = await self._read_frame(process, self.timeout)
            except TimeoutError:
                await self._kill()
                return CompileResult(
                    id=request.id,
                    fatal=(
                        f"timed out after {self.timeout:.0f}s and the worker was "
                        "restarted.\nAn infinite loop in the snippet, or a compile "
                        "this slow, needs a larger --timeout."
                    ),
                )
            except ProtocolError as exc:
                await self._kill()
                return CompileResult(
                    id=request.id,
                    fatal=f"lost sync with the worker: {exc}\n{self.stderr_tail()}",
                )
            if frame is None:
                await self._kill()
                return CompileResult(
                    id=request.id,
                    fatal="worker crashed while compiling:\n" + self.stderr_tail(),
                )
            try:
                payload = json.loads(frame)
            except json.JSONDecodeError as exc:
                await self._kill()
                return CompileResult(id=request.id, fatal=f"unreadable worker response: {exc}")
            # A response to a request that was cancelled while in flight.
            if payload.get("id") != request.id:
                continue
            return CompileResult.from_dict(payload)
