"""Wire protocol shared by the TUI and the compile worker.

Requests are newline-delimited JSON on the worker's stdin. Responses are length-prefixed
instead, because the HLO of a real model runs to megabytes and asyncio's ``readline``
refuses a line over 64 KiB.

Imports neither jax nor textual, so both sides can load it cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Self, get_args

Stage = Literal["jaxpr", "stablehlo", "optimized_hlo", "analysis"]
"""One step of the lowering chain, in pipeline order."""

ALL_STAGES: tuple[Stage, ...] = get_args(Stage)

STAGE_TITLES: dict[Stage, str] = {
    "jaxpr": "Jaxpr",
    "stablehlo": "StableHLO",
    "optimized_hlo": "Optimized HLO",
    "analysis": "Analysis",
}

# Kept distinct from Stage so that neither the worker nor STAGE_TITLES has to know about
# tabs that are not lowering stages.
Extra = Literal["passes", "llvm_ir", "errors"]
Pane = Stage | Extra
"""Anything the UI gives a tab to, whether or not the worker produces it."""

PANE_TITLES: dict[Pane, str] = {
    **STAGE_TITLES,
    "passes": "Passes",
    "llvm_ir": "LLVM IR",
    "errors": "Errors",
}


def encode_frame(payload: str) -> bytes:
    """Frame one response for the wire.

    Parameters
    ----------
    payload : str
        JSON text to send.

    Returns
    -------
    bytes
        The payload's length in decimal, a newline, then the payload's UTF-8 bytes.
    """
    data = payload.encode()
    return b"%d\n%s" % (len(data), data)


@dataclass(slots=True)
class PassSnapshot:
    """One HLO module as it stood between two XLA passes.

    Attributes
    ----------
    index : int
        Position in the order XLA ran its passes.
    pipeline : str
        Pass pipeline this snapshot was taken in, e.g. ``simplification``.
    after : str
        Pass that had just run, or ``pipeline-start`` at a pipeline boundary.
    before : str
        Pass that was about to run. Empty for the ad-hoc dump points inside
        ``copy-insertion``, which record no successor.
    text : str
        The whole module text.
    """

    index: int
    pipeline: str
    after: str
    before: str
    text: str

    @property
    def label(self) -> str:
        """One-line identification for a list or a diff heading."""
        return f"{self.index:04d}  {self.pipeline} · {self.after}"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible mapping for this snapshot."""
        return {
            "index": self.index,
            "pipeline": self.pipeline,
            "after": self.after,
            "before": self.before,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a snapshot from :meth:`to_dict` output.

        Parameters
        ----------
        data : dict
            Mapping as produced by :meth:`to_dict`.

        Returns
        -------
        PassSnapshot
        """
        return cls(
            index=int(data["index"]),
            pipeline=str(data["pipeline"]),
            after=str(data["after"]),
            before=str(data["before"]),
            text=str(data["text"]),
        )


@dataclass(slots=True)
class CompileRequest:
    """One snippet to compile.

    Attributes
    ----------
    id : int
        Monotonic request id. The TUI drops responses that are not the newest.
    source : str
        Snippet source, executed as a module.
    filename : str
        Name to compile ``source`` under. Tracebacks and HLO debug tables report it, so
        it is the buffer's path when there is one.
    stages : list of Stage
        Stages to report. The chain always runs in order regardless, since each stage
        consumes what the previous one produced.
    passes : bool
        Whether to also collect per-pass HLO snapshots and LLVM IR.
    """

    id: int
    source: str
    filename: str = "<buffer>"
    stages: list[Stage] = field(default_factory=lambda: list(ALL_STAGES))
    passes: bool = False

    def to_json(self) -> str:
        """Serialize this request to a single JSON line."""
        return json.dumps(
            {
                "id": self.id,
                "source": self.source,
                "filename": self.filename,
                "stages": list(self.stages),
                "passes": self.passes,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a request from decoded JSON.

        Parameters
        ----------
        data : dict
            Decoded :meth:`to_json` output. Only ``id`` and ``source`` are required.

        Returns
        -------
        CompileRequest
        """
        return cls(
            id=int(data["id"]),
            source=str(data["source"]),
            filename=str(data.get("filename", "<buffer>")),
            stages=list(data["stages"]) if "stages" in data else list(ALL_STAGES),
            passes=bool(data.get("passes", False)),
        )


@dataclass(slots=True)
class StageResult:
    """Outcome of a single stage.

    Attributes
    ----------
    text : str or None
        What the stage produced.
    error : str or None
        Why it failed, already trimmed to the user's own frames.
    skipped : bool
        True when the stage never ran because an earlier one failed, so the UI can stay
        quiet rather than repeat the upstream traceback.
    elapsed_ms : float
        Wall time for this stage alone.
    """

    text: str | None = None
    error: str | None = None
    skipped: bool = False
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether this stage ran and produced something."""
        return self.error is None and not self.skipped


@dataclass(slots=True)
class CompileResult:
    """Everything one request produced.

    Attributes
    ----------
    id : int
        The id of the request this answers.
    stages : dict of Stage to StageResult
        Per-stage outcome. Stages carry their own errors so that a lowering failure still
        leaves a valid jaxpr to read.
    fatal : str or None
        Why nothing ran at all: a syntax error, or a snippet that does not meet the
        contract.
    total_ms : float
        Wall time for the whole request.
    passes : list of PassSnapshot
        Per-pass HLO snapshots, empty unless the request asked for them.
    llvm_ir : str or None
        LLVM IR from backends that emit it, absent unless the request asked for it.
    stages_run : list of Stage
        Stages that actually executed, which is not the same as the keys of ``stages``: the
        chain stops after the last stage anyone asked for, so requesting only ``jaxpr``
        genuinely skips XLA. Reported separately because ``stages`` holds what was *asked
        for*, and the difference is the only way to see the work that was avoided.
    """

    id: int
    stages: dict[Stage, StageResult] = field(default_factory=dict)
    fatal: str | None = None
    total_ms: float = 0.0
    passes: list[PassSnapshot] = field(default_factory=list)
    llvm_ir: str | None = None
    stages_run: list[Stage] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize this result for :func:`encode_frame`."""
        return json.dumps(
            {
                "id": self.id,
                "fatal": self.fatal,
                "total_ms": self.total_ms,
                "passes": [p.to_dict() for p in self.passes],
                "llvm_ir": self.llvm_ir,
                "stages_run": list(self.stages_run),
                "stages": {
                    name: {
                        "text": r.text,
                        "error": r.error,
                        "skipped": r.skipped,
                        "elapsed_ms": r.elapsed_ms,
                    }
                    for name, r in self.stages.items()
                },
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a result from decoded JSON.

        Parameters
        ----------
        data : dict
            Decoded :meth:`to_json` output.

        Returns
        -------
        CompileResult
        """
        return cls(
            id=int(data["id"]),
            fatal=data.get("fatal"),
            total_ms=float(data.get("total_ms", 0.0)),
            passes=[PassSnapshot.from_dict(p) for p in (data.get("passes") or [])],
            llvm_ir=data.get("llvm_ir"),
            stages_run=list(data.get("stages_run") or []),
            stages={
                name: StageResult(
                    text=r.get("text"),
                    error=r.get("error"),
                    skipped=bool(r.get("skipped", False)),
                    elapsed_ms=float(r.get("elapsed_ms", 0.0)),
                )
                for name, r in (data.get("stages") or {}).items()
            },
        )

    def errors(self) -> list[str]:
        """Return every failure, the fatal one first, then stages in pipeline order.

        Returns
        -------
        list of str
            Stage failures are prefixed with the stage's title.
        """
        out = []
        if self.fatal:
            out.append(self.fatal)
        for name in ALL_STAGES:
            result = self.stages.get(name)
            if result is not None and result.error:
                out.append(f"[{STAGE_TITLES[name]}]\n{result.error}")
        return out
