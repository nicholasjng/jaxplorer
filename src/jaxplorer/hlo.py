"""HLO text utilities: hiding debug tables, and mapping instructions back to source.

XLA appends the Python stack that built each instruction to the module text, as four lookup
tables plus a ``stack_frame_id`` in each instruction's metadata. Those tables are noise when
reading IR and gold when locating the line that produced an instruction, which is why both
jobs live here.

Pure text handling, so neither jax nor textual is needed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jaxplorer.hlodiff import structural_pass_report

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaxplorer.protocol import PassSnapshot

# Heading of each debug table XLA appends to a module.
TABLE_HEADS = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")

_NUMBERED = re.compile(r"^(\d+)\s*(.*)$")
_QUOTED = re.compile(r'"(.*)"')
_FIELD = re.compile(r"(\w+)=(\d+)")
_STACK_FRAME_ID = re.compile(r"\bstack_frame_id=(\d+)")
# Cheap guard against a cycle in parent_frame_id, which the dump does contain.
_MAX_FRAME_WALK = 64


def strip_debug_tables(text: str) -> str:
    """Drop the debug tables from a module's text.

    They routinely push ``ENTRY main`` a full screen down and name jaxplorer's own worker, hence
    the default view hides them.

    Parameters
    ----------
    text : str
        An HLO module, with or without tables.

    Returns
    -------
    str
        The module with every table in :data:`TABLE_HEADS` removed and nothing else
        touched.
    """
    out: list[str] = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        if lines[index].strip() in TABLE_HEADS:
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1  # and the blank line that separated it
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out).strip("\n") + "\n"


def _attribute(before: PassSnapshot, after: PassSnapshot) -> str:
    """Name what changed the module between two snapshots, without overclaiming.

    A snapshot records the state *after* some pass, so a difference is the work of whatever
    ran between the two dump points: exactly one named pass when the later snapshot names
    it, otherwise an unnamed span of a pipeline.
    """
    if after.after != "pipeline-start":
        return f"{after.after}  ({after.pipeline})"
    return f"entering {after.pipeline}, after {before.after} in {before.pipeline}"


def pass_report(
    snapshots: Sequence[PassSnapshot], *, context: int = 2, structural: bool = False
) -> str:
    """Summarize what each XLA pass did to the module.

    Most passes change nothing, so the diffs lead and the pipeline order follows as an
    index: finding the pass that mattered is the whole point. Debug tables are excluded
    from the comparison, since their churn is not a transformation.

    Parameters
    ----------
    snapshots : sequence of PassSnapshot
        Snapshots in the order XLA produced them.
    context : int, optional
        Lines of context around each diff hunk.
    structural : bool, optional
        Compare the modules as graphs rather than as text, via :mod:`jaxplorer.hlodiff`. A
        rescheduling pass reports nothing under this mode and a whole-module renaming
        reports a handful of instructions, neither of which a line diff can manage. Falls
        back to the text diff, with a note, when the module text does not parse cleanly.

    Returns
    -------
    str
        A summary line, one diff per transition that changed the module, then every
        snapshot's label with a ``*`` against the ones that changed something.
    """
    if not snapshots:
        return (
            "No pass snapshots.\n\nRun jaxplorer with --passes, or press f6, to collect them.\n"
            "Once collected, f4 switches this pane between a text diff and a structural one, "
            "and ] / [ step between sections."
        )

    note = ""
    if structural:
        report = structural_pass_report(snapshots, attribute=_attribute)
        if report is not None:
            return report
        note = (
            "Structural diff unavailable: the module text did not parse cleanly enough to "
            "trust it.\nShowing a text diff instead.\n\n"
        )

    diffs: list[str] = []
    changed: set[int] = set()
    for before, after in zip(snapshots, snapshots[1:], strict=False):
        old = strip_debug_tables(before.text).split("\n")
        new = strip_debug_tables(after.text).split("\n")
        if old == new:
            continue
        changed.add(before.index)
        delta = difflib.unified_diff(old, new, "before", "after", lineterm="", n=context)
        diffs.append(f"===== {_attribute(before, after)} =====\n" + "\n".join(delta))

    head = f"{len(snapshots)} snapshots, {len(changed)} changed the module."
    body = "\n\n".join(diffs) if diffs else "No pass changed the module."
    index = "\n".join(
        f"  {'*' if snapshot.index in changed else ' '} {snapshot.label}" for snapshot in snapshots
    )
    return f"{note}{head}\n\n{body}\n\n===== pipeline order =====\n{index}\n"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where an HLO instruction came from.

    Attributes
    ----------
    file : str
        File as XLA recorded it, which is the name the snippet was compiled under.
    line : int
        1-based line number.
    function : str
        Enclosing function's name.
    """

    file: str
    line: int
    function: str

    def __str__(self) -> str:
        """Render as ``file:line in function``."""
        return f"{self.file}:{self.line} in {self.function}"


@dataclass(slots=True)
class DebugInfo:
    """The four debug tables, parsed, keyed by the ids the module refers to them with.

    Attributes
    ----------
    file_names : dict of int to str
    function_names : dict of int to str
    file_locations : dict of int to dict
        Each value holds ``file_name_id``, ``function_name_id``, ``line`` and the column
        fields XLA recorded.
    stack_frames : dict of int to dict
        Each value holds ``file_location_id`` and ``parent_frame_id``.
    """

    file_names: dict[int, str] = field(default_factory=dict)
    function_names: dict[int, str] = field(default_factory=dict)
    file_locations: dict[int, dict[str, int]] = field(default_factory=dict)
    stack_frames: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        """Whether the module carried no usable debug info."""
        return not self.stack_frames

    @classmethod
    def parse(cls, text: str) -> DebugInfo:
        """Read the debug tables out of a module.

        Parameters
        ----------
        text : str
            An HLO module. Missing tables yield empty mappings rather than an error, since
            not every backend or jax version emits them.

        Returns
        -------
        DebugInfo
        """
        info = cls()
        table: str | None = None
        for raw in text.split("\n"):
            line = raw.strip()
            if line in TABLE_HEADS:
                table = line
                continue
            if not line:
                table = None
                continue
            if table is None:
                continue
            match = _NUMBERED.match(line)
            if match is None:
                continue
            key, rest = int(match.group(1)), match.group(2)
            if table == "FileNames":
                quoted = _QUOTED.search(rest)
                if quoted:
                    info.file_names[key] = quoted.group(1)
            elif table == "FunctionNames":
                quoted = _QUOTED.search(rest)
                if quoted:
                    info.function_names[key] = quoted.group(1)
            elif table == "FileLocations":
                info.file_locations[key] = {k: int(v) for k, v in _FIELD.findall(rest)}
            elif table == "StackFrames":
                info.stack_frames[key] = {k: int(v) for k, v in _FIELD.findall(rest)}
        return info

    def _ref(self, frame_id: int) -> SourceRef | None:
        frame = self.stack_frames.get(frame_id)
        if frame is None:
            return None
        location = self.file_locations.get(frame.get("file_location_id", -1))
        if location is None:
            return None
        file = self.file_names.get(location.get("file_name_id", -1))
        if file is None:
            return None
        return SourceRef(
            file=file,
            line=location.get("line", 0),
            function=self.function_names.get(location.get("function_name_id", -1), "?"),
        )

    def locate(self, hlo_line: str, *, prefer: str | None = None) -> SourceRef | None:
        """Resolve one line of HLO to the source that produced it.

        Parameters
        ----------
        hlo_line : str
            A single line of module text. Lines without a ``stack_frame_id`` resolve to
            ``None``.
        prefer : str, optional
            File to hold out for. The walk follows ``parent_frame_id`` outwards looking for
            it, so an instruction whose innermost frame sits in jaxplorer or JAX still reports
            the snippet line that led there. Without a match the result is ``None``, since
            a location inside JAX would only mislead.

        Returns
        -------
        SourceRef or None
        """
        match = _STACK_FRAME_ID.search(hlo_line)
        if match is None:
            return None
        frame_id = int(match.group(1))

        first: SourceRef | None = None
        seen: set[int] = set()
        for _ in range(_MAX_FRAME_WALK):
            if frame_id in seen:
                break
            seen.add(frame_id)
            ref = self._ref(frame_id)
            if ref is None:
                break
            if first is None:
                first = ref
            if prefer is None or ref.file == prefer:
                return ref
            parent = self.stack_frames.get(frame_id, {}).get("parent_frame_id")
            if parent is None or parent == frame_id:
                break
            frame_id = parent
        # No frame in the snippet: the innermost frame is still better than nothing.
        return first if prefer is None else None
