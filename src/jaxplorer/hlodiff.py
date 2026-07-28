"""Compare two HLO modules by graph structure instead of by line.

A text diff of two pass snapshots answers "what changed in the printout", which is not the
question. Scheduling reorders instructions without touching the DAG; a rewrite renames a
computation and every instruction in it. Both read as large text diffs and neither is a large
change. Matching the graphs first and reporting on the correspondence answers "what changed in
the program".

This is a subset of what ``xla/hlo/tools/hlo_diff`` does, reimplemented over parsed text
because that tool is an ``xla_cc_binary`` shipped in no jaxlib wheel. If you have an XLA
checkout, prefer the real thing — it has the actual cost model and match provenance. This
exists for everyone else, and it is deliberately conservative: when the parse looks unreliable
:func:`structural_pass_report` declines rather than guessing, and the caller shows a text diff.

Pure text handling, so neither jax nor textual is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from jaxplorer.hlograph import Computation, Instruction, Module, Options, fingerprint, parse_module

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxplorer.hlograph import Fingerprints
    from jaxplorer.protocol import PassSnapshot

Change = Literal["opcode", "shape", "layout", "operands", "attributes", "called", "root"]

# Same-opcode, same-shape instructions are matched pairwise, which is quadratic. Past this
# many candidates in one bucket the pairing is done positionally and the result is flagged
# truncated, so a pathological module degrades in quality rather than in responsiveness.
MAX_BUCKET = 256
MAX_PAIRS = 20_000

# Below this Dice similarity two computations are unrelated rather than changed.
MIN_SIMILARITY = 0.4

# Below this score, two instructions from different (opcode, shape, arity) buckets are two
# different instructions rather than one changed one. Scores run to about 8.5, and a shared
# name plus shared operands alone clears it comfortably.
MIN_SCORE = 3.0

# A module whose text parses this badly is not worth diffing structurally.
MAX_UNPARSED_FRACTION = 0.02

# Per computation, in one transition. Long enough to read, short enough to scroll past.
MAX_DETAIL_LINES = 40


@dataclass(frozen=True, slots=True)
class InstructionRef:
    """Where an instruction was, and how it read.

    Attributes
    ----------
    computation : str
        Name of the computation it belonged to.
    name : str
        Instruction name.
    opcode : str
        Opcode, so a summary can group by it without re-parsing.
    line : int
        1-based line in the module text.
    text : str
        The original line, verbatim. Rendering diffs from this keeps click-to-source working.
    """

    computation: str
    name: str
    opcode: str
    line: int
    text: str

    @classmethod
    def of(cls, computation: str, instruction: Instruction) -> InstructionRef:
        """Build a reference to ``instruction`` inside ``computation``."""
        return cls(
            computation=computation,
            name=instruction.name,
            opcode=instruction.opcode,
            line=instruction.line,
            text=instruction.text,
        )


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """Two instructions the matcher believes are the same instruction.

    Attributes
    ----------
    left, right : InstructionRef
    changes : tuple of str
        Which fields differ. Empty means the instruction survived unchanged.
    renamed : bool
        Whether the name changed. Tracked separately from ``changes`` because renumbering is
        noise, but silent noise is how you fail to notice a matcher over-matching.
    """

    left: InstructionRef
    right: InstructionRef
    changes: tuple[Change, ...]
    renamed: bool

    @property
    def changed(self) -> bool:
        """Whether any field differs."""
        return bool(self.changes)


@dataclass(frozen=True, slots=True)
class ComputationDiff:
    """The correspondence between two computations.

    A ``None`` on either side means the computation only exists on the other, so the whole
    thing was added or removed.

    Attributes
    ----------
    left, right : str or None
        Computation names.
    pairs : tuple of MatchedPair
    left_only, right_only : tuple of InstructionRef
        Instructions with no counterpart: removed and added respectively.
    entry : bool
        Whether this is the entry computation, which is worth leading with.
    """

    left: str | None
    right: str | None
    pairs: tuple[MatchedPair, ...]
    left_only: tuple[InstructionRef, ...]
    right_only: tuple[InstructionRef, ...]
    entry: bool = False

    @property
    def changed(self) -> bool:
        """Whether anything here differs."""
        return bool(self.left_only or self.right_only or any(p.changed for p in self.pairs))

    @property
    def label(self) -> str:
        """A human-readable name for this computation pair."""
        if self.left is None:
            return f"%{self.right}"
        if self.right is None:
            return f"%{self.left}"
        if self.left == self.right:
            return f"%{self.left}"
        return f"%{self.left} -> %{self.right}"


@dataclass(frozen=True, slots=True)
class DiffSummary:
    """Counts, for a headline that beats scrolling.

    Attributes
    ----------
    added, removed, changed : dict
        Opcode to instruction count.
    renamed : int
        Matched instructions whose only difference is their name.
    instruction_delta : int
        Net change in instruction count.
    computations_added, computations_removed : int
    """

    added: dict[str, int]
    removed: dict[str, int]
    changed: dict[str, int]
    renamed: int
    instruction_delta: int
    computations_added: int
    computations_removed: int

    def headline(self) -> str:
        """One line naming the magnitude of the change."""
        parts: list[str] = []
        for label, counts in (
            ("changed", self.changed),
            ("removed", self.removed),
            ("added", self.added),
        ):
            total = sum(counts.values())
            if total:
                parts.append(f"{total} {label}")
        if self.renamed:
            parts.append(f"{self.renamed} renamed")
        for label, count in (
            ("added", self.computations_added),
            ("removed", self.computations_removed),
        ):
            if count:
                parts.append(f"{count} computation{'s' if count > 1 else ''} {label}")
        if not parts:
            return "no structural change"
        delta = f"  [{self.instruction_delta:+d} instructions]" if self.instruction_delta else ""
        return ", ".join(parts) + delta


@dataclass(frozen=True, slots=True)
class ModuleDiff:
    """The correspondence between two modules.

    Attributes
    ----------
    computations : tuple of ComputationDiff
        Only the ones that differ; matched-and-identical computations are dropped, since a
        pass typically leaves all but one alone.
    truncated : bool
        Whether a work cap was hit, so the matching is weaker than usual: either some
        same-shape candidates went unconsidered, or — in the extreme — leftovers were paired
        by position rather than by score.
    """

    computations: tuple[ComputationDiff, ...]
    truncated: bool = False

    @property
    def identical(self) -> bool:
        """Whether the two modules are structurally the same program."""
        return not any(computation.changed for computation in self.computations)

    def summary(self) -> DiffSummary:
        """Aggregate counts across every computation."""
        added: dict[str, int] = {}
        removed: dict[str, int] = {}
        changed: dict[str, int] = {}
        renamed = 0
        computations_added = 0
        computations_removed = 0
        for computation in self.computations:
            if computation.left is None:
                computations_added += 1
            elif computation.right is None:
                computations_removed += 1
            for ref in computation.right_only:
                added[ref.opcode] = added.get(ref.opcode, 0) + 1
            for ref in computation.left_only:
                removed[ref.opcode] = removed.get(ref.opcode, 0) + 1
            for pair in computation.pairs:
                if pair.changed:
                    changed[pair.left.opcode] = changed.get(pair.left.opcode, 0) + 1
                elif pair.renamed:
                    renamed += 1
        return DiffSummary(
            added=added,
            removed=removed,
            changed=changed,
            renamed=renamed,
            instruction_delta=sum(added.values()) - sum(removed.values()),
            computations_added=computations_added,
            computations_removed=computations_removed,
        )


def _dice(left: dict[str, int], right: dict[str, int]) -> float:
    """Similarity of two opcode histograms, 1.0 when identical."""
    shared = sum(min(count, right.get(opcode, 0)) for opcode, count in left.items())
    total = sum(left.values()) + sum(right.values())
    return 2 * shared / total if total else 1.0


def _match_computations(
    left: Module, right: Module, left_prints: Fingerprints, right_prints: Fingerprints
) -> tuple[list[tuple[Computation, Computation]], list[Computation], list[Computation]]:
    """Pair up computations before looking at any instruction.

    Matching computations first is what keeps the whole thing affordable: it turns one
    module-wide quadratic problem into one small quadratic problem per computation.
    """
    pairs: list[tuple[Computation, Computation]] = []
    spare_left = list(left.computations)
    spare_right = list(right.computations)

    def take(one: Computation, other: Computation) -> None:
        pairs.append((one, other))
        spare_left.remove(one)
        spare_right.remove(other)

    entry_left, entry_right = left.entry_computation, right.entry_computation
    if entry_left is not None and entry_right is not None:
        take(entry_left, entry_right)

    for key in (
        lambda computation, prints: prints.computation.get(computation.name, computation.name),
        lambda computation, _: computation.name,
        lambda computation, _: computation.canonical_name,
    ):
        buckets: dict[str, list[Computation]] = {}
        for computation in spare_right:
            buckets.setdefault(key(computation, right_prints), []).append(computation)
        for computation in list(spare_left):
            candidates = buckets.get(key(computation, left_prints), [])
            if len(candidates) == 1 and candidates[0] in spare_right:
                take(computation, candidates[0])

    # Whatever is left is matched on shape of content, best score first.
    scored = sorted(
        (
            (-_dice(one.opcodes(), other.opcodes()), one.name, other.name, one, other)
            for one in spare_left
            for other in spare_right
        ),
        key=lambda item: item[:3],
    )
    for negative, _, _, one, other in scored:
        if -negative < MIN_SIMILARITY:
            break
        if one in spare_left and other in spare_right:
            take(one, other)

    return pairs, spare_left, spare_right


def _bucket_by_hash(computation: Computation, prints: Fingerprints) -> dict[str, list[Instruction]]:
    """Group a computation's instructions by the hash of the subgraph they root."""
    buckets: dict[str, list[Instruction]] = {}
    for instruction in computation.instructions:
        key = prints.instruction.get((computation.name, instruction.name), instruction.name)
        buckets.setdefault(key, []).append(instruction)
    return buckets


def _score(
    one: Instruction, other: Instruction, mapping: dict[str, str], options: Options
) -> float:
    """How likely two instructions are the same instruction, given what already matched.

    Only positive evidence scores. Two instructions that merely both lack operands, both
    lack a literal and both lack a parameter number have nothing in common, and rewarding
    those agreements would pair arbitrary leaves with each other.
    """
    total = 0.0
    if one.operands:
        mapped = [mapping.get(operand, operand) for operand in one.operands]
        shared = sum(1 for name in mapped if name in other.operands)
        total += 3.0 * shared / len(one.operands)
    if one.canonical_name == other.canonical_name:
        total += 2.0
    if one.name == other.name:
        total += 1.0
    if one.opcode == other.opcode:
        total += 1.5
    if not options.ignore_shape and one.shape == other.shape:
        total += 1.0
    if not options.ignore_layout and one.layout == other.layout:
        total += 0.5
    if one.parameter_number is not None and one.parameter_number == other.parameter_number:
        total += 1.0
    if one.literal and one.literal == other.literal:
        total += 1.0
    return total


def _changes(
    one: Instruction,
    other: Instruction,
    mapping: dict[str, str],
    computations: dict[str, str],
    options: Options,
) -> tuple[Change, ...]:
    """Which fields differ between two matched instructions.

    Operand and callee names are mapped through the match before comparing, so a pure
    rename is not a change.
    """
    found: list[Change] = []
    if one.opcode != other.opcode:
        found.append("opcode")
    if not options.ignore_shape and one.shape != other.shape:
        found.append("shape")
    if not options.ignore_layout and one.layout != other.layout:
        found.append("layout")
    if tuple(mapping.get(name, name) for name in one.operands) != other.operands:
        found.append("operands")
    if one.attributes != other.attributes:
        found.append("attributes")
    if tuple(computations.get(name, name) for name in one.called) != other.called:
        found.append("called")
    if one.is_root != other.is_root:
        found.append("root")
    return tuple(found)


def _diff_computation(
    left: Computation,
    right: Computation,
    left_prints: Fingerprints,
    right_prints: Fingerprints,
    computations: dict[str, str],
    options: Options,
) -> tuple[ComputationDiff, bool]:
    """Match two computations instruction by instruction."""
    truncated = False
    mapping: dict[str, str] = {}
    pairs: list[MatchedPair] = []
    spare_left: list[Instruction] = []
    spare_right: list[Instruction] = []

    # Equal subgraph hashes are the same subgraph, whatever the printed order says. This is
    # what makes a rescheduling pass report nothing.
    left_buckets = _bucket_by_hash(left, left_prints)
    right_buckets = _bucket_by_hash(right, right_prints)
    for key, ones in left_buckets.items():
        others = right_buckets.get(key, [])
        for one, other in zip(ones, others, strict=False):
            mapping[one.name] = other.name
            pairs.append(
                MatchedPair(
                    left=InstructionRef.of(left.name, one),
                    right=InstructionRef.of(right.name, other),
                    changes=(),
                    renamed=one.name != other.name,
                )
            )
        spare_left.extend(ones[len(others) :])
        spare_right.extend(others[len(ones) :])
    for key, others in right_buckets.items():
        if key not in left_buckets:
            spare_right.extend(others)

    # Everything left over is a real difference, matched on how similar it is.
    def bucket(instruction: Instruction) -> tuple[str, str, int]:
        return (
            instruction.opcode,
            "" if options.ignore_shape else instruction.shape,
            len(instruction.operands),
        )

    taken_left: set[str] = set()
    taken_right: set[str] = set()

    def greedy(pairs: list[tuple[Instruction, Instruction]], threshold: float) -> None:
        """Take the best-scoring pairings first, skipping anything already spoken for."""
        scored = sorted(
            (
                (-_score(one, other, mapping, options), one.name, other.name, one, other)
                for one, other in pairs
            ),
            key=lambda item: item[:3],
        )
        for negative, _, _, one, other in scored:
            if -negative < threshold:
                break
            if one.name in taken_left or other.name in taken_right:
                continue
            taken_left.add(one.name)
            taken_right.add(other.name)
            mapping[one.name] = other.name

    # First within identical (opcode, shape, arity) buckets, where the bucket is itself the
    # evidence and any pairing beats reporting both sides separately.
    grouped: dict[tuple[str, str, int], list[Instruction]] = {}
    for instruction in spare_right:
        grouped.setdefault(bucket(instruction), []).append(instruction)
    strict: list[tuple[Instruction, Instruction]] = []
    for one in spare_left:
        others = grouped.get(bucket(one), [])
        if len(others) > MAX_BUCKET or len(strict) > MAX_PAIRS:
            truncated = True
            continue
        strict.extend((one, other) for other in others)
    greedy(strict, 0.0)

    # Then across buckets, so that changing an instruction's shape or opcode reports it as
    # changed rather than as one removal plus one addition. This needs a threshold: without
    # the bucket vouching for them, only real similarity should pair two instructions. Note
    # this pass sees everything the strict cap skipped, so hitting that cap costs a little
    # matching quality but never a fair hearing.
    rest_left = [one for one in spare_left if one.name not in taken_left]
    rest_right = [other for other in spare_right if other.name not in taken_right]
    if len(rest_left) * len(rest_right) <= MAX_PAIRS:
        greedy([(one, other) for one in rest_left for other in rest_right], MIN_SCORE)
    else:
        # Only here has scoring been skipped outright, and only here is a positional guess
        # better than reporting every leftover as both removed and added. Doing this whenever
        # anything was truncated would overrule the pairs the scored pass declined on merit.
        truncated = True
        for one, other in zip(rest_left, rest_right, strict=False):
            taken_left.add(one.name)
            taken_right.add(other.name)
            mapping[one.name] = other.name

    by_name = {instruction.name: instruction for instruction in spare_right}
    for one in spare_left:
        other = by_name.get(mapping.get(one.name, ""))
        if other is None:
            continue
        pairs.append(
            MatchedPair(
                left=InstructionRef.of(left.name, one),
                right=InstructionRef.of(right.name, other),
                changes=_changes(one, other, mapping, computations, options),
                renamed=one.name != other.name,
            )
        )

    return (
        ComputationDiff(
            left=left.name,
            right=right.name,
            pairs=tuple(sorted(pairs, key=lambda pair: pair.left.line)),
            left_only=tuple(
                InstructionRef.of(left.name, one)
                for one in spare_left
                if one.name not in taken_left
            ),
            right_only=tuple(
                InstructionRef.of(right.name, other)
                for other in spare_right
                if other.name not in taken_right
            ),
            entry=left.entry or right.entry,
        ),
        truncated,
    )


def diff_modules(left: Module, right: Module, *, options: Options | None = None) -> ModuleDiff:
    """Compare two parsed modules structurally.

    Parameters
    ----------
    left, right : Module
    options : Options, optional
        Must be the same options the modules were parsed with, or the fingerprints will not
        line up.

    Returns
    -------
    ModuleDiff
        Only differing computations are listed, so an unchanged module yields an empty diff
        whose :attr:`ModuleDiff.identical` is ``True``.
    """
    options = options or Options()
    left_prints = fingerprint(left, options=options)
    right_prints = fingerprint(right, options=options)
    if left_prints.module == right_prints.module:
        return ModuleDiff(computations=())

    matched, only_left, only_right = _match_computations(left, right, left_prints, right_prints)
    computations = {one.name: other.name for one, other in matched}

    truncated = False
    diffs: list[ComputationDiff] = []
    for one, other in matched:
        if left_prints.computation.get(one.name) == right_prints.computation.get(other.name):
            continue
        diff, hit_cap = _diff_computation(
            one, other, left_prints, right_prints, computations, options
        )
        truncated = truncated or hit_cap
        if diff.changed:
            diffs.append(diff)

    diffs.extend(
        ComputationDiff(
            left=one.name,
            right=None,
            pairs=(),
            left_only=tuple(InstructionRef.of(one.name, i) for i in one.instructions),
            right_only=(),
            entry=one.entry,
        )
        for one in only_left
    )
    diffs.extend(
        ComputationDiff(
            left=None,
            right=other.name,
            pairs=(),
            left_only=(),
            right_only=tuple(InstructionRef.of(other.name, i) for i in other.instructions),
            entry=other.entry,
        )
        for other in only_right
    )

    # Entry first, then by name, so the same pass reads the same way every time.
    diffs.sort(key=lambda diff: (not diff.entry, diff.label))
    return ModuleDiff(computations=tuple(diffs), truncated=truncated)


def render_module_diff(diff: ModuleDiff, *, limit: int = MAX_DETAIL_LINES) -> str:
    """Render a diff as plain text.

    Sigils rather than colour: the Passes pane renders with Rich markup disabled, because
    HLO is full of things like ``f32[8,16]`` that markup would eat.

    Parameters
    ----------
    diff : ModuleDiff
    limit : int, optional
        Maximum instruction lines per computation before the rest is summarized.

    Returns
    -------
    str
    """
    summary = diff.summary()
    if diff.identical:
        return "  no structural change"

    out: list[str] = [f"  {summary.headline()}"]
    if diff.truncated:
        out.append("  (matching hit a work cap, so some pairings are positional)")

    for computation in diff.computations:
        if computation.left is None:
            out.append(f"  + {computation.label}  ({len(computation.right_only)} instructions)")
            continue
        if computation.right is None:
            out.append(f"  - {computation.label}  ({len(computation.left_only)} instructions)")
            continue

        out.append(f"  ~ {computation.label}")
        detail: list[tuple[int, str]] = [
            (ref.line, f"      - {ref.text.strip()}") for ref in computation.left_only
        ]
        detail += [(ref.line, f"      + {ref.text.strip()}") for ref in computation.right_only]
        detail += [
            (pair.left.line, f"      ~ %{pair.left.name}  {', '.join(pair.changes)}")
            for pair in computation.pairs
            if pair.changed
        ]
        detail.sort()
        for _, line in detail[:limit]:
            out.append(line)
        if len(detail) > limit:
            out.append(f"      ... and {len(detail) - limit} more")

    return "\n".join(out)


@dataclass(slots=True)
class _Transition:
    """One before/after pair, with both modules already parsed."""

    before: PassSnapshot
    after: PassSnapshot
    left: Module
    right: Module
    diff: ModuleDiff = field(default_factory=lambda: ModuleDiff(computations=()))


def structural_pass_report(
    snapshots: Sequence[PassSnapshot],
    *,
    attribute: Callable[[PassSnapshot, PassSnapshot], str],
    options: Options | None = None,
) -> str | None:
    """Summarize what each XLA pass did to the module, structurally.

    Keeps the document shape :func:`jaxplorer.hlo.pass_report` established — headline, one
    block per transition that changed something, then the pipeline order as an index — so
    switching modes does not move everything around.

    Parameters
    ----------
    snapshots : sequence of PassSnapshot
        Snapshots in the order XLA produced them.
    attribute : callable
        Names the pass responsible for a transition. Passed in rather than imported so this
        module stays independent of :mod:`jaxplorer.hlo`.
    options : Options, optional

    Returns
    -------
    str or None
        ``None`` when the module text did not parse well enough to trust a structural
        answer, which is the caller's cue to show a text diff instead.
    """
    if not snapshots:
        return None

    options = options or Options()
    modules = [parse_module(snapshot.text, options=options) for snapshot in snapshots]
    unparsed = sum(module.unparsed for module in modules)
    read = sum(module.lines for module in modules)
    if not read or any(not module.computations for module in modules):
        return None
    if unparsed / (unparsed + read) > MAX_UNPARSED_FRACTION:
        return None

    blocks: list[str] = []
    changed: set[int] = set()
    pairs = list(zip(snapshots, modules, strict=True))
    for (before, left), (after, right) in zip(pairs, pairs[1:], strict=False):
        diff = diff_modules(left, right, options=options)
        if diff.identical:
            continue
        changed.add(before.index)
        blocks.append(f"===== {attribute(before, after)} =====\n{render_module_diff(diff)}")

    head = f"{len(snapshots)} snapshots, {len(changed)} changed the module."
    body = "\n\n".join(blocks) if blocks else "No pass changed the module."
    index = "\n".join(
        f"  {'*' if snapshot.index in changed else ' '} {snapshot.label}" for snapshot in snapshots
    )
    return f"{head}\n\n{body}\n\n===== pipeline order =====\n{index}\n"
