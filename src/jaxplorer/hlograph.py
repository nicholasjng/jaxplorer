"""An HLO module, as a graph rather than as lines of text.

Text is enough to recover the DAG a diff has to compare: XLA's printer puts one instruction per
line and never nests computations, so a fusion body, a ``while`` body and a ``conditional``
branch are all separate top-level computations referenced by name.

Pure text handling, so neither jax nor textual is needed. Nothing here raises on malformed
input: HLO text is a debug format with no stability guarantee, so an unreadable line is counted
in :attr:`Module.unparsed` and skipped, and callers decide whether the count is low enough to
trust the result.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace

# Attribute keys naming a called computation. The parser also accepts anything ending in
# ``computation`` or ``computations``, since the list grows with every control-flow opcode.
CALLEE_KEYS = frozenset(
    {
        "calls",
        "to_apply",
        "body",
        "condition",
        "select",
        "scatter",
        "update_computation",
        "true_computation",
        "false_computation",
        "branch_computations",
        "called_computations",
        "fused_computations",
    }
)

# Attributes that say nothing about what the module computes. ``metadata`` is pulled out
# separately because click-to-source needs it; the rest are dropped outright.
NOISE_KEYS = frozenset({"metadata", "sharding", "origin", "frontend_attributes"})

# A malformed line must not be able to swallow the rest of the module, so continuation
# stops here and the line is counted as unparsed.
MAX_CONTINUATION = 200

_MODULE = re.compile(r"^HloModule\s+(?P<name>[\w.$-]+)")
_HEADER = re.compile(
    r"^\s*(?P<entry>ENTRY\s+)?%?(?P<name>[\w.$-]+)\s*\((?P<params>.*)\)\s*->\s*(?P<result>.+?)\s*\{\s*$"
)
_INSTRUCTION = re.compile(r"^\s*(?:(?P<root>ROOT)\s+)?%?(?P<name>[\w.$-]+)\s*=\s*(?P<rest>\S.*)$")
_REFERENCE = re.compile(r"%(?P<name>[\w.$-]+)")
_IDENTIFIER = re.compile(r"^[A-Za-z_][\w.$-]*$")
# Trailing ``{1,0}`` on a shape is its layout, which most diffs should ignore.
_LAYOUT = re.compile(r"^(?P<shape>.*?)(?P<layout>\{[^{}]*\})$")
# ``%fusion.3`` and ``%fusion.5`` are the same instruction renumbered.
_SUFFIX = re.compile(r"\.\d+$")

_DIGEST_SIZE = 16


@dataclass(frozen=True, slots=True)
class Options:
    """What to ignore when fingerprinting.

    Attributes
    ----------
    ignore_layout : bool
        Treat ``f32[4,4]{1,0}`` and ``f32[4,4]{0,1}`` as the same. On by default: layout
        assignment is a pass like any other, and its churn drowns everything else.
    ignore_shape : bool
        Treat shapes as absent entirely, mirroring the upstream tool's ``--ignore_shape``.
    ignore_backend_config : bool
        Drop ``backend_config``, which carries autotuning results and scheduling hints.
    """

    ignore_layout: bool = True
    ignore_shape: bool = False
    ignore_backend_config: bool = True


@dataclass(frozen=True, slots=True)
class Instruction:
    """One line of an HLO computation, taken apart.

    Attributes
    ----------
    name : str
        Instruction name without the leading ``%``.
    opcode : str
        Opcode as printed, so ``get-tuple-element`` rather than ``kGetTupleElement``.
    shape : str
        Result shape with any layout split off, e.g. ``f32[4,4]``.
    layout : str
        The layout that was split off, e.g. ``{1,0}``, or ``""``.
    operands : tuple of str
        Operand names, in order, without ``%``.
    called : tuple of str
        Names of computations this instruction calls.
    attributes : tuple of tuple of str
        Remaining ``key=value`` attributes, sorted, with the noise keys removed.
    metadata : str
        The raw ``metadata={...}`` value, kept so click-to-source still works.
    literal : str
        For operand-less instructions such as ``constant`` and ``iota``, the text inside
        the parentheses. Otherwise ``""``.
    is_root : bool
        Whether the line was marked ``ROOT``.
    parameter_number : int or None
        The index for ``parameter`` instructions, otherwise ``None``.
    line : int
        1-based line number in the module text.
    text : str
        The original line, verbatim, so rendering a diff can quote it.
    """

    name: str
    opcode: str
    shape: str
    layout: str
    operands: tuple[str, ...]
    called: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    metadata: str
    literal: str
    is_root: bool
    parameter_number: int | None
    line: int
    text: str

    @property
    def canonical_name(self) -> str:
        """The name with any numbering suffix removed, so ``fusion.3`` becomes ``fusion``."""
        return _SUFFIX.sub("", self.name)


@dataclass(frozen=True, slots=True)
class Computation:
    """A single computation: a named DAG of instructions.

    Attributes
    ----------
    name : str
        Computation name without the leading ``%``.
    parameters : tuple of str
        Parameter declarations from the header, in order.
    result : str
        Result shape as printed.
    entry : bool
        Whether this was the ``ENTRY`` computation.
    instructions : tuple of Instruction
        Instructions in printed order, which scheduling passes change without changing the
        DAG. Nothing here depends on that order.
    root : str
        Name of the root instruction.
    """

    name: str
    parameters: tuple[str, ...]
    result: str
    entry: bool
    instructions: tuple[Instruction, ...]
    root: str

    @property
    def canonical_name(self) -> str:
        """The name with any numbering suffix removed."""
        return _SUFFIX.sub("", self.name)

    @property
    def parameter_shapes(self) -> tuple[str, ...]:
        """Just the shapes from the header, dropping the parameter names.

        ``x.1: f32[4,4]`` contributes ``f32[4,4]``. The name is renumbering noise, and
        including it would make a renamed computation look like a changed one.
        """
        return tuple(
            declaration.partition(":")[2].strip() or declaration for declaration in self.parameters
        )

    def index(self) -> dict[str, Instruction]:
        """Map instruction name to instruction."""
        return {instruction.name: instruction for instruction in self.instructions}

    def opcodes(self) -> dict[str, int]:
        """Count instructions per opcode, for similarity scoring."""
        counts: dict[str, int] = {}
        for instruction in self.instructions:
            counts[instruction.opcode] = counts.get(instruction.opcode, 0) + 1
        return counts


@dataclass(slots=True)
class Module:
    """A parsed HLO module.

    Attributes
    ----------
    name : str
        Module name, or ``""`` if the header was missing.
    computations : tuple of Computation
        Every computation, in printed order.
    entry : str or None
        Name of the ``ENTRY`` computation, if the text named one.
    unparsed : int
        Lines inside a computation body that could not be read. A diff built from a module
        with a high count is not trustworthy, which is the caller's cue to fall back to
        comparing text.
    lines : int
        Instruction lines read successfully, so ``unparsed`` has a denominator.
    """

    name: str
    computations: tuple[Computation, ...]
    entry: str | None
    unparsed: int
    lines: int
    _index: dict[str, Computation] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Build the name lookup once, since every matcher phase needs it."""
        self._index = {computation.name: computation for computation in self.computations}

    def get(self, name: str) -> Computation | None:
        """Look a computation up by name."""
        return self._index.get(name)

    @property
    def entry_computation(self) -> Computation | None:
        """The ``ENTRY`` computation, falling back to the last one printed.

        XLA prints callees before callers, so the last computation is the entry whenever the
        marker is missing.
        """
        if self.entry is not None:
            found = self.get(self.entry)
            if found is not None:
                return found
        return self.computations[-1] if self.computations else None

    @property
    def instruction_count(self) -> int:
        """Total instructions across every computation."""
        return sum(len(computation.instructions) for computation in self.computations)


def _depth(text: str) -> int:
    """Net bracket depth of ``text``, ignoring brackets inside double-quoted strings.

    Quote awareness is not optional: ``backend_config={"known_trip_count":{"n":"4"}}`` and
    ``custom_call_target="foo(bar)"`` both put brackets inside strings.
    """
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def _scan(text: str) -> list[tuple[int, str, int]]:
    """Yield ``(position, character, depth_before)`` for every bracket outside a string."""
    out: list[tuple[int, str, int]] = []
    depth = 0
    in_string = False
    escaped = False
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            out.append((position, char, depth))
            depth += 1
        elif char in ")]}":
            depth -= 1
            out.append((position, char, depth))
    return out


def _split_top(text: str, separator: str = ",") -> list[str]:
    """Split on ``separator`` at bracket depth zero, ignoring separators inside strings."""
    parts: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    start = 0
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:position])
            start = position + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _operand_list(rest: str) -> tuple[int, int] | None:
    """Locate the operand parentheses in an instruction's right-hand side.

    Returns
    -------
    tuple of int or None
        Positions of the opening and closing parenthesis, or ``None`` when there is no
        operand list.

    Notes
    -----
    The rule is *the first depth-zero* ``(`` *immediately preceded by a word character*. Both
    halves matter. A tuple-valued instruction prints its shape first::

        ROOT %t = (f32[4], f32[4]) tuple(%a, %b)

    so taking the first ``(`` would grab the shape; requiring a preceding word character
    skips it, because shapes only ever open a bracket after a space, a comma or another
    bracket. Taking the *first* qualifying one rather than the last keeps a word-preceded
    parenthesis in a trailing attribute from being mistaken for the operand list.
    """
    brackets = _scan(rest)
    for position, char, depth in brackets:
        if char != "(" or depth != 0 or position == 0:
            continue
        if not re.match(r"[\w.$-]", rest[position - 1]):
            continue
        for close, closing, close_depth in brackets:
            if closing == ")" and close_depth == 0 and close > position:
                return position, close
        return None
    return None


def _attributes(text: str) -> list[tuple[str, str]]:
    """Parse a trailing ``key=value, key=value`` run into pairs."""
    pairs: list[tuple[str, str]] = []
    for part in _split_top(text):
        head, _, tail = part.partition("=")
        if tail:
            pairs.append((head.strip(), tail.strip()))
        else:
            # A bare token, e.g. ``is_scheduled``. Keep it as a valueless attribute rather
            # than dropping a difference on the floor.
            pairs.append((part, ""))
    return pairs


def _operands(inside: str, opcode: str) -> tuple[tuple[str, ...], int | None, str]:
    """Split an operand list into references, a parameter number, and a literal.

    An operand may be printed bare (``p0``) or with its shape (``f32[4] %a``), so a ``%``
    reference is looked for first and a plain identifier accepted only as a fallback.
    """
    if not inside.strip():
        return (), None, ""
    names: list[str] = []
    leftovers: list[str] = []
    for part in _split_top(inside):
        reference = _REFERENCE.search(part)
        if reference is not None:
            names.append(reference.group("name"))
        elif _IDENTIFIER.match(part):
            names.append(part)
        else:
            leftovers.append(part)
    if names:
        return tuple(names), None, ""
    joined = ", ".join(leftovers)
    if opcode == "parameter":
        try:
            return (), int(joined), ""
        except ValueError:
            return (), None, joined[:128]
    # ``constant``, ``iota`` and friends: the payload is the only thing distinguishing them.
    return (), None, joined[:128]


def _split_layout(shape: str) -> tuple[str, str]:
    """Split a trailing layout off a shape, leaving tuple shapes alone."""
    match = _LAYOUT.match(shape)
    if match is None or not match.group("shape"):
        return shape, ""
    return match.group("shape"), match.group("layout")


def _parse_instruction(
    name: str, rest: str, *, is_root: bool, line: int, text: str, options: Options
) -> Instruction | None:
    """Take one instruction's right-hand side apart, or return ``None`` if unreadable."""
    span = _operand_list(rest)
    if span is None:
        return None
    open_at, close_at = span
    head = rest[:open_at].strip()
    inside = rest[open_at + 1 : close_at]
    tail = rest[close_at + 1 :].lstrip().lstrip(",")

    shape, _, opcode = head.rpartition(" ")
    if not opcode:
        return None
    shape, layout = _split_layout(shape.strip())

    operands, parameter_number, literal = _operands(inside, opcode)

    metadata = ""
    called: list[str] = []
    attributes: list[tuple[str, str]] = []
    for key, value in _attributes(tail):
        if key == "metadata":
            metadata = value
            continue
        if key == "backend_config" and options.ignore_backend_config:
            continue
        if key in NOISE_KEYS:
            continue
        if key in CALLEE_KEYS or key.endswith(("computation", "computations")):
            called.extend(match.group("name") for match in _REFERENCE.finditer(value))
            continue
        attributes.append((key, value))

    return Instruction(
        name=name,
        opcode=opcode,
        shape=shape,
        layout=layout,
        operands=operands,
        called=tuple(called),
        attributes=tuple(sorted(attributes)),
        metadata=metadata,
        literal=literal,
        is_root=is_root,
        parameter_number=parameter_number,
        line=line,
        text=text,
    )


def parse_module(text: str, *, options: Options | None = None) -> Module:
    """Parse HLO module text into a graph.

    Parameters
    ----------
    text : str
        A module as XLA's printer emits it, with or without the debug tables. Table lines
        need no special handling: they cannot match a computation header, so they are
        skipped like any other line outside a computation.
    options : Options, optional
        Controls which attributes are dropped while parsing.

    Returns
    -------
    Module
        Always a module, never an exception. Unreadable lines inside a computation body are
        counted in :attr:`Module.unparsed`.
    """
    options = options or Options()
    lines = text.split("\n")
    name = ""
    entry: str | None = None
    computations: list[Computation] = []
    unparsed = 0
    read = 0

    header: re.Match[str] | None = None
    instructions: list[Instruction] = []
    root = ""

    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1

        if header is None:
            if not name:
                match = _MODULE.match(stripped)
                if match is not None:
                    name = match.group("name")
                    continue
            found = _HEADER.match(raw)
            if found is not None:
                header = found
                instructions = []
                root = ""
                if found.group("entry"):
                    entry = found.group("name")
            continue

        if stripped == "}":
            computations.append(
                Computation(
                    name=header.group("name"),
                    parameters=tuple(_split_top(header.group("params"))),
                    result=header.group("result"),
                    entry=bool(header.group("entry")),
                    instructions=tuple(instructions),
                    root=root or (instructions[-1].name if instructions else ""),
                )
            )
            header = None
            continue

        found = _INSTRUCTION.match(raw)
        if found is None:
            if stripped:
                unparsed += 1
            continue

        # A constant literal or a long attribute can wrap, so pull in following lines until
        # the brackets balance.
        logical = raw
        start = index
        consumed = 0
        while _depth(logical) > 0 and index < len(lines) and consumed < MAX_CONTINUATION:
            logical += "\n" + lines[index]
            index += 1
            consumed += 1
        if _depth(logical) > 0:
            unparsed += 1
            index = start
            continue

        found = _INSTRUCTION.match(logical.replace("\n", " "))
        if found is None:
            unparsed += 1
            continue
        instruction = _parse_instruction(
            found.group("name"),
            found.group("rest"),
            is_root=bool(found.group("root")),
            line=start,
            text=raw.rstrip(),
            options=options,
        )
        if instruction is None:
            unparsed += 1
            continue
        read += 1
        instructions.append(instruction)
        if instruction.is_root:
            root = instruction.name

    return Module(
        name=name,
        computations=tuple(computations),
        entry=entry,
        unparsed=unparsed,
        lines=read,
    )


@dataclass(slots=True)
class Fingerprints:
    """Structural hashes for one module.

    Hashes are content-derived rather than :func:`hash`-derived so that two runs, and two
    processes, agree.

    Attributes
    ----------
    instruction : dict
        ``(computation name, instruction name)`` to the hash of the subgraph rooted there.
    computation : dict
        Computation name to the hash of its root subgraph plus its signature.
    module : str
        Hash over every computation hash, order-independent.
    """

    instruction: dict[tuple[str, str], str] = field(default_factory=dict)
    computation: dict[str, str] = field(default_factory=dict)
    module: str = ""


def _digest(*parts: object) -> str:
    """Hash ``parts`` into a short hex digest."""
    hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for part in parts:
        hasher.update(repr(part).encode())
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _token(instruction: Instruction, options: Options) -> tuple[object, ...]:
    """The part of an instruction's identity that does not depend on its operands."""
    return (
        instruction.opcode,
        "" if options.ignore_shape else instruction.shape,
        "" if options.ignore_shape or options.ignore_layout else instruction.layout,
        instruction.attributes,
        instruction.literal,
        instruction.parameter_number,
    )


def _call_order(module: Module) -> list[Computation]:
    """Order computations callees-first, so a caller can hash its callees.

    A cycle would mean a computation calling itself, which XLA does not emit; if one shows
    up anyway the remaining computations are appended in printed order rather than dropped.
    """
    order: list[Computation] = []
    state: dict[str, int] = {}
    for start in module.computations:
        if state.get(start.name):
            continue
        stack: list[tuple[Computation, bool]] = [(start, False)]
        while stack:
            computation, expanded = stack.pop()
            if expanded:
                if state.get(computation.name) != 2:
                    state[computation.name] = 2
                    order.append(computation)
                continue
            if state.get(computation.name):
                continue
            state[computation.name] = 1
            stack.append((computation, True))
            for instruction in computation.instructions:
                for callee in instruction.called:
                    found = module.get(callee)
                    if found is not None and not state.get(found.name):
                        stack.append((found, False))
    for computation in module.computations:
        if state.get(computation.name) != 2:
            order.append(computation)
    return order


def fingerprint(module: Module, *, options: Options | None = None) -> Fingerprints:
    """Hash every subgraph in ``module``.

    Two instructions share a hash when their opcodes, shapes, attributes and *operand
    subgraphs* agree, which is what makes the comparison structural: renaming an
    instruction or reordering a computation's lines cannot change it.

    Parameters
    ----------
    module : Module
    options : Options, optional

    Returns
    -------
    Fingerprints
    """
    options = options or Options()
    prints = Fingerprints()

    for computation in _call_order(module):
        by_name = computation.index()
        local: dict[str, str] = {}
        for instruction in computation.instructions:
            if instruction.name in local:
                continue
            # Iterative post-order: a 2000-instruction chain must not exhaust the stack.
            stack: list[tuple[Instruction, bool]] = [(instruction, False)]
            active: set[str] = set()
            while stack:
                current, expanded = stack.pop()
                if expanded:
                    active.discard(current.name)
                    operands = tuple(
                        local.get(operand, _digest("?", operand)) for operand in current.operands
                    )
                    callees = tuple(
                        prints.computation.get(callee, _digest("?", callee))
                        for callee in current.called
                    )
                    local[current.name] = _digest(_token(current, options), operands, callees)
                    continue
                if current.name in local or current.name in active:
                    continue
                active.add(current.name)
                stack.append((current, True))
                for operand in current.operands:
                    found = by_name.get(operand)
                    if found is not None and found.name not in local:
                        stack.append((found, False))

        for name, value in local.items():
            prints.instruction[computation.name, name] = value
        signature = (
            len(computation.parameters) if options.ignore_shape else computation.parameter_shapes
        )
        # Every instruction's hash, not just the root's: a pass that rewires a use leaves the
        # old producer unreachable until DCE runs, and an edit to it still counts. Sorted, so
        # reordering the lines changes nothing.
        prints.computation[computation.name] = _digest(
            local.get(computation.root, ""),
            tuple(sorted(local.values())),
            signature,
            len(computation.instructions),
        )

    prints.module = _digest(tuple(sorted(prints.computation.values())))
    return prints


def canonicalize(module: Module) -> Module:
    """Strip metadata from every instruction, for callers that only want structure.

    Used by tests and by the A/B recipe in ``docs/xla-introspection.md``; the matcher does
    not need it, since fingerprints already exclude metadata.
    """
    return replace(
        module,
        computations=tuple(
            replace(
                computation,
                instructions=tuple(
                    replace(instruction, metadata="") for instruction in computation.instructions
                ),
            )
            for computation in module.computations
        ),
    )
