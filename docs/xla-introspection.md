# XLA introspection workflows

Recipes for looking at what XLA did to a program, and for comparing what two XLA builds did
to the same program.

jaxplorer covers most of this itself now: `--passes` (or `f6`) fills the Passes pane with a diff per
pass and the LLVM IR pane with what went to LLVM, using the same dumps described below, and `f4`
switches that pane between a text diff and a structural one. Reach for the raw flags when you want
the files on disk, a backend jaxplorer is not running, or the comparison in the last section.

Everything marked *verified* was run while writing this, on jax and jaxlib 0.11.0, CPU
backend, macOS arm64, Python 3.13. File counts and names come from that run and will drift
with XLA; treat them as shape, not contract. GPU-only items are marked *unverified here*.

## Which layer owns which IR

Knowing who produced a piece of IR tells you where a change came from.

| IR | Produced by | jaxplorer pane |
| --- | --- | --- |
| jaxpr | JAX tracing | Jaxpr |
| StableHLO | JAX lowering | StableHLO |
| optimized HLO | XLA's HLO passes | Optimized HLO |
| LLVM IR | XLA CPU backend, via LLVM | LLVM IR (with `--passes`) |
| object code | LLVM | not shown, see below |

The practical consequence: **StableHLO is the control.** If it changes, the cause is in JAX or
in your snippet, not in XLA. Only differences that appear first in optimized HLO belong to the
compiler.

## Dumping the whole pipeline

jaxplorer's Passes and LLVM IR panes are built from exactly these dumps, requested per compile
through `compile(compiler_options=...)` rather than the environment. To get the files on disk
instead, dump them yourself.

```bash
XLA_FLAGS="--xla_dump_to=/tmp/dump" uv run jaxplorer mlp
```

*Verified:* 11 entries for a two-line function.

| entry | what it is |
| --- | --- |
| `module_0000.jit_f/module.mlir` | the StableHLO handed to XLA |
| `module_0000.jit_f/compile_options.textproto` | every compile option in effect |
| `module_0000.jit_f/topology.textproto` | the device topology compiled for |
| `module_0000.jit_f.before_optimizations.txt` | HLO as XLA received it |
| `module_0000.jit_f.cpu_after_optimizations.txt` | HLO as XLA will run it |
| `...-buffer-assignment.txt`, `...-buffer-assignment-values.txt` | where every buffer lives |
| `...-live-range.txt`, `...-memory-usage-report.txt` | liveness and peak memory |
| `...ir-no-opt.ll`, `...ir-with-opt.ll` | LLVM IR before and after LLVM's own passes |
| `...obj-file.<kernel>.o` | the object code actually executed |
| `module_0000.jit_f.debug_options` | the flags this dump was produced under |

The last three are worth knowing about: the CPU backend hands you LLVM IR and an object file
for free, which is as close to a Godbolt assembly pane as this pipeline gets. `objdump -d` on
the `.o` finishes the job.

### Per-pass dumps

The single most useful flag for compiler work, because it turns "the output changed" into
"this pass changed it":

```bash
XLA_FLAGS="--xla_dump_to=/tmp/dump --xla_dump_hlo_pass_re=.*"
```

*Verified:* 32 entries for the same function, 26 of them per-pass snapshots named

```
module_0000.jit_f.0005.simplification.after_pipeline-start.before_algsimp.txt
                  ^^^^ ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^
                  order  pipeline       state after          state before
```

The regex filters by pass name, so you can narrow to the pass you are editing rather than
reading the whole pipeline. *Verified:* `--xla_dump_hlo_pass_re=algsimp` produced 17 entries,
paired before and after each `algsimp` run in every pipeline that invokes it.

To find where two builds diverge, sort both dumps by that `NNNN` index and walk them in
lockstep. The first index whose text differs names the pass that did it. Do this before
reading any diff of final HLO: the final module reflects every downstream pass reacting to the
first change, which is why end-to-end diffs read as much larger than the actual edit.

## Comparing two XLA builds

XLA is compiled into jaxlib as a C extension, so one interpreter holds exactly one XLA. There
is no in-process A/B. Two builds means two processes, which is what jaxplorer's worker already is.

### Setup

```bash
uv venv --python 3.13 .venv-a
VIRTUAL_ENV=.venv-a uv pip install "jax==0.6.2" "jaxlib==0.6.2"
# or: VIRTUAL_ENV=.venv-a uv pip install /path/to/locally/built/jaxlib*.whl
```

The worker needs only stdlib and jax, so `PYTHONPATH` is enough to reach it and the second
environment does not need jaxplorer installed. *Verified:* jaxplorer's worker ran unmodified under jax
0.6.2 this way.

### Script

```python
"""Run one snippet through two jaxlib builds using jaxplorer's worker, then diff."""

import difflib
import json
import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path("src").resolve()
SNIPPET = """
import jax
import jax.numpy as jnp

def f(x, w):
    h = jnp.tanh(x @ w)
    return jax.nn.softmax(h, axis=-1).sum()

args = (jax.ShapeDtypeStruct((32, 64), jnp.float32),
        jax.ShapeDtypeStruct((64, 16), jnp.float32))
"""


def read_frame(stdout):
    """Responses are length-prefixed, see jaxplorer.protocol."""
    return stdout.read(int(stdout.readline()))


def run(python):
    proc = subprocess.Popen(
        [python, "-m", "jaxplorer.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={"PYTHONPATH": str(SRC), "JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"},
    )
    handshake = json.loads(read_frame(proc.stdout))
    request = {"id": 1, "source": SNIPPET, "filename": "snippet.py"}
    proc.stdin.write((json.dumps(request) + "\n").encode())
    proc.stdin.flush()
    result = json.loads(read_frame(proc.stdout))
    proc.stdin.close()
    proc.wait(timeout=10)
    return handshake, result


def normalise(hlo):
    """Drop what differs between two environments but means nothing."""
    out = []
    for line in hlo.split("\n"):
        if re.match(r"\s*\d+ [\"{]", line):  # FileNames, StackFrames and friends
            continue
        line = re.sub(r"metadata=\{[^}]*\}", "", line)
        line = re.sub(r", sharding=\{[^}]*\}", "", line)
        if line.strip():
            out.append(line.rstrip())
    return out


results = {}
for label, python in [("old", sys.argv[1]), ("new", sys.argv[2])]:
    handshake, result = run(python)
    print(f"{label}: jax {handshake['jax_version']}, fatal={result['fatal']}")
    results[label] = result

for stage in ("stablehlo", "optimized_hlo"):
    a = normalise(results["old"]["stages"][stage]["text"])
    b = normalise(results["new"]["stages"][stage]["text"])
    delta = list(difflib.unified_diff(a, b, "old", "new", lineterm="", n=1))
    print(f"\n=== {stage}: {'identical' if not delta else f'{len(delta)} diff lines'}")
    print("\n".join(delta[:40]))
```

Run it as `uv run python ab.py .venv-a/bin/python .venv/bin/python`.

### Reading the result

*Verified* between jaxlib 0.6.2 and 0.11.0:

- **StableHLO was byte-identical.** The control held, so everything below is XLA or its debug
  output, not JAX lowering.
- **Optimized HLO differed by 125 lines,** most of it `FileLocations` and `StackFrames` tables
  that newer JAX emits and 0.6 did not. Textual diffing of HLO is mostly an exercise in
  suppressing noise.
- **`cost_analysis` was the cheapest real signal:** flops 68,031 against 68,063. One scalar
  per build beats reading 100 KB of text when sweeping for regressions.

Noise to strip before a text diff means anything: `metadata={...}`, the `FileNames` /
`FunctionNames` / `FileLocations` / `StackFrames` tables, `sharding={...}`, instruction
numbering suffixes such as `%fusion.3` against `%fusion.5`, and `entry_computation_layout`
when layout is not what you are studying. Pointing both workers at one jaxplorer checkout keeps
absolute paths out of those tables in the first place.

**That list is specific to comparing two builds, and does not transfer to comparing two passes.**
Within one compile the naming is already consistent, so there is nothing line-local left to strip.
What misleads a pass-to-pass text diff is reordering and cross-computation renaming, which no
line-based normalization can reach — hence the structural mode below.

### Better: let XLA do the diff

Upstream has a semantic HLO diff, added 2025-04-05 in `xla/hlo/tools/hlo_diff`. It matches
graph structure rather than text and, per its own usage message, ignores "instruction names,
parameter ordering etc, layouts (in some instances)", which is exactly the list above.

```bash
bazel run //xla/hlo/tools/hlo_diff:hlo_diff -- \
  --first_hlo_text=/tmp/a.txt --second_hlo_text=/tmp/b.txt \
  [--ignore_shape] [--text_output=out.txt] [--html_output=out.html]
```

It takes HLO **text**, which is what jaxplorer already has in the pane and what `--xla_dump_to`
already wrote, so no protos are needed to use it. It is an `xla_cc_binary`, built from an XLA
checkout with bazel: no jaxlib wheel ships it, no jax version exposes it, and there is no
version floor that changes that. *Verified against jaxlib 0.11.0:* no CLI binaries in the
wheel, no `HloDiff` symbols in `_jax.so`, `_xla.so` or `libjax_common.dylib`, nothing
diff-related in `jaxlib.xla_client` or `jaxlib._jax`, and no Python bindings in the tool's
`BUILD`.

If you are working on XLA you have the checkout, so prefer this over anything below: it has the
real cost model and match provenance.

### Failing that: jaxplorer's own structural diff

`f4` switches the Passes pane between a text diff and a structural one. The structural mode parses
each snapshot's text into a graph (`jaxplorer.hlograph`) and matches the two graphs before reporting
(`jaxplorer.hlodiff`), which is a deliberately small subset of what upstream does — no cost model,
no HTML output, no match provenance.

Fingerprints ignore metadata, layout and numbering suffixes, and walk operands rather than trusting
printed order, so a rescheduling pass reports nothing at all and a rewrite reports the instructions
it replaced instead of every line it moved.

Text remains the default, for two reasons. For a local change the unified diff is the better view:
seeing which four lines `algsimp` dropped beats a count of them. And without an XLA checkout there
is no oracle to check the matcher against, so the text diff stays available as the view that cannot
be wrong. When a module does not parse cleanly the structural mode declines rather than guessing,
and the pane says so and shows a text diff instead.

## Traps

**Compare like with like.** jax and jaxlib move together and JAX's lowering changes between
releases, so a difference in StableHLO means you changed two things at once. Hold `jax` fixed
and vary only `jaxlib` when the question is about XLA.

**GPU output is not reproducible by default.** Autotuning picks algorithms per run, so two
runs of the *same* build can produce different optimized HLO. Set
`--xla_gpu_autotune_level=0` before diffing anything on GPU. *Unverified here*, CPU only
machine.

**CPU output is reproducible.** *Verified:* two runs of the same build produced byte-identical
dumps for every file except `module_0000.jit_f.debug_options`, which records the dump path
itself. A plain `diff -r` between two dump directories is therefore meaningful on CPU.

**jaxplorer's own frames appear in the IR.** JAX records the Python stack that built each
instruction, so `worker.py` and `<frozen runpy>` show up in the `FileNames` table of the
Optimized HLO pane. That is real XLA debug metadata about how the module was traced, not
corruption, and it is another reason to strip those tables before diffing.

## Flag reference

All verified here except where noted.

| flag | effect |
| --- | --- |
| `--xla_dump_to=DIR` | dump the endpoints, LLVM IR, object file and buffer analyses |
| `--xla_dump_hlo_pass_re=.*` | add a before/after HLO snapshot around every pass |
| `--xla_dump_hlo_pass_re=NAME` | the same, restricted to matching passes |
| `--xla_gpu_autotune_level=0` | make GPU compilation reproducible (*unverified here*) |

jaxplorer-side equivalents: `JAX_PLATFORMS` and `JAX_ENABLE_X64` are set for you by `--platform`
and `--x64`. The worker inherits the rest of the environment, so any flag above can be set
where you launch jaxplorer. *Verified:* `XLA_FLAGS="--xla_dump_to=DIR" uv run jaxplorer mlp`
dumps from inside the worker while the TUI runs.
