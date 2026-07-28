# jaxplorer

A compiler explorer TUI for JAX. Put a jitted function on the left, watch its jaxpr,
StableHLO and optimized HLO on the right, and see all three change as you type.

```
┌─ source ─────────┬─ Jaxpr │ StableHLO │ Optimized HLO │ Analysis │ Passes │ LLVM IR ─┐
│ def f(x, w):              │ { lambda ; a:f32[8,16] b:f32[16,4]. let                 │
│     return jnp.tanh(x @ w)│     c:f32[8,4] = dot_general[...] a b                   │
│                           │     d:f32[8,4] = tanh c                                 │
└───────────────────────────┴────────────────────────────────────────────────────────┘
 cpu · jax 0.11.0 · 134 ms · ok
```

## Installation

```bash
uv sync                             # in a checkout
uv tool install "jaxplorer[jax]"    # standalone, with its own jax
uv tool install jaxplorer           # standalone, borrowing a project's jax (see below)
```

Needs Python 3.12+ and `textual >= 6.0`.
`jax >= 0.9` is an extra, since the jax worth inspecting is usually a project's own.
Any CPU-only jax install is enough; a GPU or TPU backend is only needed to compile for one.

## Usage

```bash
uv run jaxplorer                       # scratch buffer
uv run jaxplorer mlp                   # open a bundled example (mlp, scan, attention)
uv run jaxplorer my_model.py           # open a snippet and edit it in place
uv run jaxplorer my_model.py --watch    # keep editing in your own editor; jaxplorer reloads on save
uv run jaxplorer mlp --print optimized_hlo   # print one pane and exit, no TUI
```

| key | |
| --- | --- |
| `f1` or `?` | list every key (the footer only fits a few) |
| `ctrl+r` | recompile now |
| `ctrl+s` | save the buffer |
| `ctrl+z`, `ctrl+y` | undo, redo (`cmd+z` / `cmd+y` also work) |
| `ctrl+f` or `/` | find in the active pane; `n` / `N` cycle the hits, `escape` clears |
| `f2` | switch backend (skips ones that already failed here) |
| `f3` | show or hide the HLO debug tables |
| `f4` | diff pass snapshots as graphs instead of as text |
| `f6` | collect per-pass HLO and LLVM IR, then recompile |
| `alt+1` … `alt+7` | jump to a pane |
| `down` | from the tab bar into the IR, then arrows scroll it |
| `]`, `[` | next, previous section in the Passes pane |
| `y` | copy the active pane to the clipboard |
| `escape` | from the IR back to the tab bar |
| `ctrl+q` | quit |

Click an instruction in the Optimized HLO or Passes pane to select the source line that
produced it.

Other options: `--version`, `--python PATH` (compile under another environment's jax, see
below), `--platform cpu|gpu|tpu`, `--x64`, `--timeout SECONDS`,
`--stages jaxpr,stablehlo,...` (the chain stops after the last stage asked for, so leaving out
`optimized_hlo` skips XLA — most of a compile on a large model), `--passes` to collect per-pass
HLO and LLVM IR from the start, `--structural-diff` to start with `f4` on,
`--print PANE` to write one pane to stdout and exit instead of starting the TUI, and
`--examples` to list the bundled snippets.

## Against your own project's jax

jaxplorer compiles in a subprocess, and that subprocess can be your project's interpreter:

```bash
cd my-project
uv run jaxplorer model.py                     # lookup via uv (VIRTUAL_ENV)
jaxplorer model.py --python .venv/bin/python   # or name the interpreter outright.
```

The interpreter is chosen in this order: `--python`, then `$VIRTUAL_ENV`, then the one running
jaxplorer. `uv run` and a plain `activate` both export `VIRTUAL_ENV`, so inside a project the
flag is rarely needed.

## Anatomy of a snippet

A snippet is an ordinary Python module that defines a callable `f` and a tuple `args` of
example inputs. Nothing needs to import jaxplorer.

```python
import jax
import jax.numpy as jnp


def f(x, w):
    return jnp.tanh(x @ w).sum()


args = (
    jax.ShapeDtypeStruct((8, 16), jnp.float32),
    jax.ShapeDtypeStruct((16, 4), jnp.float32),
)
```

`args` may hold concrete arrays or `jax.ShapeDtypeStruct` specs. jaxplorer only traces and
compiles `f`, never runs it, so shape specs are enough. Optionally define `kwargs`,
`static_argnums`, `static_argnames` or `donate_argnums`; they are passed to `jax.jit`. Three
examples ship in the wheel — `jaxplorer mlp`, `jaxplorer scan`, `jaxplorer attention` — for an
MLP, a `lax.scan` loop, and causal attention with a static argument. They open as a scratch
buffer, so editing one cannot write over your installation.

## How it works

Each pane is one step of JAX's public lowering chain:

| pane | source |
| --- | --- |
| Jaxpr | `jax.jit(f).trace(*args).jaxpr` |
| StableHLO | `.lower().as_text()` |
| Optimized HLO | `.compile().as_text()`, after XLA's optimization passes |
| Analysis | `.cost_analysis()` and `.memory_analysis()` |
| Passes | a snapshot between every XLA pass, diffed to show which pass changed what |
| LLVM IR | the CPU backend's LLVM IR, after LLVM's own passes |

Stages are reported independently, so a lowering failure still leaves you a valid jaxpr to
read, and a buffer that does not even parse keeps the last IR that did compile on screen.

Compilation happens in a subprocess (`python -m jaxplorer.worker`) that stays warm between edits:
JAX takes seconds to boot, XLA can abort the process outright, and the platform and `x64` flags can only be set before JAX is imported.

For the rest of the pipeline (per-pass HLO dumps, LLVM IR, object code, and comparing two XLA
builds against each other) see [docs/xla-introspection.md](https://github.com/nicholasjng/jaxplorer/blob/master/docs/xla-introspection.md).

**jaxplorer executes the buffer.** It is your own code in your own environment, but a snippet is
run at module level on every recompile, so treat it the way you would treat `python snippet.py`.

## Development

Contributions welcome. Here's the general testing and formatting workflow of the repo:

```bash
uv run --group test pytest
uvx prek run --all-files
uv run --group typing ty check
```
