# Changelog

All notable changes to jaxplorer are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Bundled examples: `mlp`, `scan` and `attention` now ship inside the wheel, so
  `jaxplorer mlp` opens one from anywhere, with no git checkout required. They open as a
  scratch buffer, so editing one cannot write over the installed copy. `--examples` lists
  the bundled names.
- `--print <pane>` compiles once, writes a single pane to stdout, and exits without
  starting the TUI, for piping and scripting.
- Structural pass diff: `jaxplorer.hlograph` parses HLO text into a graph and
  `jaxplorer.hlodiff` matches two such graphs, so a rescheduling pass reports nothing and a
  rewrite reports only the instructions it touched, instead of both reading as a large line
  diff. Toggle with `f4` in the TUI, or start with it on via `--structural-diff`. Falls back
  to the text diff, with a note, when a module's text does not parse cleanly enough to trust.
- `f1` / `?` opens a full list of keybindings, since the footer only fits a few.
- `]` / `[` step to the next/previous section in the Passes pane.
- The TUI now fits a standard 80-column terminal.
- `CompileResult.stages_run` reports which stages actually executed, distinct from which
  were asked for.

### Changed

- `--stages` now stops the compile chain after the last stage asked for, instead of running
  every stage regardless and only hiding the ones not requested. Leaving out `optimized_hlo`
  really skips XLA now, which is most of a compile on a large model.
- Search highlighting is capped at 2000 matches and the current match is visually
  distinguished, cutting `n` / `N` and `]` / `[` latency from roughly 200ms to 24ms on large
  panes.
- The Passes report is re-rendered only when it is stale and visible, instead of on every
  keystroke.
- Examples moved from a top-level `examples/` directory into `src/jaxplorer/examples`, so
  they are bundled with the package.

### Fixed

- A worker that fails to start because `--python` cannot be executed as an interpreter (e.g.
  it names a directory) now reports a clear startup error instead of an unhandled `OSError`.
  `CompileResult.startup_failed` distinguishes this from an ordinary snippet failure in a
  healthy worker.
- XLA dump directories left behind by a worker killed with `SIGKILL` (which runs no cleanup)
  are now reclaimed: immediately by the session for the worker it just killed, and by a sweep
  of anything older than an hour when a new worker starts.

## [0.1.0] - 2026-07-29

Initial release.
