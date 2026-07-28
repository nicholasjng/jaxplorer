"""The jaxplorer TUI: source on the left, every IR JAX will give you on the right.

The app owns a :class:`~jaxplorer.session.WorkerSession` and renders whatever it returns; all
compiling happens out of process.
"""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import var
from textual.widgets import Footer, Header, Input, Static, TabbedContent, TabPane, Tabs, TextArea

from jaxplorer.hlo import DebugInfo, pass_report, strip_debug_tables
from jaxplorer.protocol import ALL_STAGES, PANE_TITLES, CompileResult, Pane, Stage
from jaxplorer.session import WorkerSession, WorkerStartupError
from jaxplorer.watch import POLL_INTERVAL, FileWatcher

PANES: tuple[Pane, ...] = (*ALL_STAGES, "passes", "llvm_ir", "errors")

# Panes whose text is an HLO module, so the debug-table filter and click-to-source apply.
HLO_PANES: frozenset[Pane] = frozenset({"optimized_hlo", "passes"})

# Long enough that ordinary typing does not trigger a compile, short enough that a
# pause feels immediate.
DEBOUNCE = 0.35

PLATFORM_CYCLE = ("cpu", "gpu", "tpu")

MATCH_STYLE = "black on yellow"

# Short names for the status line, where four timings plus a total have to fit.
STAGE_ABBREV: dict[Stage, str] = {
    "jaxpr": "jaxpr",
    "stablehlo": "shlo",
    "optimized_hlo": "hlo",
    "analysis": "cost",
}

PLACEHOLDER = """import jax
import jax.numpy as jnp


def f(x, w):
    return jnp.tanh(x @ w).sum()


args = (
    jax.ShapeDtypeStruct((8, 16), jnp.float32),
    jax.ShapeDtypeStruct((16, 4), jnp.float32),
)
"""


class SnippetEditor(TextArea):
    """The source editor.

    Undo and redo are re-declared only to flip Textual's ``show`` flag: they work either
    way, but nobody reaching for undo should have to guess the key.

    See Also
    --------
    textual.widgets.TextArea : Everything else about this widget's behavior.
    """

    BINDINGS: ClassVar = [
        Binding("ctrl+z,super+z", "undo", "Undo"),
        Binding("ctrl+y,super+y", "redo", "Redo"),
    ]


class OutputPane(ScrollableContainer):
    """A scrollable, unwrapped view of one pane's text.

    Holds the raw text and re-renders it through the metadata filter and the search
    highlight, so callers never reapply either themselves.

    The single-letter bindings are safe here and nowhere else, since they only fire while a
    pane has focus rather than the editor.

    Parameters
    ----------
    pane : Pane
        Which pane this is. Sets the widget ids and decides whether the HLO filter and
        click-to-source apply.

    Attributes
    ----------
    raw : str
        Text as the worker sent it, before filtering.
    """

    BINDINGS: ClassVar = [
        Binding("slash", "find", "Find"),
        Binding("n", "next_match", "Next match", show=False),
        Binding("N", "prev_match", "Previous match", show=False),
        Binding("y", "copy", "Copy"),
    ]

    class Clicked(Message):
        """A line of this pane was clicked.

        Attributes
        ----------
        pane : Pane
            Pane that was clicked.
        line : int
            0-based index into the pane's *displayed* lines, which is not the raw text's
            numbering while the metadata filter is on.
        """

        def __init__(self, pane: Pane, line: int) -> None:
            super().__init__()
            self.pane = pane
            self.line = line

    def __init__(self, pane: Pane) -> None:
        super().__init__(id=f"pane-{pane}")
        self.pane = pane
        self.raw = ""
        self._body = Static("", markup=False, id=f"body-{pane}")

    def compose(self) -> ComposeResult:
        """Yield the body this pane scrolls."""
        yield self._body

    @property
    def displayed(self) -> str:
        """The text as shown, which is what search and correlation must agree with."""
        app = self.app
        hide = not isinstance(app, JaxplorerApp) or not app.show_metadata
        if hide and self.pane in HLO_PANES and self.raw:
            return strip_debug_tables(self.raw)
        return self.raw

    @property
    def lines(self) -> list[str]:
        """The displayed text, split into lines."""
        return self.displayed.split("\n")

    def show(self, text: str) -> None:
        """Replace the pane's contents.

        Parameters
        ----------
        text : str
            Raw text from the worker. Any active search highlight is dropped.
        """
        self.raw = text
        self.refresh_text()

    def refresh_text(self, query: str = "") -> None:
        """Re-render the body, optionally highlighting every occurrence of ``query``."""
        # A Text object rather than a markup string: HLO is full of f32[8,16], which Rich
        # would otherwise try to parse.
        body = Text(self.displayed)
        if query:
            body.highlight_regex(re.escape(query), style=MATCH_STYLE)
        self._body.update(body)

    def match_lines(self, query: str) -> list[int]:
        """Return the indices of displayed lines containing ``query``, case-insensitively."""
        if not query:
            return []
        needle = query.casefold()
        return [i for i, line in enumerate(self.lines) if needle in line.casefold()]

    def on_click(self, event: events.Click) -> None:
        """Translate a click into a :class:`Clicked` message carrying the line index."""
        # The body is as tall as its content and the container scrolls it, so a y offset
        # relative to the body is already a line index.
        line = event.y if event.widget is self._body else event.y + self.scroll_offset.y
        self.post_message(self.Clicked(self.pane, line))

    @property
    def _jaxplorer(self) -> JaxplorerApp:
        app = self.app
        assert isinstance(app, JaxplorerApp)
        return app

    # Search and copy are app-wide concerns; the pane only owns the keys for them.
    def action_find(self) -> None:
        self._jaxplorer.action_find()

    def action_next_match(self) -> None:
        self._jaxplorer.action_next_match()

    def action_prev_match(self) -> None:
        self._jaxplorer.action_prev_match()

    def action_copy(self) -> None:
        self._jaxplorer.action_copy_pane()


class JaxplorerApp(App[None]):
    """The application.

    Parameters
    ----------
    path : Path, optional
        Snippet to open. ``None`` gives a scratch buffer.
    source : str, optional
        Initial buffer contents, overriding what ``path`` holds. Mostly for tests.
    watch : bool, optional
        Reload ``path`` when it changes on disk and keep the buffer read-only, so an
        external editor is the only writer.
    platform : str, optional
        Backend to compile for. ``None`` leaves the choice to JAX.
    x64 : bool, optional
        Whether to enable 64-bit values.
    stages : list of Stage, optional
        Stages to request. Defaults to all of them.
    passes : bool, optional
        Whether to collect per-pass HLO and LLVM IR from the first compile.
    timeout : float, optional
        Seconds to allow one compile.
    executable : str, optional
        Interpreter to run the worker under, so the jax being inspected can come from a
        project's virtualenv rather than jaxplorer's own environment.

    Attributes
    ----------
    status : str
        Reactive status-bar text.
    show_metadata : bool
        Reactive: whether HLO debug tables are displayed.
    session : WorkerSession
        The worker this app renders.
    """

    CSS = """
    #split { height: 1fr; }
    #editor { width: 45%; border-right: solid $panel; }
    #outputs { width: 1fr; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #search { display: none; height: 3; }
    #search.visible { display: block; }
    OutputPane { padding: 0 1; }
    /* Sizing the body to its longest line is what stops IR from being wrapped; the pane
       scrolls sideways instead. */
    OutputPane > Static { width: auto; }
    """

    BINDINGS: ClassVar = [
        ("ctrl+r", "recompile", "Recompile"),
        ("ctrl+s", "save", "Save"),
        # Also bound by SnippetEditor. These make undo work from the IR panes too, where
        # the editor is not focused.
        ("ctrl+z", "undo", "Undo"),
        ("ctrl+y", "redo", "Redo"),
        # priority=True because the focused TextArea binds these itself: f6 to select_line
        # today, and ctrl+f to delete_word_right before textual 8.2. Without it the
        # documented key silently does something else, or nothing, while you are editing.
        Binding("ctrl+f", "find", "Find", priority=True),
        ("f2", "cycle_platform", "Platform"),
        ("f3", "toggle_metadata", "Metadata"),
        Binding("f6", "toggle_passes", "Passes", priority=True),
        ("alt+1", "show_pane('jaxpr')", "Jaxpr"),
        ("alt+2", "show_pane('stablehlo')", "StableHLO"),
        ("alt+3", "show_pane('optimized_hlo')", "Opt HLO"),
        ("alt+4", "show_pane('analysis')", "Analysis"),
        ("alt+5", "show_pane('passes')", "Passes"),
        ("alt+6", "show_pane('llvm_ir')", "LLVM IR"),
        ("alt+7", "show_pane('errors')", "Errors"),
        ("ctrl+q", "quit", "Quit"),
    ]

    status: var[str] = var("starting worker …")
    show_metadata: var[bool] = var(False)

    def __init__(
        self,
        *,
        path: Path | None = None,
        source: str | None = None,
        watch: bool = False,
        platform: str | None = None,
        x64: bool = False,
        stages: list[Stage] | None = None,
        passes: bool = False,
        timeout: float = 20.0,
        executable: str | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.watch_mode = watch and path is not None
        self.session = WorkerSession(
            platform=platform, x64=x64, timeout=timeout, executable=executable
        )
        self.stages = list(stages) if stages else list(ALL_STAGES)
        self.collect_passes = passes
        if source is None:
            source = path.read_text() if path is not None else PLACEHOLDER
        self._initial_source = source
        self._watcher = FileWatcher(path) if self.watch_mode and path else None
        self._debounce_timer = None
        self._last_result: CompileResult | None = None
        self._debug_info: DebugInfo | None = None
        # Platforms whose worker refused to start, so f2 stops offering them.
        self._dead_platforms: set[str] = set()
        self._query = ""
        self._matches: list[int] = []
        self._match = 0

    # -- composition ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Lay out the editor, the output tabs, the search bar and the status bar."""
        yield Header()
        with Horizontal(id="split"):
            yield SnippetEditor.code_editor(
                self._initial_source,
                language="python",
                id="editor",
                # In watch mode the file on disk is the source of truth; a second
                # writer here would mean lost edits.
                read_only=self.watch_mode,
            )
            with TabbedContent(id="outputs"):
                for pane in PANES:
                    with TabPane(PANE_TITLES[pane], id=pane):
                        yield OutputPane(pane)
        yield Input(placeholder="find in pane", id="search")
        yield Static(self.status, id="status")
        yield Footer()

    def watch_status(self, status: str) -> None:
        """Mirror the ``status`` reactive into the status bar."""
        # The reactive is set before compose runs, so the widget may not exist yet.
        with suppress(NoMatches):
            self.query_one("#status", Static).update(status)

    def watch_show_metadata(self, _show: bool) -> None:
        """Re-render every pane when the metadata filter is toggled."""
        with suppress(NoMatches):
            for pane in self.query(OutputPane):
                pane.refresh_text(self._query if pane.pane == self.active_pane else "")

    async def on_mount(self) -> None:
        """Start watching if asked to, then compile once so the panes are never empty."""
        self.title = "jaxplorer"
        self.sub_title = str(self.path) if self.path else "scratch buffer"
        if self._watcher is not None:
            self.set_interval(POLL_INTERVAL, self._poll_file)
        self.action_recompile()

    async def on_unmount(self) -> None:
        """Shut the worker down, so quitting leaves no orphan holding a JAX backend."""
        await self.session.close()

    # -- input ------------------------------------------------------------------

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        """Arm the debounce timer so a pause in typing, not a keystroke, compiles."""
        if self.watch_mode:
            return
        # Re-arm on every keystroke so only the pause at the end costs a compile.
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._debounce_timer = self.set_timer(DEBOUNCE, self.action_recompile)

    def on_key(self, event: events.Key) -> None:
        """Move between the tab bar and the IR under it.

        The tab bar spends left and right on switching tabs, so Down is the only key
        left that reads as "into the thing this tab names", where scrolling happens.
        """
        if isinstance(self.focused, Tabs) and event.key == "down":
            with suppress(NoMatches):
                self.query_one(f"#pane-{self.active_pane}", OutputPane).focus()
                event.stop()
        elif isinstance(self.focused, OutputPane) and event.key == "escape":
            self.query_one(TabbedContent).query_one(Tabs).focus()
            event.stop()

    def _poll_file(self) -> None:
        if self._watcher is None:
            return
        source = self._watcher.poll()
        if source is None:
            return
        # load_text, not .text, so undo history is reset along with the buffer.
        self.query_one("#editor", TextArea).load_text(source)
        self.action_recompile()

    # -- search -----------------------------------------------------------------

    def action_find(self) -> None:
        """Open the search bar, keeping any previous query."""
        search = self.query_one("#search", Input)
        search.add_class("visible")
        search.value = self._query
        search.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Search as the query is typed."""
        self._apply_query(event.value)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Close the search bar and hand focus to the pane, keeping the highlight."""
        self._close_search()
        with suppress(NoMatches):
            self.query_one(f"#pane-{self.active_pane}", OutputPane).focus()

    def _close_search(self) -> None:
        self.query_one("#search", Input).remove_class("visible")

    def _apply_query(self, query: str) -> None:
        self._query = query
        pane = self.query_one(f"#pane-{self.active_pane}", OutputPane)
        pane.refresh_text(query)
        self._matches = pane.match_lines(query)
        self._match = 0
        if self._matches:
            self._scroll_to_match(pane)
        self._report_matches()

    def _scroll_to_match(self, pane: OutputPane) -> None:
        # A little context above the hit rather than pinning it to the top edge.
        pane.scroll_to(y=max(0, self._matches[self._match] - 2), animate=False)

    def _report_matches(self) -> None:
        if not self._query:
            return
        total = len(self._matches)
        where = f"{self._match + 1}/{total}" if total else "no match"
        self.status = f"find {self._query!r}: {where}"

    def _step_match(self, delta: int) -> None:
        if not self._matches:
            return
        self._match = (self._match + delta) % len(self._matches)
        self._scroll_to_match(self.query_one(f"#pane-{self.active_pane}", OutputPane))
        self._report_matches()

    def action_next_match(self) -> None:
        """Scroll to the next match, wrapping at the end."""
        self._step_match(1)

    def action_prev_match(self) -> None:
        """Scroll to the previous match, wrapping at the start."""
        self._step_match(-1)

    def on_tabbed_content_tab_activated(self, _event: TabbedContent.TabActivated) -> None:
        """Re-run the active query against the newly shown pane."""
        # A query follows the user to whichever pane they switch to.
        if self._query:
            self._apply_query(self._query)

    # -- actions ----------------------------------------------------------------

    def action_recompile(self) -> None:
        """Compile the buffer now, cancelling any compile already in flight."""
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer = None
        source = self.query_one("#editor", TextArea).text
        self.status = "compiling …"
        # exclusive=True cancels any in-flight compile; the UI tracks only the newest.
        self.run_worker(self._compile(source), group="compile", exclusive=True)

    async def _compile(self, source: str) -> None:
        try:
            result = await self.session.compile(
                source,
                filename=self.compile_filename,
                stages=self.stages,
                passes=self.collect_passes,
            )
        except Exception as exc:  # noqa: BLE001 - a jaxplorer bug must not kill the session
            self._render(CompileResult(id=0, fatal=f"jaxplorer failed to compile: {exc!r}"))
            return
        if result is None:
            return  # superseded by a newer request
        self._render(result)

    @property
    def compile_filename(self) -> str:
        """Name the buffer is compiled under, which tracebacks and HLO metadata report."""
        return str(self.path) if self.path else "<buffer>"

    def action_save(self) -> None:
        """Write the buffer back to its file, if it has one and owns it."""
        if self.path is None:
            self.notify("no file backing this buffer", severity="warning")
            return
        if self.watch_mode:
            self.notify("buffer is read-only in --watch mode", severity="warning")
            return
        self.path.write_text(self.query_one("#editor", TextArea).text)
        self.notify(f"saved {self.path}")

    def action_toggle_metadata(self) -> None:
        """Show or hide the HLO debug tables."""
        self.show_metadata = not self.show_metadata
        self.notify(f"HLO debug tables {'shown' if self.show_metadata else 'hidden'}")

    def action_toggle_passes(self) -> None:
        """Turn per-pass collection on or off, then recompile to apply it."""
        self.collect_passes = not self.collect_passes
        self.notify(f"pass snapshots {'on' if self.collect_passes else 'off'}")
        self.action_recompile()

    def action_copy_pane(self) -> None:
        """Copy the active pane's displayed text to the clipboard."""
        pane = self.query_one(f"#pane-{self.active_pane}", OutputPane)
        self.copy_to_clipboard(pane.displayed)
        self.notify(f"copied {PANE_TITLES[pane.pane]} to the clipboard")

    def action_cycle_platform(self) -> None:
        """Switch to the next backend, skipping any that already failed to start here."""
        current = self.session.platform or "cpu"
        options = [p for p in PLATFORM_CYCLE if p not in self._dead_platforms or p == current]
        if len(options) < 2:
            self.notify("no other backend is available here", severity="warning")
            return
        start = options.index(current) if current in options else -1
        self.session.platform = options[(start + 1) % len(options)]
        self.status = f"switching to {self.session.platform} …"
        self.run_worker(self._restart(), group="compile", exclusive=True)

    async def _restart(self) -> None:
        # A platform change only takes effect before JAX is imported, hence a fresh worker.
        try:
            await self.session.restart()
        except WorkerStartupError as exc:  # e.g. no such backend on this machine
            # Remembering the failure is what stops f2 walking into it again.
            self._dead_platforms.add(self.session.platform or "")
            self._show("errors", str(exc))
            self.query_one(TabbedContent).active = "errors"
            self.status = f"{self.session.platform} unavailable"
            return
        self.action_recompile()

    def action_show_pane(self, pane: str) -> None:
        """Bring ``pane`` to the front.

        Parameters
        ----------
        pane : str
            A :data:`~jaxplorer.protocol.Pane` name, which is also the tab's id.
        """
        self.query_one(TabbedContent).active = pane

    @property
    def active_pane(self) -> str:
        """Id of the pane currently on top."""
        return self.query_one(TabbedContent).active

    def action_undo(self) -> None:
        """Undo in the editor, unless the file on disk owns the history."""
        if not self.watch_mode:
            self.query_one("#editor", TextArea).undo()

    def action_redo(self) -> None:
        """Redo in the editor, unless the file on disk owns the history."""
        if not self.watch_mode:
            self.query_one("#editor", TextArea).redo()

    # -- click to source --------------------------------------------------------

    def on_output_pane_clicked(self, event: OutputPane.Clicked) -> None:
        """Select the source line that produced the clicked HLO instruction.

        Silent for panes that hold no HLO, and for lines XLA recorded no stack frame for,
        which is most of a module's structural text.
        """
        if event.pane not in HLO_PANES or self._debug_info is None:
            return
        lines = self.query_one(f"#pane-{event.pane}", OutputPane).lines
        if not 0 <= event.line < len(lines):
            return
        ref = self._debug_info.locate(lines[event.line], prefer=self.compile_filename)
        if ref is None:
            self.status = "no source recorded for that line"
            return
        editor = self.query_one("#editor", TextArea)
        row = max(0, ref.line - 1)
        editor.move_cursor((row, 0))
        editor.select_line(row)
        self.status = str(ref)

    # -- rendering --------------------------------------------------------------

    def _show(self, pane: Pane, text: str) -> None:
        self.query_one(f"#pane-{pane}", OutputPane).show(text)

    def _render(self, result: CompileResult) -> None:
        self._last_result = result

        # A fatal means the buffer never ran, the normal state halfway through an edit.
        # Blanking every pane on an unfinished keystroke would destroy the context the
        # user is working against, so keep the last IR that did compile.
        if result.fatal is None:
            for stage in ALL_STAGES:
                outcome = result.stages.get(stage)
                if outcome is None:
                    body = f"(not requested; --stages {','.join(self.stages)})"
                elif outcome.error:
                    body = outcome.error
                elif outcome.skipped:
                    body = "(skipped: an earlier stage failed, see Errors)"
                else:
                    body = outcome.text or ""
                self._show(stage, body)
                self._mark_tab(stage, failed=outcome is not None and bool(outcome.error))

            hlo = result.stages.get("optimized_hlo")
            self._debug_info = DebugInfo.parse(hlo.text) if hlo and hlo.text else None
            self._show("passes", pass_report(result.passes))
            self._show("llvm_ir", result.llvm_ir or _NO_LLVM_IR)

        errors = result.errors()
        # A jax too old for a feature (notably click-to-source) is a warning, not a failure.
        warning = self.session.info.warning if self.session.info else None
        notes = [f"[environment]\n{warning}"] if warning else []
        shown = errors + notes
        self._show("errors", "\n\n".join(shown) if shown else "No errors.")
        self._mark_tab("errors", failed=bool(errors))

        if result.fatal:
            self.query_one(TabbedContent).active = "errors"

        self.status = self._status_line(result, errors)

    def _status_line(self, result: CompileResult, errors: list[str]) -> str:
        """Compose the status bar: backend, per-stage timings, totals, counts."""
        info = self.session.info
        parts = [info.summary() if info else "no worker"]
        # Per-stage timings, since "which half is slow" is the first question about a
        # compile that feels sluggish.
        timings = " ".join(
            f"{STAGE_ABBREV[stage]} {result.stages[stage].elapsed_ms:.0f}ms"
            for stage in ALL_STAGES
            if stage in result.stages and result.stages[stage].ok
        )
        if timings:
            parts.append(timings)
        parts.append(f"total {result.total_ms:.0f} ms")
        parts.append(f"{len(errors)} error(s)" if errors else "ok")
        if result.passes:
            parts.append(f"{len(result.passes)} passes")
        if self.watch_mode:
            parts.append("watching")
        return " · ".join(parts)

    def _mark_tab(self, pane: Pane, *, failed: bool) -> None:
        """Flag a tab whose stage failed, so a problem is visible without hunting."""
        title = PANE_TITLES[pane]
        tab = self.query_one(TabbedContent).get_tab(pane)
        tab.label = f"{title} ●" if failed else title  # type: ignore[assignment]


_NO_LLVM_IR = (
    "No LLVM IR.\n\nRun jaxplorer with --passes, or press f6, on a backend that emits it "
    "(the CPU backend does)."
)
