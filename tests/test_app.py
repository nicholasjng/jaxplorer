"""TUI tests driven through Textual's pilot."""

import asyncio
from dataclasses import replace
from typing import cast

from rich.text import Text
from textual.widgets import HelpPanel, Input, Static, TabbedContent, Tabs, TextArea

from jaxplorer.app import JaxplorerApp, OutputPane
from jaxplorer.protocol import CompileResult


def make_app(source: str, **kwargs) -> JaxplorerApp:
    return JaxplorerApp(source=source, platform="cpu", **kwargs)


async def settle(
    app: JaxplorerApp,
    *,
    pilot,
    after: CompileResult | None = None,
    timeout: float = 60.0,
) -> CompileResult:
    """Wait until a compile newer than ``after`` has landed in the panes.

    ``after`` must be captured *before* the edit that triggers the recompile: a small
    snippet compiles in milliseconds, so the result can arrive while the test is still
    waiting out the debounce.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        result = app._last_result
        if result is not None and result is not after:
            return result
        await asyncio.sleep(0.05)
    raise AssertionError(f"compile never completed (status: {app.status})")


def pane_text(app: JaxplorerApp, pane: str) -> str:
    return str(app.query_one(f"#body-{pane}", Static).content)


def status_text(app: JaxplorerApp) -> str:
    return str(app.query_one("#status", Static).content)


async def test_startup_populates_every_pane(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        assert "dot_general" in pane_text(app, "jaxpr")
        assert "stablehlo" in pane_text(app, "stablehlo")
        assert "ENTRY" in pane_text(app, "optimized_hlo")
        assert "Cost:" in pane_text(app, "analysis")
        assert pane_text(app, "errors") == "No errors."


async def test_status_line_reports_backend_and_timing(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        status = status_text(app)
        assert "cpu" in status
        assert "jax " in status
        assert "ms" in status
        assert "ok" in status


async def test_editing_the_buffer_recompiles(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        first = await settle(app, pilot=pilot)
        assert "tanh" in pane_text(app, "jaxpr")

        # load_text posts the same Changed message that typing does, which arms the
        # debounce timer.
        app.query_one("#editor", TextArea).load_text(snippet.replace("jnp.tanh", "jnp.sin"))
        await settle(app, pilot=pilot, after=first)

        assert "sin" in pane_text(app, "jaxpr")
        assert "tanh" not in pane_text(app, "jaxpr")


async def test_a_broken_snippet_surfaces_on_the_errors_tab(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        first = await settle(app, pilot=pilot)

        app.query_one("#editor", TextArea).load_text("def f(:\n")
        await settle(app, pilot=pilot, after=first)

        tabs = app.query_one(TabbedContent)
        assert tabs.active == "errors"
        assert "SyntaxError" in pane_text(app, "errors")
        assert "Errors ●" in str(tabs.get_tab("errors").label)
        # The last IR that did compile is kept: an unfinished edit should not wipe the
        # context the user is working against.
        assert "dot_general" in pane_text(app, "jaxpr")


async def test_a_failing_stage_flags_its_tab(snippet):
    app = make_app(snippet.replace("(16, 4)", "(17, 4)"))
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        tabs = app.query_one(TabbedContent)
        assert "●" in str(tabs.get_tab("jaxpr").label)
        assert "●" not in str(tabs.get_tab("stablehlo").label)
        assert "skipped" in pane_text(app, "stablehlo")
        assert "contracting dimensions" in pane_text(app, "errors")


async def test_recovering_from_an_error_clears_the_tab_flag(snippet):
    app = make_app("def f(:\n")
    async with app.run_test() as pilot:
        first = await settle(app, pilot=pilot)
        tabs = app.query_one(TabbedContent)
        assert "●" in str(tabs.get_tab("errors").label)

        app.query_one("#editor", TextArea).load_text(snippet)
        await settle(app, pilot=pilot, after=first)

        assert "●" not in str(tabs.get_tab("errors").label)
        assert pane_text(app, "errors") == "No errors."
        assert "dot_general" in pane_text(app, "jaxpr")


async def test_ir_is_scrolled_sideways_rather_than_wrapped(snippet):
    app = make_app(snippet)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot=pilot)

        app.query_one("#outputs", TabbedContent).active = "optimized_hlo"
        await pilot.pause()

        pane = app.query_one("#pane-optimized_hlo")
        # Wrapping IR makes it unreadable, so the body keeps its natural width.
        assert pane.virtual_size.width > pane.container_size.width
        assert pane.show_horizontal_scrollbar


async def test_pane_bindings_switch_tabs(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        tabs = app.query_one(TabbedContent)

        await pilot.press("alt+3")
        assert tabs.active == "optimized_hlo"
        await pilot.press("alt+1")
        assert tabs.active == "jaxpr"


async def test_undo_and_redo_reach_the_editor(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        editor = app.query_one("#editor", TextArea)
        editor.focus()
        await pilot.pause()

        await pilot.press("x", "y", "z")
        await pilot.pause()
        assert editor.text != snippet

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert editor.text == snippet

        await pilot.press("ctrl+y")
        await pilot.pause()
        assert editor.text != snippet


async def test_the_footer_advertises_undo_while_editing(snippet):
    app = make_app(snippet)
    async with app.run_test(size=(240, 20)) as pilot:
        await settle(app, pilot=pilot)

        # TextArea ships undo with show=False, so without SnippetEditor re-declaring it
        # the footer hides the key exactly when someone is editing and wants it.
        for target in (app.query_one("#editor"), app.query_one("#pane-jaxpr")):
            target.focus()
            await pilot.pause()
            footer = "".join(segment.text for segment in app.screen._compositor.render_strips()[-1])
            assert "Undo" in footer, f"footer hid undo with {type(target).__name__} focused"


async def test_undo_is_a_no_op_in_watch_mode(tmp_path, snippet):
    path = tmp_path / "snippet.py"
    path.write_text(snippet)

    app = JaxplorerApp(path=path, watch=True, platform="cpu")
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        app.action_undo()
        await pilot.pause()

        assert app.query_one("#editor", TextArea).text == snippet


async def test_down_moves_from_the_tab_bar_into_the_ir(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        tab_bar = app.query_one(TabbedContent).query_one(Tabs)
        tab_bar.focus()
        await pilot.pause()

        await pilot.press("down")
        assert app.focused is app.query_one("#pane-jaxpr")

        # Escape is the way back, since the pane wants the arrow keys for scrolling.
        await pilot.press("escape")
        assert app.focused is tab_bar


async def test_down_follows_the_selected_tab(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        app.query_one(TabbedContent).query_one(Tabs).focus()
        await pilot.pause()

        await pilot.press("right")  # the tab bar still owns left and right
        assert app.active_pane == "stablehlo"

        await pilot.press("down")
        assert app.focused is app.query_one("#pane-stablehlo")


async def test_a_focused_pane_scrolls_with_the_arrow_keys(snippet):
    app = make_app(snippet)
    async with app.run_test(size=(80, 14)) as pilot:
        await settle(app, pilot=pilot)
        # The HLO is the one pane guaranteed to overflow a small window.
        app.query_one("#outputs", TabbedContent).active = "optimized_hlo"
        app.query_one(TabbedContent).query_one(Tabs).focus()
        await pilot.pause()

        await pilot.press("down")  # into the pane
        pane = app.query_one("#pane-optimized_hlo")
        assert app.focused is pane
        assert pane.scroll_offset.y == 0

        for _ in range(4):
            await pilot.press("down")
        await pilot.pause()
        assert pane.scroll_offset.y > 0


async def test_hlo_debug_tables_are_hidden_until_asked_for(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        pane = app.query_one("#pane-optimized_hlo", OutputPane)

        # They routinely double the length of the module and name jaxplorer's own worker.
        assert "StackFrames" in pane.raw
        assert "StackFrames" not in pane.displayed
        assert "ENTRY" in pane.displayed

        await pilot.press("f3")
        await pilot.pause()
        assert "StackFrames" in pane.displayed

        await pilot.press("f3")
        await pilot.pause()
        assert "StackFrames" not in pane.displayed


async def test_search_highlights_counts_and_cycles(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        app.query_one("#outputs", TabbedContent).active = "optimized_hlo"
        # With the editor focused, which is where a user actually is: TextArea binds ctrl+f
        # itself, so only a priority binding gets the key.
        app.query_one("#editor", TextArea).focus()
        await pilot.pause()

        await pilot.press("ctrl+f")
        await pilot.pause()
        search = app.query_one("#search", Input)
        assert search.has_class("visible")

        search.value = "f32"
        await pilot.pause()
        total = len(app._matches)
        assert total > 1
        assert f"1/{total}" in status_text(app)

        app.action_next_match()
        assert f"2/{total}" in status_text(app)
        app.action_prev_match()
        assert f"1/{total}" in status_text(app)
        # Wrapping backwards from the first hit lands on the last.
        app.action_prev_match()
        assert f"{total}/{total}" in status_text(app)


def highlighted(app: JaxplorerApp, pane: str) -> int:
    """How many spans the search actually painted, as opposed to claimed to find."""
    body = cast("Text", app.query_one(f"#body-{pane}", Static).content)
    return sum(1 for span in body.spans if "yellow" in str(span.style))


async def test_an_uppercase_query_highlights_the_lowercase_hits_it_matched(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        app.action_show_pane("optimized_hlo")
        await pilot.pause()

        # Matching casefolds, so this finds lines; highlighting has to agree, or the pane
        # scrolls to a hit and shows nothing.
        app._apply_query("TANH")

        assert app._matches
        assert highlighted(app, "optimized_hlo") > 0


async def test_matches_do_not_outlive_the_text_they_indexed(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        result = await settle(app, pilot=pilot)
        app.action_show_pane("optimized_hlo")
        await pilot.pause()
        app._apply_query("tanh")
        assert app._matches

        # Stand in for a recompile that produced a much shorter module.
        app._show("optimized_hlo", "one\ntwo\nthree\n")
        app._render(result)
        await pilot.pause()

        lines = len(app.query_one("#pane-optimized_hlo", OutputPane).lines)
        assert all(index < lines for index in app._matches)
        # And the highlight the re-render dropped is back, so n/N mean something.
        assert not app._matches or highlighted(app, "optimized_hlo") > 0


async def test_escape_abandons_the_search(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        app.action_show_pane("optimized_hlo")
        await pilot.pause()
        await pilot.press("ctrl+f")
        await pilot.pause()
        app.query_one("#search", Input).value = "f32"
        await pilot.pause()
        assert app._matches

        await pilot.press("escape")
        await pilot.pause()

        assert not app.query_one("#search", Input).has_class("visible")
        assert app._query == ""
        assert app._matches == []
        assert highlighted(app, "optimized_hlo") == 0
        # Focus lands somewhere useful rather than on the hidden input.
        assert isinstance(app.focused, OutputPane)


async def test_search_scrolls_the_pane_to_the_hit(snippet):
    app = make_app(snippet)
    async with app.run_test(size=(80, 14)) as pilot:
        await settle(app, pilot=pilot)
        app.query_one("#outputs", TabbedContent).active = "optimized_hlo"
        await pilot.pause()
        pane = app.query_one("#pane-optimized_hlo", OutputPane)

        app._apply_query("ROOT")
        await pilot.pause()
        assert app._matches
        # The last hit is below the fold, so finding it has to move the pane.
        while app._match < len(app._matches) - 1:
            app.action_next_match()
        await pilot.pause()
        assert pane.scroll_offset.y > 0


async def test_a_query_with_no_match_says_so(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        app._apply_query("definitely-not-in-any-hlo")
        assert "no match" in status_text(app)


async def test_the_query_follows_a_tab_switch(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        app._apply_query("tanh")
        hlo_matches = len(app._matches)

        app.query_one("#outputs", TabbedContent).active = "jaxpr"
        await pilot.pause()

        assert app._matches != [] or hlo_matches == 0
        assert "tanh" in status_text(app)


async def test_clicking_an_hlo_instruction_selects_the_source_line(tmp_path, snippet):
    path = tmp_path / "snippet.py"
    path.write_text(snippet)

    app = JaxplorerApp(path=path, platform="cpu")
    async with app.run_test(size=(160, 40)) as pilot:
        await settle(app, pilot=pilot)
        app.query_one("#outputs", TabbedContent).active = "optimized_hlo"
        await pilot.pause()
        pane = app.query_one("#pane-optimized_hlo", OutputPane)

        assert app._debug_info is not None
        target = next(
            i for i, line in enumerate(pane.lines) if app._debug_info.locate(line, prefer=str(path))
        )
        await pilot.click(pane, offset=(2, target - pane.scroll_offset.y))
        await pilot.pause()

        editor = app.query_one("#editor", TextArea)
        # The snippet's only real computation is on the `return` line.
        assert "tanh" in editor.selected_text or "@" in editor.selected_text
        assert str(path) in status_text(app)


async def test_clicking_a_line_with_no_mapping_says_so(snippet):
    app = make_app(snippet)
    async with app.run_test(size=(120, 30)) as pilot:
        await settle(app, pilot=pilot)
        app.query_one("#outputs", TabbedContent).active = "optimized_hlo"
        await pilot.pause()
        pane = app.query_one("#pane-optimized_hlo", OutputPane)

        blank = next(i for i, line in enumerate(pane.lines) if not line.strip())
        await pilot.click(pane, offset=(2, blank - pane.scroll_offset.y))
        await pilot.pause()

        assert "no source recorded" in status_text(app)


async def test_copying_a_pane_puts_its_text_on_the_clipboard(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        pane = app.query_one("#pane-jaxpr", OutputPane)
        pane.focus()
        await pilot.pause()

        await pilot.press("y")
        await pilot.pause()

        assert app.clipboard == pane.displayed
        assert "dot_general" in app.clipboard


async def test_the_status_line_breaks_the_time_down_by_stage(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        status = status_text(app)
        # Knowing whether lowering or XLA is the slow half is the first question.
        for abbrev in ("jaxpr", "shlo", "hlo", "cost"):
            assert f"{abbrev} " in status
        assert "total " in status


async def test_passes_and_llvm_panes_explain_themselves_when_empty(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        assert "--passes" in pane_text(app, "passes")
        assert "--passes" in pane_text(app, "llvm_ir")


async def test_collecting_passes_fills_both_panes(snippet):
    app = make_app(snippet, passes=True)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        passes = pane_text(app, "passes")
        assert "snapshots" in passes
        assert "pipeline order" in passes
        assert "define" in pane_text(app, "llvm_ir")
        assert "passes" in status_text(app)


async def test_f4_switches_the_passes_pane_between_diff_modes(snippet):
    app = make_app(snippet, passes=True)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        # Text is the default, because it is the view that cannot be wrong.
        assert not app.structural_diff
        assert "text diff" in status_text(app)
        assert "@@" in pane_text(app, "passes")

        await pilot.press("f4")
        await pilot.pause()

        assert app.structural_diff
        assert "structural diff" in status_text(app)
        structural = pane_text(app, "passes")
        # Both renderings keep the same document shape, so only the bodies differ.
        assert "snapshots" in structural
        assert "pipeline order" in structural
        assert "@@" not in structural

        await pilot.press("f4")
        await pilot.pause()

        assert not app.structural_diff
        assert "@@" in pane_text(app, "passes")


async def test_the_status_line_says_when_the_chain_stopped_early(snippet):
    app = make_app(snippet, stages=["jaxpr", "stablehlo"])
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        # The timings shown cover a subset of the pipeline, so say so.
        assert "ran 2/4 stages" in status_text(app)


async def test_the_status_line_stays_quiet_when_every_stage_ran(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        assert "stages" not in status_text(app)


async def test_f3_leaves_the_passes_pane_alone(snippet):
    app = make_app(snippet, passes=True)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        before = pane_text(app, "passes")

        app.show_metadata = True
        await pilot.pause()

        # pass_report already strips the tables from both sides of every diff, so there is
        # nothing left for the filter to do here; it must not mangle the report either.
        assert pane_text(app, "passes") == before
        # And the pane it does own still changes.
        assert "FileNames" in pane_text(app, "optimized_hlo")


async def test_f1_toggles_the_key_panel(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        assert not app.query(HelpPanel)

        await pilot.press("f1")
        await pilot.pause()
        assert app.query(HelpPanel)

        await pilot.press("f1")
        await pilot.pause()
        assert not app.query(HelpPanel)


async def test_question_mark_opens_the_keys_from_a_pane_but_types_in_the_editor(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        app.query_one("#pane-optimized_hlo", OutputPane).focus()
        await pilot.pause()

        await pilot.press("question_mark")
        await pilot.pause()
        assert app.query(HelpPanel)

        # The reason it is bound on the pane and not the app: in the editor it is a character.
        editor = app.query_one("#editor", TextArea)
        editor.focus()
        await pilot.pause()
        before = editor.text
        await pilot.press("question_mark")
        await pilot.pause()

        assert editor.text != before


async def test_bracket_keys_step_between_pass_report_blocks(snippet):
    app = make_app(snippet, passes=True)
    async with app.run_test(size=(100, 24)) as pilot:
        await settle(app, pilot=pilot)
        app.action_show_pane("passes")
        await pilot.pause()
        pane = app.query_one("#pane-passes", OutputPane)
        pane.focus()
        await pilot.pause()
        assert len(pane.blocks) > 2, "need several blocks to have somewhere to step"

        await pilot.press("]")
        await pilot.pause()
        first = pane.scroll_offset.y
        await pilot.press("]")
        await pilot.pause()
        second = pane.scroll_offset.y
        await pilot.press("[")
        await pilot.pause()

        assert second > first
        assert pane.scroll_offset.y == first


async def test_the_bracket_keys_do_nothing_in_a_pane_without_blocks(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)
        pane = app.query_one("#pane-jaxpr", OutputPane)
        pane.focus()
        await pilot.pause()
        assert pane.blocks == []

        await pilot.press("]")
        await pilot.pause()

        assert pane.scroll_offset.y == 0


async def test_f6_toggles_pass_collection_at_runtime(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        first = await settle(app, pilot=pilot)
        assert not first.passes

        # TextArea binds f6 to select_line, so without a priority binding this key selects
        # a line in the buffer instead of toggling anything.
        editor = app.query_one("#editor", TextArea)
        editor.focus()
        await pilot.pause()
        await pilot.press("f6")
        second = await settle(app, pilot=pilot, after=first)

        assert app.collect_passes
        assert second.passes
        assert editor.selected_text == ""  # and the buffer was left alone


async def test_a_stage_subset_leaves_the_other_panes_labelled(snippet):
    app = make_app(snippet, stages=["jaxpr"])
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        assert "dot_general" in pane_text(app, "jaxpr")
        assert "not requested" in pane_text(app, "optimized_hlo")
        assert "--stages jaxpr" in pane_text(app, "analysis")


async def test_f2_stops_offering_a_backend_that_failed(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        app.session.platform = "definitely-not-a-backend"
        await app._restart()
        await pilot.pause()

        # Walking into the same dead backend twice is the trap this avoids.
        assert "definitely-not-a-backend" in app._dead_platforms
        app.action_cycle_platform()
        assert app.session.platform != "definitely-not-a-backend"


async def test_an_unavailable_backend_reports_instead_of_hanging(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        # Whether a real gpu/tpu exists depends on the machine, so aim at a backend that
        # cannot exist anywhere.
        app.session.platform = "definitely-not-a-backend"
        await app._restart()
        await pilot.pause()

        assert app.query_one(TabbedContent).active == "errors"
        assert "definitely-not-a-backend" in status_text(app)
        assert pane_text(app, "errors")


async def test_watch_mode_reloads_from_disk(tmp_path, snippet):
    path = tmp_path / "snippet.py"
    path.write_text(snippet)

    app = JaxplorerApp(path=path, watch=True, platform="cpu")
    async with app.run_test() as pilot:
        first = await settle(app, pilot=pilot)
        assert "tanh" in pane_text(app, "jaxpr")
        assert app.query_one("#editor", TextArea).read_only

        path.write_text(snippet.replace("jnp.tanh", "jnp.sin"))
        await settle(app, pilot=pilot, after=first)

        assert "sin" in pane_text(app, "jaxpr")
        assert "watching" in status_text(app)


async def test_save_writes_the_buffer(tmp_path, snippet):
    path = tmp_path / "snippet.py"
    path.write_text(snippet)

    app = JaxplorerApp(path=path, platform="cpu")
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        app.query_one("#editor", TextArea).load_text("# edited\n" + snippet)
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert path.read_text().startswith("# edited")


async def test_save_is_refused_in_watch_mode(tmp_path, snippet):
    path = tmp_path / "snippet.py"
    path.write_text(snippet)

    app = JaxplorerApp(path=path, watch=True, platform="cpu")
    async with app.run_test() as pilot:
        await settle(app, pilot=pilot)

        await pilot.press("ctrl+s")
        await pilot.pause()

        assert path.read_text() == snippet


async def test_an_old_jax_is_reported_without_counting_as_an_error(snippet):
    app = make_app(snippet)
    async with app.run_test() as pilot:
        result = await settle(app, pilot=pilot)

        # As if the interpreter behind --python carried a jax below the floor.
        assert app.session.info is not None
        app.session.info = replace(app.session.info, warning="jax 0.8.0 is older than …")
        app._render(result)
        await pilot.pause()

        assert "[environment]" in pane_text(app, "errors")
        # A warning is not a failure: it must not flag the tab or inflate the count.
        assert "●" not in str(app.query_one(TabbedContent).get_tab("errors").label)
        assert "ok" in status_text(app)
        assert "error(s)" not in status_text(app)
