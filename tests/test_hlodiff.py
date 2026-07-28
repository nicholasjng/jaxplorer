"""Matching two HLO graphs and reporting the difference. No JAX needed.

Each test is one phenomenon a text diff gets wrong, so the cases double as the argument for
the feature existing.
"""

from jaxplorer.hlo import pass_report
from jaxplorer.hlodiff import diff_modules, render_module_diff, structural_pass_report
from jaxplorer.hlograph import Options, parse_module
from jaxplorer.protocol import PassSnapshot

BEFORE = """HloModule m

%region_0.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add.1 = f32[] add(%a, %b)
}

ENTRY %main.2 (x.1: f32[4,4]) -> f32[] {
  %x.1 = f32[4,4]{1,0} parameter(0), metadata={op_name="x"}
  %constant.1 = f32[] constant(0)
  %tanh.1 = f32[4,4]{1,0} tanh(%x.1), metadata={op_name="jit(f)/tanh"}
  ROOT %reduce.1 = f32[] reduce(%tanh.1, %constant.1), dimensions={0,1}, to_apply=%region_0.1
}
"""


def diff(left: str, right: str, **kwargs):
    options = Options(**kwargs)
    return diff_modules(
        parse_module(left, options=options), parse_module(right, options=options), options=options
    )


def test_a_module_compared_with_itself_is_identical():
    assert diff(BEFORE, BEFORE).identical


def test_renumbering_every_instruction_is_not_a_change():
    renumbered = BEFORE.replace("tanh.1", "tanh.9").replace("reduce.1", "reduce.4")

    result = diff(BEFORE, renumbered)

    assert result.identical
    assert "no structural change" in render_module_diff(result)


def test_metadata_churn_is_not_a_change():
    rewritten = BEFORE.replace('metadata={op_name="x"}', 'metadata={op_name="renamed"}')

    assert diff(BEFORE, rewritten).identical


def test_reordering_instructions_is_not_a_change():
    # A scheduling pass. The lines move; the program does not.
    lines = BEFORE.split("\n")
    swap = lines.index("  %constant.1 = f32[] constant(0)")
    lines[swap], lines[swap - 1] = lines[swap - 1], lines[swap]

    assert diff(BEFORE, "\n".join(lines)).identical


def test_a_changed_shape_is_reported_on_the_matched_instruction():
    reshaped = BEFORE.replace("%tanh.1 = f32[4,4]{1,0}", "%tanh.1 = f32[8,8]{1,0}")

    result = diff(BEFORE, reshaped)

    assert not result.identical
    changed = [
        pair
        for computation in result.computations
        for pair in computation.pairs
        if pair.changed and pair.left.name == "tanh.1"
    ]
    assert len(changed) == 1
    assert changed[0].changes == ("shape",)


def test_a_changed_opcode_is_reported_as_such():
    result = diff(BEFORE, BEFORE.replace("tanh(%x.1)", "exponential(%x.1)"))

    changed = [
        pair for computation in result.computations for pair in computation.pairs if pair.changed
    ]
    assert any("opcode" in pair.changes for pair in changed)


def test_removing_an_instruction_reports_it_removed_and_nothing_added():
    tanh = '  %tanh.1 = f32[4,4]{1,0} tanh(%x.1), metadata={op_name="jit(f)/tanh"}\n'
    fused = BEFORE.replace(tanh, "").replace("reduce(%tanh.1,", "reduce(%x.1,")

    result = diff(BEFORE, fused)

    removed = [ref for computation in result.computations for ref in computation.left_only]
    added = [ref for computation in result.computations for ref in computation.right_only]
    assert [ref.opcode for ref in removed] == ["tanh"]
    assert added == []
    assert result.summary().instruction_delta == -1


def test_a_new_computation_is_reported_as_added():
    with_fusion = BEFORE.replace(
        "ENTRY",
        "%fused_computation (p: f32[4,4]) -> f32[4,4] {\n"
        "  ROOT %p = f32[4,4]{1,0} parameter(0)\n}\n\nENTRY",
    ).replace("tanh(%x.1)", "fusion(%x.1), kind=kLoop, calls=%fused_computation")

    result = diff(BEFORE, with_fusion)

    assert result.summary().computations_added == 1
    assert any("+ %fused_computation" in line for line in render_module_diff(result).split("\n"))


def test_a_removed_computation_is_reported_as_removed():
    result = diff(
        BEFORE.replace(
            "ENTRY",
            "%dead (p: f32[]) -> f32[] {\n  ROOT %p = f32[] parameter(0)\n}\n\nENTRY",
        ),
        BEFORE,
    )

    assert result.summary().computations_removed == 1


def test_the_diff_of_a_swap_is_the_mirror_of_the_diff():
    reshaped = BEFORE.replace("%tanh.1 = f32[4,4]{1,0}", "%tanh.1 = f32[8,8]{1,0}")

    forward = diff(BEFORE, reshaped).summary()
    backward = diff(reshaped, BEFORE).summary()

    assert forward.added == backward.added
    assert forward.removed == backward.removed
    assert forward.changed == backward.changed
    assert forward.instruction_delta == -backward.instruction_delta


def test_renames_are_counted_rather_than_hidden():
    renamed = BEFORE.replace("tanh.1", "tanh.9")

    summary = diff(BEFORE, renamed).summary()

    # Structurally identical, so nothing changed, but the churn stays countable: an
    # over-matching matcher shows up here first.
    assert summary.changed == {}
    assert diff(BEFORE, renamed).identical


def test_the_headline_names_the_magnitude():
    reshaped = BEFORE.replace("%tanh.1 = f32[4,4]{1,0}", "%tanh.1 = f32[8,8]{1,0}")

    headline = diff(BEFORE, reshaped).summary().headline()

    assert "changed" in headline


def test_ignoring_shape_hides_a_pure_reshape():
    reshaped = BEFORE.replace("f32[4,4]", "f32[8,8]")

    assert not diff(BEFORE, reshaped).identical
    assert diff(BEFORE, reshaped, ignore_shape=True).identical


def test_an_edit_to_dead_code_is_reported_rather_than_swallowed():
    dead = BEFORE.replace(
        "  ROOT %reduce.1", "  %dead = s32[7] iota(), iota_dimension=0\n  ROOT %reduce.1"
    )
    changed = dead.replace("iota(), iota_dimension=0", "constant({1,2,3,4,5,6,7})")

    result = diff(dead, changed)

    assert not result.identical
    pairs = [p for c in result.computations for p in c.pairs if p.left.name == "dead"]
    assert len(pairs) == 1
    assert "opcode" in pairs[0].changes


def test_hitting_a_work_cap_does_not_force_pairs_that_scoring_refused(monkeypatch):
    # Truncation used to fall back to pairing leftovers by position whenever any cap fired,
    # which overruled the pairs the scored pass had declined on merit. Only a cap that skips
    # scoring outright may guess.
    monkeypatch.setattr("jaxplorer.hlodiff.MAX_BUCKET", 1)
    unrelated = BEFORE.replace(
        "%tanh.1 = f32[4,4]{1,0} tanh(%x.1)", "%wholly.9 = s32[3]{0} iota(), iota_dimension=0"
    ).replace("reduce(%tanh.1,", "reduce(%x.1,")

    result = diff(BEFORE, unrelated)

    forced = [p for c in result.computations for p in c.pairs if p.left.name == "tanh.1"]
    removed = [ref.name for c in result.computations for ref in c.left_only]
    assert not forced, "an unrelated instruction was paired anyway"
    assert "tanh.1" in removed


def test_a_large_module_diffs_without_blowing_up():
    def build(count: int, opcode: str) -> str:
        body = ["  %v0 = f32[4] parameter(0)"]
        body += [f"  %v{i} = f32[4] {opcode}(%v0)" for i in range(1, count)]
        body.append(f"  ROOT %t = f32[4] add(%v1, %v{count - 1})")
        return "HloModule m\n\nENTRY %main (v0: f32[4]) -> f32[4] {\n" + "\n".join(body) + "\n}\n"

    # 1000 identically shaped siblings per side is the worst case for pairwise matching, so
    # the work cap has to hold rather than the wall clock.
    result = diff(build(1000, "tanh"), build(1000, "exponential"))

    assert not result.identical
    assert render_module_diff(result)


def snapshot(index: int, text: str, before: str = "", after: str = "start") -> PassSnapshot:
    return PassSnapshot(
        index=index, pipeline="simplification", after=after, before=before, text=text
    )


def test_the_structural_report_keeps_the_shape_of_the_text_report():
    reshaped = BEFORE.replace("%tanh.1 = f32[4,4]{1,0}", "%tanh.1 = f32[8,8]{1,0}")
    snapshots = [
        snapshot(0, BEFORE, before="algsimp"),
        snapshot(1, reshaped, before="cse", after="algsimp"),
        snapshot(2, reshaped, before="dce", after="cse"),
    ]

    report = pass_report(snapshots, structural=True)

    assert "3 snapshots, 1 changed the module." in report
    assert "===== algsimp  (simplification) =====" in report
    assert "===== pipeline order =====" in report
    assert report.count("simplification ·") == 3


def test_the_structural_report_stays_quiet_about_a_reordering_pass():
    lines = BEFORE.split("\n")
    swap = lines.index("  %constant.1 = f32[] constant(0)")
    lines[swap], lines[swap - 1] = lines[swap - 1], lines[swap]
    snapshots = [snapshot(0, BEFORE, before="scheduler"), snapshot(1, "\n".join(lines))]

    structural = pass_report(snapshots, structural=True)
    textual = pass_report(snapshots)

    assert "No pass changed the module." in structural
    # The text diff cannot tell: this is the case the feature exists for.
    assert "No pass changed the module." not in textual


def test_the_report_falls_back_to_text_when_the_module_does_not_parse():
    snapshots = [snapshot(0, "not hlo at all\n"), snapshot(1, "still not hlo\n")]

    assert structural_pass_report(snapshots, attribute=lambda _a, _b: "x") is None
    report = pass_report(snapshots, structural=True)
    assert "Structural diff unavailable" in report
    # Falling back means an actual text diff, not an apology.
    assert "-not hlo at all" in report


def test_a_report_without_snapshots_still_explains_how_to_get_them():
    assert "--passes" in pass_report([], structural=True)
