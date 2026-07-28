"""HLO text handling: table stripping, source mapping, pass reports. No JAX needed."""

from jaxplorer.hlo import DebugInfo, pass_report, strip_debug_tables
from jaxplorer.protocol import PassSnapshot

MODULE = """HloModule jit_f, is_scheduled=true

FileNames
1 "<string>"
2 "snippet.py"

FunctionNames
1 "<module>"
2 "f"

FileLocations
1 {file_name_id=1 function_name_id=1 line=5 end_line=5 column=4 end_column=72}
2 {file_name_id=2 function_name_id=2 line=12 end_line=12 column=17 end_column=23}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=1}

ENTRY %main (p: f32[4]) -> f32[4] {
  %p = f32[4] parameter(0)
  ROOT %tanh = f32[4] tanh(%p), metadata={op_name="jit(f)/tanh" stack_frame_id=2}
}
"""


def test_stripping_keeps_the_module_and_drops_the_tables():
    stripped = strip_debug_tables(MODULE)

    assert "HloModule jit_f" in stripped
    assert "ENTRY %main" in stripped
    assert "ROOT %tanh" in stripped
    for table in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
        assert table not in stripped
    # The whole point is that this is shorter than what XLA emitted.
    assert len(stripped.split("\n")) < len(MODULE.split("\n"))


def test_stripping_a_module_without_tables_is_a_no_op():
    plain = "HloModule m\n\nENTRY %main () -> () {\n  ROOT %t = () tuple()\n}\n"

    assert strip_debug_tables(plain).strip() == plain.strip()


def test_tables_parse():
    info = DebugInfo.parse(MODULE)

    assert not info.empty
    assert info.file_names == {1: "<string>", 2: "snippet.py"}
    assert info.function_names[2] == "f"
    assert info.file_locations[2]["line"] == 12
    assert info.stack_frames[2]["file_location_id"] == 2


def test_an_instruction_resolves_to_its_source_line():
    info = DebugInfo.parse(MODULE)
    line = [ln for ln in MODULE.split("\n") if "ROOT %tanh" in ln][0]

    ref = info.locate(line, prefer="snippet.py")

    assert ref is not None
    assert (ref.file, ref.line, ref.function) == ("snippet.py", 12, "f")
    assert str(ref) == "snippet.py:12 in f"


def test_a_line_with_no_metadata_resolves_to_nothing():
    info = DebugInfo.parse(MODULE)

    assert info.locate("  %p = f32[4] parameter(0)") is None


def test_a_frame_outside_the_snippet_is_walked_out_of():
    # stack_frame_id=1 is the <string> frame; asking to prefer the snippet must not
    # invent a location.
    info = DebugInfo.parse(MODULE)
    line = 'ROOT %x = f32[] add(), metadata={op_name="a" stack_frame_id=1}'

    assert info.locate(line, prefer="snippet.py") is None
    unfiltered = info.locate(line)
    assert unfiltered is not None and unfiltered.file == "<string>"


def test_a_self_parented_frame_does_not_loop():
    info = DebugInfo.parse(MODULE)
    info.stack_frames[9] = {"file_location_id": 1, "parent_frame_id": 9}

    assert info.locate("x metadata={stack_frame_id=9}", prefer="nope.py") is None


def snapshot(index: int, text: str, before: str = "", after: str = "start") -> PassSnapshot:
    return PassSnapshot(
        index=index, pipeline="simplification", after=after, before=before, text=text
    )


def test_pass_report_names_only_the_passes_that_changed_something():
    snapshots = [
        snapshot(0, "HloModule m\n  a\n", before="algsimp"),
        snapshot(1, "HloModule m\n  b\n", before="cse", after="algsimp"),
        snapshot(2, "HloModule m\n  b\n", before="dce", after="cse"),
    ]

    report = pass_report(snapshots)

    assert "3 snapshots, 1 changed the module." in report
    assert "===== algsimp  (simplification) =====" in report
    assert "cse" not in report.split("===== pipeline order =====")[0].replace("before_cse", "")
    assert "-  a" in report and "+  b" in report
    # The index lists every snapshot so the whole pipeline order stays visible.
    assert report.count("simplification ·") == 3


def test_pass_report_ignores_debug_table_churn():
    with_tables = MODULE
    reordered = MODULE.replace('2 "snippet.py"', '2 "renamed.py"')
    report = pass_report([snapshot(0, with_tables), snapshot(1, reordered)])

    assert "No pass changed the module." in report


def test_pass_report_without_snapshots_explains_how_to_get_them():
    report = pass_report([])

    assert "--passes" in report
