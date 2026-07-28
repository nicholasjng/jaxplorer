"""Parsing HLO text into a graph, and fingerprinting it. No JAX needed.

The parser is the load-bearing half of the structural diff, and HLO text is a debug format
with no stability guarantee, so there is a test per printer construct rather than one big
module.
"""

from jaxplorer.hlograph import Options, canonicalize, fingerprint, parse_module

ENTRY = """HloModule m, entry_computation_layout={(f32[4,4]{1,0})->f32[]}

ENTRY %main.2 (x.1: f32[4,4]) -> f32[] {
  %x.1 = f32[4,4]{1,0} parameter(0), metadata={op_name="x"}
  %constant.1 = f32[] constant(0)
  %dot_general.1 = f32[4,4]{1,0} dot(%x.1, %x.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %reduce.1 = f32[] reduce(%dot_general.1, %constant.1), dimensions={0,1}, to_apply=%region_0.1
}
"""


def test_parsing_an_entry_computation_recovers_the_dag():
    module = parse_module(ENTRY)

    assert module.name == "m"
    assert module.entry == "main.2"
    assert module.unparsed == 0
    assert len(module.computations) == 1

    computation = module.computations[0]
    assert computation.entry
    assert computation.root == "reduce.1"
    assert [i.opcode for i in computation.instructions] == [
        "parameter",
        "constant",
        "dot",
        "reduce",
    ]

    by_name = computation.index()
    assert by_name["x.1"].parameter_number == 0
    assert by_name["dot_general.1"].operands == ("x.1", "x.1")
    assert by_name["reduce.1"].operands == ("dot_general.1", "constant.1")
    assert by_name["reduce.1"].called == ("region_0.1",)
    assert by_name["reduce.1"].is_root


def test_a_tuple_shape_is_not_mistaken_for_the_operand_list():
    # The shape prints before the opcode, so taking the first parenthesis would parse
    # `f32[4], f32[4]` as operands and lose the real ones.
    module = parse_module(
        "HloModule m\n\nENTRY %main () -> (f32[4], f32[4]) {\n"
        "  %a = f32[4] parameter(0)\n"
        "  %b = f32[4] parameter(1)\n"
        "  ROOT %t = (f32[4], f32[4]) tuple(%a, %b)\n}\n"
    )

    root = module.computations[0].index()["t"]
    assert root.opcode == "tuple"
    assert root.shape == "(f32[4], f32[4])"
    assert root.operands == ("a", "b")


def test_a_trailing_layout_is_split_off_but_a_tuple_shape_is_left_alone():
    module = parse_module(ENTRY)
    by_name = module.computations[0].index()

    assert by_name["x.1"].shape == "f32[4,4]"
    assert by_name["x.1"].layout == "{1,0}"
    # No layout to split: the scalar shape ends in a bracket of a different kind.
    assert by_name["constant.1"].shape == "f32[]"
    assert by_name["constant.1"].layout == ""


def test_braces_inside_attributes_do_not_end_the_attribute_list():
    module = parse_module(ENTRY)
    attributes = dict(module.computations[0].index()["dot_general.1"].attributes)

    assert attributes["lhs_contracting_dims"] == "{1}"
    assert attributes["rhs_contracting_dims"] == "{0}"


def test_metadata_is_kept_aside_rather_than_treated_as_an_attribute():
    module = parse_module(ENTRY)
    instruction = module.computations[0].index()["x.1"]

    assert "op_name" in instruction.metadata
    assert "metadata" not in dict(instruction.attributes)
    # Click-to-source reads the original line, so it has to survive verbatim.
    assert instruction.text.strip().startswith("%x.1 = f32[4,4]{1,0} parameter(0)")


def test_a_quoted_json_backend_config_does_not_confuse_the_bracket_scanner():
    # Real modules carry this on every fusion, braces and colons inside a quoted string.
    text = (
        "HloModule m\n\nENTRY %main (p: s32[]) -> s32[] {\n"
        "  %p = s32[] parameter(0)\n"
        "  ROOT %w = s32[] while(%p), condition=%cond, body=%body, "
        'backend_config={"known_trip_count":{"n":"4"}}\n'
        "}\n"
    )

    module = parse_module(text)

    assert module.unparsed == 0
    root = module.computations[0].index()["w"]
    assert root.opcode == "while"
    assert root.operands == ("p",)
    assert root.called == ("cond", "body")
    # Dropped by default, since it carries autotuning results rather than semantics.
    assert "backend_config" not in dict(root.attributes)


def test_backend_config_can_be_kept_when_it_is_the_thing_being_studied():
    text = (
        "HloModule m\n\nENTRY %main (p: s32[]) -> s32[] {\n"
        '  ROOT %p = s32[] parameter(0), backend_config={"n":"4"}\n}\n'
    )

    module = parse_module(text, options=Options(ignore_backend_config=False))

    assert dict(module.computations[0].index()["p"].attributes)["backend_config"] == '{"n":"4"}'


def test_a_constant_literal_that_wraps_across_lines_is_read_as_one_instruction():
    text = (
        "HloModule m\n\nENTRY %main () -> f32[4] {\n"
        "  ROOT %c = f32[4] constant({1.0, 2.0,\n"
        "    3.0, 4.0})\n}\n"
    )

    module = parse_module(text)

    assert module.unparsed == 0
    assert len(module.computations[0].instructions) == 1
    assert module.computations[0].index()["c"].opcode == "constant"


def test_control_flow_bodies_are_separate_computations_referenced_by_name():
    text = (
        "HloModule m\n\n"
        "%cond (p: s32[]) -> pred[] {\n  ROOT %p = pred[] parameter(0)\n}\n\n"
        "%body (p: s32[]) -> s32[] {\n  ROOT %p = s32[] parameter(0)\n}\n\n"
        "%branch (p: s32[]) -> s32[] {\n  ROOT %p = s32[] parameter(0)\n}\n\n"
        "ENTRY %main (x: s32[]) -> s32[] {\n"
        "  %x = s32[] parameter(0)\n"
        "  %w = s32[] while(%x), condition=%cond, body=%body\n"
        "  ROOT %c = s32[] conditional(%w), branch_computations={%branch}\n}\n"
    )

    module = parse_module(text)

    assert len(module.computations) == 4
    assert module.entry == "main"
    by_name = module.computations[-1].index()
    assert by_name["w"].called == ("cond", "body")
    assert by_name["c"].called == ("branch",)


def test_the_debug_tables_are_skipped_without_being_special_cased():
    tables = (
        '\nFileNames\n1 "snippet.py"\n\nStackFrames\n1 {file_location_id=1 parent_frame_id=1}\n\n'
    )
    head, _, body = ENTRY.partition("\n")

    with_tables = parse_module(head + tables + body)
    without = parse_module(ENTRY)

    assert with_tables.unparsed == 0
    assert with_tables.instruction_count == without.instruction_count
    assert fingerprint(with_tables).module == fingerprint(without).module


def test_an_unreadable_line_is_counted_rather_than_raised():
    text = (
        "HloModule m\n\nENTRY %main () -> f32[4] {\n"
        "  this is not an instruction at all\n"
        "  ROOT %c = f32[4] constant(0)\n}\n"
    )

    module = parse_module(text)

    assert module.unparsed == 1
    assert module.lines == 1
    assert len(module.computations[0].instructions) == 1


def test_a_module_that_is_not_hlo_at_all_yields_nothing_and_survives():
    module = parse_module('this is a stack trace\n  File "x.py", line 1\n')

    assert module.computations == ()
    assert module.entry_computation is None


REORDERED = """HloModule m

ENTRY %main.2 (x.1: f32[4,4]) -> f32[] {
  %constant.1 = f32[] constant(0)
  %x.1 = f32[4,4]{1,0} parameter(0), metadata={op_name="x"}
  %dot_general.1 = f32[4,4]{1,0} dot(%x.1, %x.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %reduce.1 = f32[] reduce(%dot_general.1, %constant.1), dimensions={0,1}, to_apply=%region_0.1
}
"""


def test_reordering_instructions_does_not_change_the_fingerprint():
    # What a scheduling pass does. Defs no longer precede uses here, which is why
    # the hash walks operands instead of trusting printed order.
    assert fingerprint(parse_module(ENTRY)).module == fingerprint(parse_module(REORDERED)).module


def test_renumbering_instructions_does_not_change_the_fingerprint():
    renumbered = ENTRY.replace("dot_general.1", "dot_general.7").replace("x.1", "x.9")

    assert fingerprint(parse_module(ENTRY)).module == fingerprint(parse_module(renumbered)).module


def test_an_instruction_the_root_cannot_reach_still_counts():
    # Rewiring a use leaves the old producer unreferenced until DCE runs, so between passes
    # there is plenty of dead code, and an edit to it is still an edit.
    with_dead = ENTRY.replace(
        "  ROOT %reduce.1", "  %dead = s32[7] iota(), iota_dimension=0\n  ROOT %reduce.1"
    )
    changed = with_dead.replace("iota(), iota_dimension=0", "constant({1,2,3,4,5,6,7})")

    assert fingerprint(parse_module(with_dead)).module != fingerprint(parse_module(changed)).module
    # And it is a difference from the original too, not just between the two variants.
    assert fingerprint(parse_module(ENTRY)).module != fingerprint(parse_module(with_dead)).module


def test_changing_an_opcode_does_change_the_fingerprint():
    changed = ENTRY.replace("dot(", "add(")

    assert fingerprint(parse_module(ENTRY)).module != fingerprint(parse_module(changed)).module


def test_shapes_count_by_default_and_can_be_ignored_on_request():
    reshaped = ENTRY.replace("f32[4,4]", "f32[8,8]")

    assert fingerprint(parse_module(ENTRY)).module != fingerprint(parse_module(reshaped)).module
    options = Options(ignore_shape=True)
    assert (
        fingerprint(parse_module(ENTRY, options=options), options=options).module
        == fingerprint(parse_module(reshaped, options=options), options=options).module
    )


def test_layout_is_ignored_by_default():
    relaid = ENTRY.replace("{1,0}", "{0,1}")

    assert fingerprint(parse_module(ENTRY)).module == fingerprint(parse_module(relaid)).module


def test_fingerprints_are_stable_across_processes():
    # A content digest rather than hash(), which is salted per interpreter.
    assert fingerprint(parse_module(ENTRY)).module == fingerprint(parse_module(ENTRY)).module


def test_a_long_operand_chain_does_not_exhaust_the_stack():
    # Recursion over a 2000-deep DAG would; the walk is iterative for this reason.
    depth = 2000
    body = ["  %v0 = f32[4] parameter(0)"]
    body += [f"  %v{i} = f32[4] tanh(%v{i - 1})" for i in range(1, depth)]
    body[-1] = "  ROOT " + body[-1].strip()
    header = "HloModule m\n\nENTRY %main (v0: f32[4]) -> f32[4] {\n"
    text = header + "\n".join(body) + "\n}\n"

    module = parse_module(text)

    assert module.unparsed == 0
    assert module.instruction_count == depth
    assert fingerprint(module).module


def test_canonicalize_drops_metadata_without_touching_structure():
    module = canonicalize(parse_module(ENTRY))

    assert all(not i.metadata for c in module.computations for i in c.instructions)
    assert fingerprint(module).module == fingerprint(parse_module(ENTRY)).module
