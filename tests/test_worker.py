"""Worker tests: the compile chain and, more importantly, how it reports failure."""

import pathlib

from conftest import fatal, stage_error, stage_text
from jaxplorer.protocol import ALL_STAGES, CompileRequest, CompileResult
from jaxplorer.worker import handle


def run(source: str, **kwargs) -> CompileResult:
    return handle(CompileRequest(id=1, source=source, **kwargs))


def test_all_stages_populated(snippet):
    result = run(snippet)

    assert result.fatal is None
    assert set(result.stages) == set(ALL_STAGES)
    assert all(stage.ok for stage in result.stages.values())

    assert "dot_general" in stage_text(result, "jaxpr")
    assert "tanh" in stage_text(result, "jaxpr")
    assert "stablehlo" in stage_text(result, "stablehlo")
    assert "ENTRY" in stage_text(result, "optimized_hlo")
    assert "flops" in stage_text(result, "analysis")
    assert result.total_ms > 0


def test_concrete_array_args_work_too():
    source = (
        "import jax.numpy as jnp\n"
        "def f(x):\n"
        "    return jnp.sin(x).sum()\n"
        "args = (jnp.arange(4.0),)\n"
    )
    result = run(source)

    assert result.fatal is None
    assert "sin" in stage_text(result, "jaxpr")


def test_single_non_tuple_arg_is_accepted():
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "def f(x):\n"
        "    return x * 2\n"
        "args = jax.ShapeDtypeStruct((3,), jnp.float32)\n"
    )
    result = run(source)

    assert result.fatal is None
    assert result.stages["jaxpr"].ok


def test_static_argnums_is_honored():
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "def f(x, n):\n"
        "    return jnp.tile(x, n)\n"
        "args = (jax.ShapeDtypeStruct((3,), jnp.float32), 4)\n"
        "static_argnums = (1,)\n"
    )
    result = run(source)

    assert result.fatal is None
    # The static argument is baked in, so the traced signature takes one operand and the
    # result is the tiled shape.
    assert "f32[12]" in stage_text(result, "jaxpr")


def test_an_already_jitted_f_keeps_its_own_options():
    # Re-jitting would drop static_argnums that jaxplorer never saw, and tracing would then
    # fail on the string argument.
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "@jax.jit\n"
        "def g(x):\n"
        "    return x * 2\n"
        "f = g\n"
        "args = (jax.ShapeDtypeStruct((3,), jnp.float32),)\n"
    )
    result = run(source)

    assert result.fatal is None
    assert result.stages["jaxpr"].ok


def test_snippet_jit_options_win_over_an_already_jitted_f():
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "@jax.jit\n"
        "def g(x, n):\n"
        "    return jnp.tile(x, n)\n"
        "f = g\n"
        "args = (jax.ShapeDtypeStruct((3,), jnp.float32), 4)\n"
        "static_argnums = (1,)\n"
    )
    result = run(source)

    assert result.fatal is None
    assert "f32[12]" in stage_text(result, "jaxpr")


def test_kwargs_are_passed():
    source = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "def f(x, *, scale):\n"
        "    return x * scale\n"
        "args = (jax.ShapeDtypeStruct((3,), jnp.float32),)\n"
        "kwargs = {'scale': 3.0}\n"
    )
    result = run(source)

    assert result.fatal is None
    assert result.stages["jaxpr"].ok


def test_syntax_error_is_fatal_with_buffer_line_number():
    result = run("def f(:\n", filename="<buffer>")

    assert result.stages == {}
    assert "SyntaxError" in fatal(result)
    assert '"<buffer>", line 1' in fatal(result)


def test_module_level_exception_points_at_user_code():
    result = run("x = 1 / 0\n")

    assert "ZeroDivisionError" in fatal(result)
    assert "<buffer>" in fatal(result)
    # Worker frames must not leak into what the user reads.
    assert "worker.py" not in fatal(result)


def test_missing_f_explains_the_contract():
    result = run("args = ()\n")

    assert result.stages == {}
    assert "no `f`" in fatal(result)
    assert "def f" in fatal(result)


def test_missing_args_explains_the_contract():
    result = run("def f(x):\n    return x\n")

    assert "no `args`" in fatal(result)
    assert "ShapeDtypeStruct" in fatal(result)


def test_non_callable_f_is_rejected():
    result = run("f = 3\nargs = ()\n")

    assert "must be callable" in fatal(result)


def test_trace_failure_keeps_earlier_stages_and_skips_the_rest(snippet):
    # Contracting dimensions no longer agree, so tracing fails.
    result = run(snippet.replace("(16, 4)", "(17, 4)"))

    assert result.fatal is None
    assert result.stages["jaxpr"].error is not None
    assert "contracting dimensions" in stage_error(result, "jaxpr")
    # The user's own frame is shown; JAX's traceback-filtering footnote is not.
    assert "in f" in stage_error(result, "jaxpr")
    assert "JAX_TRACEBACK_FILTERING" not in stage_error(result, "jaxpr")

    for stage in ("stablehlo", "optimized_hlo", "analysis"):
        assert result.stages[stage].skipped
        assert result.stages[stage].error is None

    assert len(result.errors()) == 1


def test_stage_subset_only_reports_requested_stages(snippet):
    result = handle(CompileRequest(id=7, source=snippet, stages=["jaxpr", "stablehlo"]))

    assert set(result.stages) == {"jaxpr", "stablehlo"}
    assert result.stages["stablehlo"].ok


def test_analysis_table_is_readable(snippet):
    text = stage_text(run(snippet), "analysis")

    assert "Cost:" in text
    assert "Memory:" in text
    # A serialized proto in a table meant to be skimmed is noise.
    assert "serialized_buffer_assignment_proto" not in text


def test_pass_snapshots_and_llvm_ir_are_collected_on_request(snippet):
    result = handle(CompileRequest(id=1, source=snippet, passes=True))

    assert result.fatal is None
    assert len(result.passes) > 5
    # Indices are the order XLA ran them in, and each snapshot is a whole module.
    assert [p.index for p in result.passes] == sorted(p.index for p in result.passes)
    assert all("HloModule" in p.text for p in result.passes)
    assert any(p.before for p in result.passes)
    # The CPU backend emits LLVM IR, which is the closest thing to an assembly view.
    assert result.llvm_ir is not None
    assert "define" in result.llvm_ir


def test_no_dumps_are_collected_unless_asked(snippet):
    result = handle(CompileRequest(id=1, source=snippet))

    assert result.passes == []
    assert result.llvm_ir is None


def test_the_dump_directory_does_not_outlive_the_request(snippet, tmp_path):
    import tempfile

    before = set(pathlib.Path(tempfile.gettempdir()).glob("jaxplorer-dump-*"))
    handle(CompileRequest(id=1, source=snippet, passes=True))
    after = set(pathlib.Path(tempfile.gettempdir()).glob("jaxplorer-dump-*"))

    # Dumps of a large model are tens of MB; leaking one per keystroke would be rough.
    assert after == before


def test_namespace_does_not_leak_between_requests():
    first = run("leaked = 1\ndef f(x):\n    return x\nargs = (1.0,)\n")
    assert first.fatal is None

    second = run("def f(x):\n    return leaked\nargs = (1.0,)\n")
    assert second.stages["jaxpr"].error is not None
    assert "NameError" in stage_error(second, "jaxpr")
