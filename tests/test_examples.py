"""Every bundled example must satisfy the snippet contract and compile end to end."""

import pytest

from conftest import stage_text
from jaxplorer.protocol import ALL_STAGES, CompileRequest
from jaxplorer.worker import handle


def example_files(examples_dir):
    return sorted(examples_dir.glob("*.py"))


def test_examples_exist(examples_dir):
    assert example_files(examples_dir)


@pytest.mark.parametrize("name", ["mlp.py", "scan.py", "attention.py"])
def test_example_compiles(examples_dir, name):
    path = examples_dir / name
    result = handle(CompileRequest(id=1, source=path.read_text(), filename=str(path)))

    assert result.fatal is None, result.fatal
    for stage in ALL_STAGES:
        assert result.stages[stage].ok, result.stages[stage].error


def test_scan_example_keeps_the_loop_as_a_primitive(examples_dir):
    source = (examples_dir / "scan.py").read_text()
    result = handle(CompileRequest(id=1, source=source))

    # The point of the example: scan is one primitive with a nested jaxpr, not 64
    # unrolled steps.
    assert "scan[" in stage_text(result, "jaxpr")
