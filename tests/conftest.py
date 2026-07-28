from pathlib import Path

import pytest

from jaxplorer.protocol import CompileResult, Stage

REPO_ROOT = Path(__file__).resolve().parents[1]


def stage_text(result: CompileResult, stage: Stage) -> str:
    """The text of a stage that is expected to have succeeded."""
    outcome = result.stages[stage]
    assert outcome.text is not None, f"{stage} did not produce text: {outcome.error}"
    return outcome.text


def stage_error(result: CompileResult, stage: Stage) -> str:
    """The error of a stage that is expected to have failed."""
    outcome = result.stages[stage]
    assert outcome.error is not None, f"{stage} unexpectedly succeeded"
    return outcome.error


def fatal(result: CompileResult) -> str:
    assert result.fatal is not None, "expected a fatal error"
    return result.fatal


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return REPO_ROOT / "examples"


@pytest.fixture
def snippet() -> str:
    """A minimal snippet that compiles cleanly on CPU."""
    return (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "\n"
        "def f(x, w):\n"
        "    return jnp.tanh(x @ w).sum()\n"
        "\n"
        "args = (\n"
        "    jax.ShapeDtypeStruct((8, 16), jnp.float32),\n"
        "    jax.ShapeDtypeStruct((16, 4), jnp.float32),\n"
        ")\n"
    )
