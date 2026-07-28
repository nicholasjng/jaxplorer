"""Snippets that satisfy jaxplorer's contract, shipped inside the wheel.

They live in the package rather than beside it so that ``jaxplorer mlp`` works from any
directory and for anyone who installed rather than cloned. v0.1.0 shipped without them while
the README told users to open ``examples/mlp.py``, which no install contained.

The snippets themselves are read as source and never imported: :func:`load` hands the text to
the worker, which execs it in its own namespace. That keeps jax out of jaxplorer's own import
graph, which is the whole reason the worker is a separate process.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files


@cache
def names() -> tuple[str, ...]:
    """Return the bundled example names, without the ``.py``, alphabetically.

    Cached, and a tuple rather than a list, because the CLI interpolates this into ``--help``:
    scanning the package directory on every invocation would put a filesystem walk in front
    of a help message that is meant to be instant.
    """
    return tuple(
        sorted(
            path.name.removesuffix(".py")
            for path in files(__name__).iterdir()
            if path.name.endswith(".py") and path.name != "__init__.py"
        )
    )


def load(name: str) -> str | None:
    """Return the source of a bundled example, or ``None`` if there is no such example.

    Parameters
    ----------
    name : str
        Example name, with or without the ``.py`` suffix.

    Returns
    -------
    str or None
    """
    stem = name.removesuffix(".py")
    if stem not in names():
        return None
    return (files(__name__) / f"{stem}.py").read_text(encoding="utf-8")
