"""jaxplorer: a compiler explorer TUI for JAX.

Deliberately empty of re-exports. Importing :mod:`jaxplorer.protocol` runs this module first, and
the compile worker imports that, so anything pulled in here would land in every worker's
startup path, textual included.

The pieces live in :mod:`jaxplorer.app` (the TUI), :mod:`jaxplorer.session` (worker ownership),
:mod:`jaxplorer.worker` (the compile chain), :mod:`jaxplorer.protocol` (the wire types) and
:mod:`jaxplorer.hlo` (HLO text handling).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
