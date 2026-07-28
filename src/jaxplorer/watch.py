"""Detect external edits to the snippet file.

An mtime poll rather than a filesystem-notification dependency: a quarter of a second of
latency is imperceptible next to a JAX compile, and it keeps jaxplorer's runtime dependencies
at two.
"""

from __future__ import annotations

from pathlib import Path

POLL_INTERVAL = 0.25

Stamp = tuple[float, int] | None
"""An mtime and a size, or ``None`` when the file is absent."""


class FileWatcher:
    """Watches one file for changes made outside jaxplorer.

    Parameters
    ----------
    path : Path
        File to watch. It need not exist yet.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stamp = self._read_stamp()

    def _read_stamp(self) -> Stamp:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        # Size too, because two writes inside one mtime tick would otherwise be missed.
        return (stat.st_mtime, stat.st_size)

    def poll(self) -> str | None:
        """Return the new contents if the file changed, else ``None``.

        Stamping only after the read is what makes a torn read recoverable: an editor that
        writes in place can be caught mid-write, and committing that stamp would leave the
        half-file on screen until the next edit.

        Returns
        -------
        str or None
            The file's contents, once per change. ``None`` while it is unchanged, missing,
            unreadable, or still being written.
        """
        stamp = self._read_stamp()
        if stamp is None or stamp == self._stamp:
            return None
        try:
            source = self.path.read_text()
        except OSError:
            return None
        if self._read_stamp() != stamp:
            return None  # still being written; try again on the next tick
        self._stamp = stamp
        return source
