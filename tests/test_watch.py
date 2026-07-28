"""File watcher tests. No JAX involved, so these are fast."""

from jaxplorer.watch import FileWatcher


def test_no_change_reports_nothing(tmp_path):
    path = tmp_path / "snippet.py"
    path.write_text("x = 1\n")
    watcher = FileWatcher(path)

    assert watcher.poll() is None


def test_a_write_is_reported_once(tmp_path):
    path = tmp_path / "snippet.py"
    path.write_text("x = 1\n")
    watcher = FileWatcher(path)

    path.write_text("x = 2\n")
    assert watcher.poll() == "x = 2\n"
    assert watcher.poll() is None


def test_a_same_mtime_write_is_still_caught(tmp_path):
    path = tmp_path / "snippet.py"
    path.write_text("x = 1\n")
    watcher = FileWatcher(path)

    stat = path.stat()
    path.write_text("x = 22\n")
    import os

    os.utime(path, (stat.st_atime, stat.st_mtime))  # editor wrote inside one mtime tick

    assert watcher.poll() == "x = 22\n"


def test_a_deleted_file_is_ignored_until_it_returns(tmp_path):
    path = tmp_path / "snippet.py"
    path.write_text("x = 1\n")
    watcher = FileWatcher(path)

    path.unlink()
    assert watcher.poll() is None

    path.write_text("x = 3\n")
    assert watcher.poll() == "x = 3\n"


def test_a_torn_read_is_retried_rather_than_committed(tmp_path, monkeypatch):
    path = tmp_path / "snippet.py"
    path.write_text("x = 1\n")
    watcher = FileWatcher(path)

    path.write_text("half-writ")
    # Simulate the file growing between the read and the re-stat, i.e. an editor that
    # writes in place rather than renaming.
    real_read_text = type(path).read_text

    def read_then_grow(self, *args, **kwargs):
        text = real_read_text(self, *args, **kwargs)
        real_write = type(self).write_text
        real_write(self, "half-written but complete now\n")
        return text

    monkeypatch.setattr(type(path), "read_text", read_then_grow)
    assert watcher.poll() is None

    monkeypatch.undo()
    # The next tick sees the settled file rather than staying stuck on the torn one.
    assert watcher.poll() == "half-written but complete now\n"
