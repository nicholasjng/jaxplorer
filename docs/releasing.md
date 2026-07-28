# Crafting a release

`.github/workflows/release.yml` builds and publishes on a *published* GitHub release,
through trusted publishing in the `pypi` environment.
Tags are `vX.Y.Z`; the version lives in `src/jaxplorer/__init__.py` and `uv.lock` does not pin it.

## 1. Bump the version

```python
# src/jaxplorer/__init__.py
__version__ = "X.Y.Z"
```

## 2. Check

```bash
uv run pytest -q
uvx prek run --all-files
uv run ty check
uv run jaxplorer --version    # X.Y.Z
```

## 3. Land it on master

```bash
jj commit -m "jaxplorer vX.Y.Z"
jj bookmark set master -r @-
jj git push --bookmark master
```

## 4. Publish

Creates the tag and triggers the workflow:

```bash
gh release create vX.Y.Z --target master --title "jaxplorer vX.Y.Z" --generate-notes
```

## 5. Verify

```bash
gh run list --workflow=release.yml --limit 1
uvx jaxplorer@X.Y.Z --version
```

## Notes

- Bump the minor for behaviour changes, the patch for fixes and docs; pre-1.0, breaking
  changes go in a minor.
- `jj git push` cannot push tags, which is why step 4 lets GitHub create it.
- A failed publish cannot be retried against the same version: bump, and release again.
