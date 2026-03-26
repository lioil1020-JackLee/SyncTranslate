# Changelog

## 2026-03-26

### Runtime and architecture cleanup

- Removed legacy single-line assumptions between ASR and TTS.
- Fixed runtime behavior to always use bidirectional audio routing.
- Removed obsolete session-mode UI dependency from live runtime behavior.
- Made ASR and LLM strategy permanently direction-specific.
- Removed startup warmup / preheat flow.
- Renamed health check to system check across the app.

### Repository cleanup

- Removed unused compatibility module `app/infra/config/migration.py`.
- Removed generated runtime log files from the repository.
- Moved project documentation into `docs/`.
- Deleted outdated planning and archive markdown files.
- Updated packaging metadata to point to `docs/README.md`.

### Validation

Verified with:

```powershell
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python .\main.py --check
```
