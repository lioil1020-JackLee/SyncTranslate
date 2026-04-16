"""GlossaryLoader — 從 YAML / JSON 載入詞彙表。

詞典格式範例（YAML）：
    entries:
      - pattern: "chat gpt"
        replace: "ChatGPT"
      - pattern: "faster whisper"
        replace: "faster-whisper"
        mode: exact
        case_sensitive: false
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.glossary import GlossaryEntry, GlossaryStore

_logger = logging.getLogger(__name__)


def load_glossary(path: str | Path | None) -> GlossaryStore:
    """從指定路徑載入詞彙表，回傳 GlossaryStore。

    若路徑為空、不存在或解析失敗，回傳空的 GlossaryStore（不拋例外）。
    """
    store = GlossaryStore()
    if not path:
        return store

    resolved = Path(path)
    if not resolved.exists():
        _logger.warning("Glossary file not found: %s", resolved)
        return store

    try:
        raw = _read_file(resolved)
        entries = _parse_entries(raw)
        store.load(entries)
        _logger.info("Loaded %d glossary entries from %s", len(store), resolved)
    except Exception as exc:
        _logger.warning("Failed to load glossary from %s: %s", resolved, exc)

    return store


def _read_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _parse_yaml(text)
    if suffix == ".json":
        return json.loads(text)
    # 嘗試先 YAML 後 JSON
    try:
        return _parse_yaml(text)
    except Exception:
        return json.loads(text)


def _parse_yaml(text: str) -> dict:
    """簡易 YAML 解析（僅支援本詞典格式，避免強制依賴 PyYAML）。

    若系統有 yaml 模組則優先使用。
    """
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    # 極簡自解析（只支援 entries 清單格式）
    return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> dict:
    """極簡 YAML 解析，僅支援本詞典格式。"""
    entries = []
    current: dict | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- pattern:"):
            if current is not None:
                entries.append(current)
            current = {"pattern": stripped[len("- pattern:"):].strip().strip("\"'")}
        elif stripped.startswith("pattern:") and current is not None:
            current["pattern"] = stripped[len("pattern:"):].strip().strip("\"'")
        elif stripped.startswith("replace:") and current is not None:
            current["replace"] = stripped[len("replace:"):].strip().strip("\"'")
        elif stripped.startswith("mode:") and current is not None:
            current["mode"] = stripped[len("mode:"):].strip().strip("\"'")
        elif stripped.startswith("case_sensitive:") and current is not None:
            val = stripped[len("case_sensitive:"):].strip().lower()
            current["case_sensitive"] = val in ("true", "yes", "1")
    if current is not None:
        entries.append(current)
    return {"entries": entries}


def _parse_entries(raw: dict) -> list[GlossaryEntry]:
    entries_raw = raw.get("entries") or []
    result: list[GlossaryEntry] = []
    for item in entries_raw:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern", "")).strip()
        replace = str(item.get("replace", "")).strip()
        if not pattern:
            continue
        mode = str(item.get("mode", "substring")).strip().lower()
        if mode not in ("exact", "substring"):
            mode = "substring"
        case_sensitive = bool(item.get("case_sensitive", False))
        result.append(GlossaryEntry(
            pattern=pattern,
            replace=replace,
            mode=mode,
            case_sensitive=case_sensitive,
        ))
    return result


__all__ = ["load_glossary"]
