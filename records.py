"""Parse downloaded judicial opendata archives into record dicts.

Archives arrive as RAR (司法院 uses .rar format; see fetcher.py). Legacy ZIPs
may also appear. Record payloads come in several JSON shapes:
  - one large JSON array
  - one JSON object per file
  - JSONL / NDJSON

For RAR support, install `rarfile` and make `unrar` or `7z` available on PATH
(on Windows, WinRAR's unrar.exe or 7-Zip's 7z.exe both work). If `rarfile` is
absent, ZIP and JSONL paths still work and we surface a clear error for RAR.
"""
from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

try:
    import rarfile  # type: ignore
    _HAS_RAR = True
except ImportError:
    rarfile = None  # type: ignore
    _HAS_RAR = False

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """One judgment record."""
    jid: str
    jyear: str
    jcase: str
    jno: str
    jdate: str
    jtitle: str
    jfull: str
    jpdf: str = ""
    source_zip: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, source: str = "") -> "Record":
        return cls(
            jid=str(data.get("JID") or data.get("ID") or ""),
            jyear=str(data.get("JYEAR", "")),
            jcase=str(data.get("JCASE", "")),
            jno=str(data.get("JNO", "")),
            jdate=str(data.get("JDATE", "")),
            jtitle=str(data.get("JTITLE", "")),
            jfull=str(data.get("JFULL", "")),
            jpdf=str(data.get("JPDF", "")),
            source_zip=source,
            raw=data,
        )


def _parse_json_bytes(raw: bytes, source_tag: str) -> list[dict]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("decode error in %s", source_tag)
        return []

    text = text.strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            data = json.loads(text)
            return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    if text.startswith("{"):
        try:
            data = json.loads(text)
            return [data] if isinstance(data, dict) else []
        except json.JSONDecodeError:
            pass

    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if isinstance(d, dict):
                out.append(d)
        except json.JSONDecodeError:
            logger.debug("bad line in %s: %s...", source_tag, line[:80])
    return out


def iter_records_zip(zip_path: Path) -> Iterator[Record]:
    """Yield records from a ZIP."""
    source = zip_path.name
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        logger.error("bad zip %s: %s", source, e)
        return

    with zf:
        for name in zf.namelist():
            if not name.lower().endswith((".json", ".jsonl", ".ndjson")):
                continue
            try:
                with zf.open(name) as fh:
                    raw = fh.read()
            except (zipfile.BadZipFile, RuntimeError) as e:
                logger.warning("extract fail %s/%s: %s", source, name, e)
                continue
            for data in _parse_json_bytes(raw, f"{source}/{name}"):
                yield Record.from_dict(data, source=source)


def iter_records_rar(rar_path: Path) -> Iterator[Record]:
    """Yield records from a RAR (requires `rarfile` + unrar/7z binary)."""
    if not _HAS_RAR:
        raise RuntimeError(
            f"RAR file {rar_path.name} but `rarfile` not installed. "
            "Run: pip install rarfile  (and ensure unrar or 7z is on PATH)"
        )
    source = rar_path.name
    try:
        rf = rarfile.RarFile(rar_path)
    except rarfile.Error as e:
        logger.error("bad rar %s: %s", source, e)
        return

    with rf:
        for name in rf.namelist():
            if not name.lower().endswith((".json", ".jsonl", ".ndjson")):
                continue
            try:
                with rf.open(name) as fh:
                    raw = fh.read()
            except rarfile.Error as e:
                logger.warning("extract fail %s/%s: %s", source, name, e)
                continue
            for data in _parse_json_bytes(raw, f"{source}/{name}"):
                yield Record.from_dict(data, source=source)


def iter_records_jsonl(jsonl_path: Path) -> Iterator[Record]:
    """Yield records from a plain JSONL (one record object per line)."""
    source = jsonl_path.name
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield Record.from_dict(data, source=source)


def iter_records(archive_path: Path) -> Iterator[Record]:
    """Dispatch by extension."""
    ext = archive_path.suffix.lower()
    if ext == ".zip":
        yield from iter_records_zip(archive_path)
    elif ext == ".rar":
        yield from iter_records_rar(archive_path)
    elif ext in (".jsonl", ".ndjson"):
        yield from iter_records_jsonl(archive_path)
    else:
        logger.warning("unsupported archive extension: %s", archive_path)


def iter_records_dir(archive_dir: Path) -> Iterator[Record]:
    """Iterate records across all .zip/.rar/.jsonl files in a directory."""
    paths: list[Path] = []
    for ext in ("*.zip", "*.rar", "*.jsonl", "*.ndjson"):
        paths.extend(archive_dir.glob(ext))
    for p in sorted(paths):
        yield from iter_records(p)
