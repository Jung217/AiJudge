"""Parse downloaded judicial opendata ZIPs into record dicts.

The opendata ZIPs may contain JSON files in several shapes:
  - one large JSON array
  - one JSON object per file (with fields)
  - JSONL / NDJSON

We handle all three defensively since the exact layout isn't guaranteed stable.
"""
from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

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

    # JSONL fallback
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


def iter_records(zip_path: Path) -> Iterator[Record]:
    """Yield records from one monthly ZIP."""
    source = zip_path.name
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        logger.error("bad zip %s: %s", source, e)
        return

    with zf:
        for name in zf.namelist():
            lower = name.lower()
            if not lower.endswith((".json", ".jsonl", ".ndjson")):
                continue
            try:
                with zf.open(name) as fh:
                    raw = fh.read()
            except (zipfile.BadZipFile, RuntimeError) as e:
                logger.warning("extract fail %s/%s: %s", source, name, e)
                continue

            for data in _parse_json_bytes(raw, f"{source}/{name}"):
                yield Record.from_dict(data, source=source)


def iter_records_dir(zip_dir: Path) -> Iterator[Record]:
    """Iterate records across all ZIPs in a directory."""
    paths = sorted(zip_dir.glob("*.zip"))
    for zip_path in paths:
        yield from iter_records(zip_path)
