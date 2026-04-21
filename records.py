"""Parse downloaded judicial opendata archives into record dicts.

Archives arrive as RAR (司法院 uses .rar format; see fetcher.py). Inside each
RAR: one JSON file per judgment record. Legacy ZIPs may also appear.

For RAR extraction we prefer Windows' built-in `bsdtar` (ships with Win10+,
at C:\\Windows\\System32\\tar.exe, libarchive-backed so handles RAR natively).
Falls back to 7-Zip, WinRAR's unrar, or pure Python `rarfile` if present.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
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


_BSDTAR_CANDIDATES = (
    r"C:\Windows\System32\tar.exe",
    "bsdtar",
)
_UNRAR_CANDIDATES = (
    "unrar",
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
)
_7Z_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "7z",
)


def _probe(cmd: str, version_check: tuple[str, ...] = ("--version",)) -> bool:
    try:
        r = subprocess.run([cmd, *version_check], capture_output=True,
                           timeout=10, text=True)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_extractor() -> tuple[str, str] | None:
    """Return (kind, command) for a working RAR extractor, or None."""
    for cmd in _BSDTAR_CANDIDATES:
        if _probe(cmd):
            r = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if "bsdtar" in r.stdout or "libarchive" in r.stdout:
                return ("bsdtar", cmd)
    for cmd in _UNRAR_CANDIDATES:
        if _probe(cmd, ("--help",)) or _probe(cmd, ()):
            return ("unrar", cmd)
    for cmd in _7Z_CANDIDATES:
        if _probe(cmd, ("i",)):
            return ("7z", cmd)
    return None


_EXTRACTOR: tuple[str, str] | None = None


def get_extractor() -> tuple[str, str] | None:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = _find_extractor()
    return _EXTRACTOR

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


def iter_records_rar(rar_path: Path, keep_extracted: Path | None = None) -> Iterator[Record]:
    """Yield records from a RAR.

    Strategy: extract the whole RAR to a temp (or persistent) directory, then
    iterate the JSON files within. bsdtar / unrar / 7z all work; see
    `get_extractor`. If `keep_extracted` is given, files are extracted to that
    dir and left in place (re-running is cheap).
    """
    extractor = get_extractor()
    if extractor is None:
        raise RuntimeError(
            f"no RAR extractor available for {rar_path.name}. "
            "Install 7-Zip / WinRAR, or rely on Windows 10+ built-in bsdtar."
        )
    kind, cmd = extractor
    source = rar_path.name

    if keep_extracted:
        work_dir = keep_extracted / rar_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        tmpctx = None
    else:
        tmpctx = tempfile.TemporaryDirectory()
        work_dir = Path(tmpctx.name)

    try:
        if kind == "bsdtar":
            args = [cmd, "-xf", str(rar_path), "-C", str(work_dir)]
        elif kind == "unrar":
            args = [cmd, "x", "-y", "-inul", str(rar_path), str(work_dir) + os.sep]
        else:  # 7z
            args = [cmd, "x", "-y", f"-o{work_dir}", str(rar_path)]

        logger.info("extracting %s via %s ...", source, kind)
        r = subprocess.run(args, capture_output=True, text=False)
        if r.returncode != 0:
            logger.error("extract fail %s: %s", source, r.stderr[:300])
            return

        for json_path in work_dir.rglob("*.json"):
            try:
                raw = json_path.read_bytes()
            except OSError as e:
                logger.warning("read fail %s: %s", json_path, e)
                continue
            for data in _parse_json_bytes(raw, f"{source}/{json_path.name}"):
                yield Record.from_dict(data, source=source)
    finally:
        if tmpctx is not None:
            tmpctx.cleanup()


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
    """Iterate records across all .zip/.rar/.jsonl files in a directory.

    If no archives are found, falls back to recursively scanning *.json
    (pre-extracted tree). This lets callers extract RARs once and re-run the
    filter cheaply.
    """
    paths: list[Path] = []
    for ext in ("*.zip", "*.rar", "*.jsonl", "*.ndjson"):
        paths.extend(archive_dir.glob(ext))

    if paths:
        for p in sorted(paths):
            yield from iter_records(p)
        return

    # Pre-extracted JSON tree mode
    for json_path in sorted(archive_dir.rglob("*.json")):
        try:
            raw = json_path.read_bytes()
        except OSError:
            continue
        for data in _parse_json_bytes(raw, f"{archive_dir.name}/{json_path.name}"):
            yield Record.from_dict(data, source=archive_dir.name)
