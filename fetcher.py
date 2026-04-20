"""Download monthly judicial opendata archives.

Source: 司法院開放資料平台
URL pattern: https://opendata.judicial.gov.tw/api/FilesetLists/{id}/file

Each ID returns a ZIP archive containing JSON records for one monthly release.
The ZIP covers all courts — filtering by 臺灣基隆地方法院 happens in `filter.py`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests
from tqdm import tqdm

OPENDATA_URL = "https://opendata.judicial.gov.tw/api/FilesetLists/{id}/file"
DEFAULT_UA = "AiJudge-Research/0.1"

logger = logging.getLogger(__name__)


def download_one(
    dataset_id: int,
    out_dir: Path,
    session: requests.Session | None = None,
    retries: int = 3,
    timeout: int = 180,
) -> Path:
    """Download a single monthly ZIP. Idempotent — skips if file already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.debug("id=%s already downloaded, skipping", dataset_id)
        return out_path

    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", DEFAULT_UA)
    url = OPENDATA_URL.format(id=dataset_id)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = out_path.with_suffix(".zip.part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
                return out_path
        except (requests.RequestException, IOError) as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning("id=%s attempt=%d failed: %s (retry in %ds)",
                           dataset_id, attempt + 1, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to download dataset {dataset_id}: {last_err}")


def download_range(
    start_id: int,
    end_id: int,
    out_dir: Path,
    delay: float = 0.5,
) -> Iterator[Path]:
    """Download a range of monthly ZIPs inclusive. Polite delay between requests."""
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})

    ids = range(start_id, end_id + 1)
    for dataset_id in tqdm(list(ids), desc="Downloading"):
        try:
            path = download_one(dataset_id, out_dir, session=session)
            yield path
        except RuntimeError as e:
            logger.error("skipping id=%s: %s", dataset_id, e)
            continue
        time.sleep(delay)


def download_ids(
    ids: Iterable[int],
    out_dir: Path,
    delay: float = 0.5,
) -> Iterator[Path]:
    """Download an arbitrary list of IDs (useful for retrying missing IDs)."""
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})
    for dataset_id in tqdm(list(ids), desc="Downloading"):
        try:
            path = download_one(dataset_id, out_dir, session=session)
            yield path
        except RuntimeError as e:
            logger.error("skipping id=%s: %s", dataset_id, e)
            continue
        time.sleep(delay)
