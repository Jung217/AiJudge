"""Download monthly judicial opendata archives.

Source: 司法院開放資料平台
URL pattern: https://opendata.judicial.gov.tw/api/FilesetLists/{fileSetId}/file
Archive format: RAR (not ZIP) — each file is named e.g. "199601裁判書.rar".

**Bot defense notice.** The endpoint is behind F5 BIG-IP ASM + Cloudflare
Turnstile. A fresh request without the right cookies returns HTTP 500 with
`{"succeeded":false,"message":"Entity \"GetFilesetListFileQuery\" (...) was not
found."}`. To succeed, the client must first load the homepage in a real
browser (to solve the JS challenge) and forward the resulting cookies here.

For automated pipelines, either:
  1. Drive Playwright / Selenium once to obtain the TS cookie and inject it;
  2. Register an account for the per-judgment JDoc API at data.judicial.gov.tw
     (operates 00:00–06:00 only); or
  3. Download RARs manually via browser and drop into --out dir; this module
     will still resume-skip already-present files.

Catalog enumeration (works without challenge):
    GET https://opendata.judicial.gov.tw/api/Datasets?Keyword=裁判書&Page={n}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests
from tqdm import tqdm

OPENDATA_BASE = "https://opendata.judicial.gov.tw"
OPENDATA_URL = OPENDATA_BASE + "/api/FilesetLists/{id}/file"
OPENDATA_CATALOG = OPENDATA_BASE + "/api/Datasets"

# Browser-like UA is necessary to pass F5 ASM fingerprinting.
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

logger = logging.getLogger(__name__)


def load_cookies_json(path: Path = Path(".cookies.json")) -> dict:
    """Load cookies exported by `scripts/prime_cookies.py`."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {c["name"]: c["value"] for c in raw if isinstance(c, dict)}


def make_session(cookie_jar: dict | None = None) -> requests.Session:
    """Build a browser-like session. Priming visits the homepage to collect
    the TS cookie; callers who already solved the JS challenge elsewhere can
    pass cookies via ``cookie_jar``."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Referer": OPENDATA_BASE + "/",
    })
    if cookie_jar:
        for k, v in cookie_jar.items():
            s.cookies.set(k, v, domain="opendata.judicial.gov.tw")
    else:
        try:
            s.get(OPENDATA_BASE + "/", timeout=30)
        except requests.RequestException as e:
            logger.warning("homepage prime failed: %s", e)
    return s


def download_one(
    dataset_id: int,
    out_dir: Path,
    session: requests.Session | None = None,
    retries: int = 3,
    timeout: int = 180,
) -> Path:
    """Download one monthly RAR. Idempotent — skips if a non-empty file exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}.rar"
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.debug("id=%s already downloaded, skipping", dataset_id)
        return out_path

    sess = session or make_session()
    url = OPENDATA_URL.format(id=dataset_id)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=timeout) as r:
                # Detect bot-defense JSON payload even when status might be 200
                ct = r.headers.get("content-type", "")
                if "application/json" in ct:
                    body = r.text[:300]
                    raise RuntimeError(
                        f"id={dataset_id} got JSON ({r.status_code}), "
                        f"likely bot-defense: {body}"
                    )
                r.raise_for_status()
                tmp = out_path.with_suffix(".rar.part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
                return out_path
        except (requests.RequestException, IOError, RuntimeError) as e:
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
    session: requests.Session | None = None,
) -> Iterator[Path]:
    """Download an inclusive ID range. Polite delay between requests."""
    sess = session or make_session()
    ids = range(start_id, end_id + 1)
    for dataset_id in tqdm(list(ids), desc="Downloading"):
        try:
            yield download_one(dataset_id, out_dir, session=sess)
        except RuntimeError as e:
            logger.error("skipping id=%s: %s", dataset_id, e)
            continue
        time.sleep(delay)


def download_ids(
    ids: Iterable[int],
    out_dir: Path,
    delay: float = 0.5,
    session: requests.Session | None = None,
) -> Iterator[Path]:
    sess = session or make_session()
    for dataset_id in tqdm(list(ids), desc="Downloading"):
        try:
            yield download_one(dataset_id, out_dir, session=sess)
        except RuntimeError as e:
            logger.error("skipping id=%s: %s", dataset_id, e)
            continue
        time.sleep(delay)


def list_judgment_datasets(session: requests.Session | None = None) -> list[dict]:
    """Enumerate monthly judgment datasets via catalog API (no challenge needed).

    Returns list of dicts: {datasetId, title, fileSetId, resourceFormat}.
    """
    sess = session or make_session()
    out: list[dict] = []
    page = 1
    while True:
        try:
            r = sess.get(OPENDATA_CATALOG,
                         params={"Keyword": "裁判書", "Page": page},
                         timeout=60)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("catalog page=%d failed: %s", page, e)
            break

        items = data.get("data") or data.get("items") or data
        if not items:
            break
        if isinstance(items, dict):
            items = items.get("results") or items.get("list") or []
        if not items:
            break

        for ds in items:
            for fs in ds.get("filesetLists", []):
                out.append({
                    "datasetId": ds.get("datasetId"),
                    "title": ds.get("title"),
                    "fileSetId": fs.get("fileSetId"),
                    "resourceFormat": fs.get("resourceFormat"),
                })
        page += 1
        if page > 100:
            break
        time.sleep(0.3)
    return out
