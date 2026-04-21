"""JDoc REST API client (data.judicial.gov.tw/jdg).

Per spec (114.08.22):
  - POST /jdg/api/Auth   {user, password} → {Token}  (valid 6h)
  - POST /jdg/api/JList  {token}          → [{date, list: [JID, ...]}]  (last 7 days)
  - POST /jdg/api/JDoc   {token, j: JID}  → {ATTACHMENTS, JFULLX, JID, JYEAR, JCASE, JNO, JDATE, JTITLE}

Service window: **00:00–06:00 Taiwan time only**. Requests outside this window
return errors. Use opendata RAR archives (via fetcher.py) for historical bulk;
use this client for ongoing 7-day incremental sync.

Credentials must be registered at data.judicial.gov.tw beforehand.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import requests

from records import Record

BASE = "https://data.judicial.gov.tw/jdg/api"
logger = logging.getLogger(__name__)


@dataclass
class JDocError(Exception):
    status: int
    message: str
    def __str__(self) -> str:
        return f"[{self.status}] {self.message}"


class JDocClient:
    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        *,
        token: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: int = 60,
    ):
        self.user = user or os.environ.get("JDOC_USER")
        self.password = password or os.environ.get("JDOC_PASS")
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()

    def _post(self, path: str, payload: dict) -> dict | list:
        url = f"{BASE}{path}"
        r = self.session.post(url, json=payload, timeout=self.timeout,
                              headers={"Content-Type": "application/json"})
        try:
            data = r.json()
        except ValueError:
            raise JDocError(r.status_code, r.text[:300])
        if isinstance(data, dict) and data.get("error"):
            raise JDocError(r.status_code, str(data["error"]))
        r.raise_for_status()
        return data

    def authenticate(self) -> str:
        if not self.user or not self.password:
            raise JDocError(0, "credentials missing; set JDOC_USER and JDOC_PASS env vars")
        data = self._post("/Auth", {"user": self.user, "password": self.password})
        if not isinstance(data, dict) or "Token" not in data:
            raise JDocError(0, f"unexpected Auth response: {data!r}")
        self.token = data["Token"]
        return self.token

    def _ensure_token(self) -> str:
        return self.token or self.authenticate()

    def list_recent_jids(self) -> list[dict]:
        """Return last-7-day JID deltas: [{date, list:[JID, ...]}]."""
        token = self._ensure_token()
        data = self._post("/JList", {"token": token})
        if not isinstance(data, list):
            raise JDocError(0, f"unexpected JList response: {data!r}")
        return data

    def get_document(self, jid: str) -> Record:
        """Fetch one judgment by JID."""
        token = self._ensure_token()
        data = self._post("/JDoc", {"token": token, "j": jid})
        if not isinstance(data, dict):
            raise JDocError(0, f"unexpected JDoc response: {data!r}")

        jfullx = data.get("JFULLX") or {}
        jfull_text = jfullx.get("JFULLCONTENT", "") if jfullx.get("JFULLTYPE") == "text" else ""
        jpdf = jfullx.get("JFULLPDF", "")

        return Record(
            jid=str(data.get("JID", jid)),
            jyear=str(data.get("JYEAR", "")),
            jcase=str(data.get("JCASE", "")),
            jno=str(data.get("JNO", "")),
            jdate=str(data.get("JDATE", "")),
            jtitle=str(data.get("JTITLE", "")),
            jfull=jfull_text,
            jpdf=jpdf,
            source_zip="jdoc-api",
            raw=data,
        )

    def iter_documents(
        self,
        jids: Iterable[str],
        delay: float = 0.3,
    ) -> Iterator[Record]:
        """Fetch many JIDs sequentially with polite delay."""
        for jid in jids:
            try:
                yield self.get_document(jid)
            except JDocError as e:
                logger.warning("JDoc fail %s: %s", jid, e)
            time.sleep(delay)


# 基隆地方法院 JID prefix. Confirmed from real 2026 data:
# KLDM (刑事庭), other KL* variants for different divisions.
KEELUNG_COURT_PREFIXES = ("KL",)


def is_keelung_jid(jid: str) -> bool:
    """Fast pre-filter by JID prefix before calling expensive JDoc."""
    return any(jid.startswith(p) for p in KEELUNG_COURT_PREFIXES)
