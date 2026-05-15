"""Filter records for 北部地方法院 drug-related first-instance guilty judgments.

By default keeps only 基隆地院; pass ``courts=NORTHERN_5_COURT_CODES`` to
``filter_drug_cases`` for the 5-court pretraining set (plan §5.1).
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, Sequence

from records import Record

# JID prefix → (court display name, list of header aliases).
# "臺" is the official 正體 form; "台" is a common variant in older / OCR'd data.
# The 2-letter JID prefix is the primary filter — jfull substring checks were
# leaking cases that merely *referenced* another court (e.g. a Yunlin case
# citing a prior Keelung ruling).
COURT_REGISTRY: dict[str, tuple[str, tuple[str, ...]]] = {
    "KL": ("基隆", ("臺灣基隆地方法院", "台灣基隆地方法院")),
    "TP": ("臺北", ("臺灣臺北地方法院", "台灣台北地方法院", "臺灣台北地方法院")),
    "SL": ("士林", ("臺灣士林地方法院", "台灣士林地方法院")),
    "PC": ("新北", ("臺灣新北地方法院", "台灣新北地方法院",
                     "臺灣板橋地方法院", "台灣板橋地方法院")),
    "TY": ("桃園", ("臺灣桃園地方法院", "台灣桃園地方法院")),
}

NORTHERN_5_COURT_CODES = ("KL", "TP", "SL", "PC", "TY")
KEELUNG_ONLY = ("KL",)
# Backward-compat aliases.
KEELUNG_COURT_NAMES = COURT_REGISTRY["KL"][1]
KEELUNG_JID_PREFIXES = KEELUNG_ONLY

DRUG_KEYWORDS = (
    "毒品危害防制條例",
    "毒品",
)

# 字別排除理由：
# - 聲：多為聲請觀察勒戒、聲請強制戒治、聲請定應執行刑等非量刑判決
# - 抗：抗告，屬第二審程序
# - 簡上：簡易判決上訴，為地院刑事庭第二審；與原 基簡 同案事實會雙算且刑期不同
EXCLUDE_CASE_TYPES = frozenset({"聲", "抗", "簡上"})

GUILTY_MARKERS = (
    "處有期徒刑",
    "處拘役",
    "處罰金",
    "免刑",
)

NOT_GUILTY_MARKERS = (
    "無罪",
    "公訴不受理",
    "不受理",
    "免訴",
)

PLEA_BARGAIN_MARKERS = (
    "協商判決",
    "協商程序",
)

# Lookahead stops capture at the start of the next structural header.
_MAIN_TEXT_RE = re.compile(
    r"主\s*文\s*(.+?)(?=\s*(?:犯\s*罪\s*事\s*實|事\s*實|理\s*由))",
    re.DOTALL,
)


def _extract_main_text(jfull: str) -> str:
    m = _MAIN_TEXT_RE.search(jfull)
    if m:
        return m.group(1)
    idx = jfull.find("主文")
    if idx < 0:
        return ""
    return jfull[idx : idx + 2000]


def court_code_for(record: Record,
                    courts: Sequence[str] = KEELUNG_ONLY) -> str | None:
    """Two-letter court code if ``record`` belongs to one of ``courts`` and the
    court name shows up in the JFULL header. Returns None otherwise.

    The header check is defensive: the JID prefix alone has been seen to leak
    a few cases that reference another court (e.g. citation of a prior ruling).
    """
    if not record.jid:
        return None
    for code in courts:
        if record.jid.startswith(code):
            names = COURT_REGISTRY[code][1]
            if any(name in record.jfull[:150] for name in names):
                return code
            return None
    return None


def is_keelung(record: Record) -> bool:
    return court_code_for(record, KEELUNG_ONLY) is not None


def is_drug_case(record: Record) -> bool:
    return any(kw in record.jtitle for kw in DRUG_KEYWORDS)


def is_plea_bargain(record: Record) -> bool:
    head_block = record.jfull[:2000]
    return any(m in head_block for m in PLEA_BARGAIN_MARKERS)


def is_first_instance_guilty(record: Record) -> bool:
    if record.jcase in EXCLUDE_CASE_TYPES:
        return False

    main = _extract_main_text(record.jfull)
    if not main:
        return False

    if any(m in main for m in NOT_GUILTY_MARKERS):
        return False

    return any(m in main for m in GUILTY_MARKERS)


def filter_drug_cases(
    records: Iterable[Record],
    courts: Sequence[str] = KEELUNG_ONLY,
) -> Iterator[tuple[Record, str]]:
    """Yield ``(record, court_code)`` pairs for one of ``courts``.

    The court code is the 2-letter JID prefix (e.g. ``"KL"``); callers that
    want the display name can look it up in :data:`COURT_REGISTRY`.
    """
    for r in records:
        code = court_code_for(r, courts)
        if code is None:
            continue
        if not is_drug_case(r):
            continue
        if is_plea_bargain(r):
            continue
        if not is_first_instance_guilty(r):
            continue
        yield r, code


def keelung_drug_cases(records: Iterable[Record]) -> Iterator[Record]:
    """Backward-compat: drop the court code, only 基隆 cases."""
    for r, _ in filter_drug_cases(records, KEELUNG_ONLY):
        yield r
