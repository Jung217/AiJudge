"""Filter records for 臺灣基隆地方法院 drug-related first-instance guilty judgments."""
from __future__ import annotations

import re
from typing import Iterable, Iterator

from records import Record

# "臺" is the official Taiwan-正體 form; "台" is a common variant occasionally
# found in older or auto-OCR'd records.
KEELUNG_COURT_NAMES = (
    "臺灣基隆地方法院",
    "台灣基隆地方法院",
)

DRUG_KEYWORDS = (
    "毒品危害防制條例",
    "毒品",
)

# 聲 / 抗 字別排除理由：
# - 聲：多為聲請觀察勒戒、聲請強制戒治、聲請定應執行刑等非量刑判決
# - 抗：抗告，屬第二審程序
EXCLUDE_CASE_TYPES = frozenset({"聲", "抗"})

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


def is_keelung(record: Record) -> bool:
    head = record.jfull[:400]
    return any(name in head for name in KEELUNG_COURT_NAMES)


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


def keelung_drug_cases(records: Iterable[Record]) -> Iterator[Record]:
    """Yield only records matching all filter predicates."""
    for r in records:
        if not is_keelung(r):
            continue
        if not is_drug_case(r):
            continue
        if is_plea_bargain(r):
            continue
        if not is_first_instance_guilty(r):
            continue
        yield r
