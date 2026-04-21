"""Feature extraction from judgment text.

Coverage in this skeleton:
  - Drug level (1-4) via regex on facts section
  - Behavior (施用/持有/販賣/轉讓/運輸/製造) via regex
  - Article 17 §1 / §2 detection (core statutory reduction)
  - Recidivism, self-surrender, Art.59 酌減
  - Sentence months (first "處有期徒刑" occurrence)
  - Probation months

Not yet implemented (v2 — planned):
  - 純質淨重 extraction (requires parsing forensic report segments)
  - 併科罰金 amount (Chinese amount-to-int conversion is non-trivial)
  - §57 量刑因子 (犯後態度, 品行) — LLM extraction
  - law_version dating via crime date
  - Aggregate-sentence detection for 數罪併罰
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from records import Record

# ---------------------------------------------------------------------------
# Chinese numerals
# ---------------------------------------------------------------------------
# Taiwan judgments frequently use both regular (一, 二, ...) and formal 大寫
# (壹, 貳, ...) numerals. Sentence spans and dates use these interchangeably.

_CN_NUM: dict[str, int] = {
    "零": 0, "○": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
    "壹": 1, "貳": 2, "參": 3, "叁": 3, "肆": 4, "伍": 5,
    "陸": 6, "柒": 7, "捌": 8, "玖": 9,
    "十": 10, "拾": 10,
    "百": 100, "佰": 100,
    "千": 1000, "仟": 1000,
    "萬": 10000,
}
_CN_CHARS = "".join(sorted(_CN_NUM.keys()))


def _cn_to_int(s: str) -> Optional[int]:
    """Minimal CJK numeral parser. Handles 1-9999 with standard positional form."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    total = 0
    current = 0
    for ch in s:
        n = _CN_NUM.get(ch)
        if n is None:
            return None
        if n >= 10:
            if current == 0:
                current = 1
            total += current * n
            current = 0
        else:
            current = n
    total += current
    return total


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseFeatures:
    jid: str
    jdate: str

    # Objective
    drug_levels: list[int] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    net_pure_weight_g: Optional[float] = None  # TODO v2

    # Statutory reductions / enhancements
    art17_1_applied: bool = False   # 毒品§17Ⅰ 供出來源因而查獲
    art17_2_applied: bool = False   # 毒品§17Ⅱ 偵審均自白
    art59_applied: bool = False     # 刑法§59 酌減
    self_surrender: bool = False    # 刑法§62 自首
    recidivism: bool = False        # 刑法§47 累犯

    # Labels
    sentence_months: Optional[int] = None       # 有期徒刑 月數
    detention_days: Optional[int] = None        # 拘役 日數（施用/持有輕案常見）
    probation_months: Optional[int] = None
    fine_ntd: Optional[int] = None               # TODO v2
    can_convert_to_fine: bool = False

    law_version: str = ""                         # TODO v2


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

_ART17_1_PATTERNS = (
    r"毒品危害防制條例\s*第\s*(?:十七|17)\s*條\s*第\s*(?:一|1)\s*項",
    r"供出毒品來源[^。]*?(?:因而|進而)\s*查獲",
)
_ART17_2_PATTERNS = (
    r"毒品危害防制條例\s*第\s*(?:十七|17)\s*條\s*第\s*(?:二|2)\s*項",
    r"偵查[^。]*?審判[^。]*?均[^。]*?自白",
)
_ART59_PATTERNS = (
    r"刑法\s*第\s*(?:五十九|59)\s*條",
    r"情[輕堪]憫恕",
)
_ART62_PATTERNS = (
    r"刑法\s*第\s*(?:六十二|62)\s*條",
)
_ART47_PATTERNS = (
    r"刑法\s*第\s*(?:四十七|47)\s*條",
    r"累\s*犯",
)

_DRUG_LEVEL_PATTERNS: dict[int, tuple[str, ...]] = {
    1: (r"第[一1]級毒品",),
    2: (r"第[二2]級毒品",),
    3: (r"第[三3]級毒品",),
    4: (r"第[四4]級毒品",),
}

_BEHAVIOR_PATTERNS: dict[str, tuple[str, ...]] = {
    "販賣": (r"販\s*賣(?:第[一二三四1-4]級)?毒品", r"販賣.{0,3}予"),
    "施用": (r"施\s*用(?:第[一二三四1-4]級)?毒品",),
    "持有": (r"持\s*有(?:第[一二三四1-4]級)?毒品",),
    "運輸": (r"運\s*輸(?:第[一二三四1-4]級)?毒品",),
    "轉讓": (r"轉\s*讓(?:第[一二三四1-4]級)?毒品",),
    "製造": (r"製\s*造(?:第[一二三四1-4]級)?毒品",),
    "意圖販賣而持有": (r"意圖販賣而持有.{0,6}毒品",),
}

_SENTENCE_RE = re.compile(
    r"處有期徒刑\s*"
    rf"(?:(?P<years>[{_CN_CHARS}\d]+)\s*年)?"
    r"\s*"
    rf"(?:(?P<months>[{_CN_CHARS}\d]+)\s*(?:個)?月)?"
)
_PROBATION_RE = re.compile(
    rf"緩\s*刑\s*(?P<years>[{_CN_CHARS}\d]+)\s*年"
)
_DETENTION_RE = re.compile(
    rf"處\s*拘\s*役\s*(?P<days>[{_CN_CHARS}\d]+)\s*日"
)


def _any_match(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _find_drug_levels(text: str) -> list[int]:
    return sorted(lvl for lvl, pats in _DRUG_LEVEL_PATTERNS.items()
                  if _any_match(pats, text))


def _find_behaviors(text: str) -> list[str]:
    return [b for b, pats in _BEHAVIOR_PATTERNS.items() if _any_match(pats, text)]


def _extract_section(jfull: str, start_marker: str, end_patterns: tuple[str, ...]) -> str:
    """Extract text between a labelled start and any matching end pattern.

    Tolerates whitespace (ASCII or full-width U+3000) between characters of
    the start marker — real judgments commonly write "主　文", "事　實", etc.
    """
    start_re = r"\s*".join(re.escape(c) for c in start_marker)
    m = re.search(start_re, jfull)
    if not m:
        return ""
    tail = jfull[m.end():]
    end_idx = len(tail)
    for pat in end_patterns:
        em = re.search(pat, tail)
        if em and em.start() < end_idx:
            end_idx = em.start()
    return tail[:end_idx]


def _extract_sentence_months(main_text: str) -> Optional[int]:
    m = _SENTENCE_RE.search(main_text)
    if not m:
        return None
    y_raw, mo_raw = m.group("years"), m.group("months")
    if not y_raw and not mo_raw:
        return None
    y = _cn_to_int(y_raw) if y_raw else 0
    mo = _cn_to_int(mo_raw) if mo_raw else 0
    if y is None:
        y = 0
    if mo is None:
        mo = 0
    total = y * 12 + mo
    return total or None


def _extract_probation_months(main_text: str) -> Optional[int]:
    m = _PROBATION_RE.search(main_text)
    if not m:
        return None
    y = _cn_to_int(m.group("years"))
    return y * 12 if y else None


def _extract_detention_days(main_text: str) -> Optional[int]:
    m = _DETENTION_RE.search(main_text)
    if not m:
        return None
    return _cn_to_int(m.group("days"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(record: Record) -> CaseFeatures:
    jfull = record.jfull

    main_text = _extract_section(jfull, "主文",
                                  (r"犯\s*罪\s*事\s*實", r"事\s*實\s*及\s*理\s*由",
                                   r"事\s*實", r"理\s*由"))

    # 簡易判決 uses compound "事實及理由" header (breaks naive "事實" lookup because
    # the end-pattern "理由" matches immediately inside the header itself). Try
    # the compound form first, then fall back to separate 通常判決 sections.
    # Minimum-length gate catches near-empty extractions that leaked before.
    MIN_LEN = 50
    facts_text = _extract_section(
        jfull, "事實及理由",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"附\s*錄", r"書\s*記\s*官"),
    )
    if len(facts_text) < MIN_LEN:
        facts_text = _extract_section(jfull, "犯罪事實",
                                       (r"\n\s*理\s*由", r"證\s*據\s*清\s*單"))
    if len(facts_text) < MIN_LEN:
        facts_text = _extract_section(jfull, "事實", (r"\n\s*理\s*由",))

    reason_text = _extract_section(
        jfull, "事實及理由",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官"),
    )
    if len(reason_text) < MIN_LEN:
        reason_text = _extract_section(jfull, "理由",
                                        (r"附.{0,3}表", r"中\s*華\s*民\s*國\s*\d{3}"))

    text_for_facts = facts_text if len(facts_text) >= MIN_LEN else jfull
    text_for_reason = reason_text if len(reason_text) >= MIN_LEN else jfull

    return CaseFeatures(
        jid=record.jid,
        jdate=record.jdate,
        drug_levels=_find_drug_levels(text_for_facts),
        behaviors=_find_behaviors(text_for_facts),
        art17_1_applied=_any_match(_ART17_1_PATTERNS, text_for_reason),
        art17_2_applied=_any_match(_ART17_2_PATTERNS, text_for_reason),
        art59_applied=_any_match(_ART59_PATTERNS, text_for_reason),
        self_surrender=_any_match(_ART62_PATTERNS, text_for_reason),
        recidivism=_any_match(_ART47_PATTERNS, text_for_reason),
        sentence_months=_extract_sentence_months(main_text),
        detention_days=_extract_detention_days(main_text),
        probation_months=_extract_probation_months(main_text),
        can_convert_to_fine="易科罰金" in main_text,
    )
