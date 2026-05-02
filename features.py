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

# §17 detection: a citation must co-occur with a reduction-application verb
# in the same local window, with no rejection markers nearby.
#
# Pure factual confession patterns ("於偵查及審理均自白") are NOT used as a
# standalone trigger — they over-fire on:
#   1. Statute recitals quoting the literal text of §17Ⅱ
#   2. §10/§11 (施用/持有) cases where defendant confessed but §17Ⅱ is
#      categorically inapplicable (it only governs §4-§8 acts)
# When §17Ⅱ is genuinely applied, the article is virtually always cited.
_ART17_1_CITATION = re.compile(
    r"(?:毒品危害防制)?條例\s*第\s*(?:十七|17)\s*條\s*第\s*(?:一|1)\s*項"
)
_ART17_2_CITATION = re.compile(
    r"(?:毒品危害防制)?條例\s*第\s*(?:十七|17)\s*條\s*第\s*(?:二|2)\s*項"
)
# Rejection markers that indicate §17 was *not* applied:
#   - 不符/不適用/未予 etc. — explicit non-application
#   - (雖|並|均)無...之適用 — alternative phrasing
#   - 改口/嗣後/矢口否認 — defendant retracted confession at trial
_ART17_REJECT = re.compile(
    r"不符|未[符能達及]|難認|無從|無法|(?:雖|並|均)\s*無|不適用"
    r"|無[^。，]{0,40}?之?適用"  # 無...之適用 / 無...規定適用
    r"|不予[適減以]|未予|改口|嗣[後改]|矢口否認"
)
_ART17_APPLY = re.compile(r"減輕|遞減|減其刑")
# §59 detection: same citation+window strategy as §17. The bare cite was
# matching釋字 775 boilerplate ("於不符合刑法第59條所定要件之情形下..."),
# rejection clauses ("雖無刑法第59條...適用餘地"), and case-law recitations.
_ART59_CITATION = re.compile(r"刑\s*法\s*第\s*(?:五十九|59)\s*條")
# Recital indicator unique to §59: 釋字 775 explanation references §59 to
# describe what would happen *if* it applied — never an actual application.
_ART59_RECITAL = re.compile(r"釋\s*字\s*第?\s*775|司法院\s*大?\s*法官\s*釋\s*字")
_ART59_APPLY = re.compile(r"減輕|遞減|減其刑|酌減|酌量\s*減")
_ART59_REJECT = re.compile(
    r"不符|未[符能達及]|難認|無從|無法|(?:雖|並|均|仍|卻)\s*無"
    r"|不適用|無[^。，]{0,40}?之?適用|無\s*[^。，]{0,30}?\s*餘\s*地"
    r"|不予[適減以]|未予"
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

# Behavior patterns tolerate whitespace (including line breaks and U+3000)
# between every char — judgment text often wraps mid-word, e.g. "毒\n品".
# 禁藥 is included as an alternative target for 轉讓 (per 藥事法 §83 charges
# that get §17Ⅱ relief via case-law extension).
_DRUG_TARGET = r"(?:毒\s*品|禁\s*藥)"
_BEHAVIOR_PATTERNS: dict[str, tuple[str, ...]] = {
    "販賣": (
        rf"販\s*賣(?:\s*第\s*[一二三四1-4]\s*級)?\s*{_DRUG_TARGET}",
        r"販\s*賣.{0,3}予",
    ),
    "施用": (rf"施\s*用(?:\s*第\s*[一二三四1-4]\s*級)?\s*{_DRUG_TARGET}",),
    "持有": (rf"持\s*有(?:\s*第\s*[一二三四1-4]\s*級)?\s*{_DRUG_TARGET}",),
    "運輸": (rf"運\s*輸(?:\s*第\s*[一二三四1-4]\s*級)?\s*{_DRUG_TARGET}",),
    "轉讓": (rf"轉\s*讓(?:\s*第\s*[一二三四1-4]\s*級)?\s*{_DRUG_TARGET}",),
    "製造": (rf"製\s*造(?:\s*第\s*[一二三四1-4]\s*級)?\s*{_DRUG_TARGET}",),
    "意圖販賣而持有": (rf"意\s*圖\s*販\s*賣\s*而\s*持\s*有.{{0,6}}{_DRUG_TARGET}",),
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


# Recital indicators — the citation is quoting the statute rather than
# applying it. These appear in preambles like "按犯第4條至第8條之罪...，
# 毒品危害防制條例第17條第2項定有明文。" before the judge reaches a finding.
#   pre:  "按…犯第[四4]條" — literal opening of §17Ⅱ's statute text
#   post: "規定：「" / "定有明文" / "明定" — quotation/proclamation markers
# Bare "按" alone is intentionally not matched (too broad).
_ART17_RECITAL_PRE = re.compile(
    r"按\s*[^。]{0,15}?犯第\s*(?:[四4]|四)\s*條"
)
_ART17_RECITAL_POST = re.compile(r"^\s*(?:規定[：「:]|定有明文|明定)")


def _check_art59(text: str) -> bool:
    """Return True iff §59 reduction was actually applied to this defendant.

    Same shape as _check_art17:
      1. Skip if 釋字 775 boilerplate is in the same sentence.
      2. Sentence-scoped window (clip to nearest 句號).
      3. Require a reduction-application verb.
      4. Reject on rejection markers (不符, 雖無...適用, etc.).
    """
    for m in _ART59_CITATION.finditer(text):
        prev_p = text.rfind("。", max(0, m.start() - 120), m.start())
        next_p = text.find("。", m.end(), m.end() + 120)
        win_start = (prev_p + 1) if prev_p >= 0 else max(0, m.start() - 80)
        win_end = next_p if next_p >= 0 else min(len(text), m.end() + 80)
        window = text[win_start:win_end]
        if _ART59_RECITAL.search(window):
            continue
        if not _ART59_APPLY.search(window):
            continue
        if _ART59_REJECT.search(window):
            continue
        return True
    return False


def _check_art17(text: str, citation_re: re.Pattern[str]) -> bool:
    """Return True iff §17 reduction was actually applied to this defendant.

    For each citation occurrence:
      1. Skip if the citation is part of a statute recital (preamble before
         the judge reaches the actual finding).
      2. Bound the analysis window to the surrounding *sentence* (text
         between the previous and next 句號 markers, fallback ±80 chars).
         Sentence-scoped windows prevent §17Ⅰ rejections in adjacent
         sentences from poisoning the §17Ⅱ check (and vice-versa).
      3. Require a reduction-application verb (減輕/遞減/減其刑) in the
         sentence and reject if a rejection marker appears.
    """
    for m in citation_re.finditer(text):
        before_close = text[max(0, m.start() - 15): m.start()]
        after_close = text[m.end(): m.end() + 15]
        if _ART17_RECITAL_PRE.search(before_close) or _ART17_RECITAL_POST.match(after_close):
            continue
        # Sentence-scoped window: clip to nearest 句號 in each direction.
        prev_p = text.rfind("。", max(0, m.start() - 120), m.start())
        next_p = text.find("。", m.end(), m.end() + 120)
        win_start = (prev_p + 1) if prev_p >= 0 else max(0, m.start() - 80)
        win_end = next_p if next_p >= 0 else min(len(text), m.end() + 80)
        window = text[win_start:win_end]
        if not _ART17_APPLY.search(window):
            continue
        if _ART17_REJECT.search(window):
            continue
        return True
    return False


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

    # 主文 carries the formal conviction phrasing ("犯販賣第二級毒品罪"); for
    # 轉讓禁藥 (藥事法 §83) cases the offense name appears only there. Search
    # both 主文 and 事實 for behaviors / drug levels.
    behavior_text = main_text + "\n" + text_for_facts
    # Statutory citations and reduction phrases can appear in 主文, 事實, or 理由
    # depending on judgment style — search the whole document rather than gating
    # on a possibly-misextracted section.
    return CaseFeatures(
        jid=record.jid,
        jdate=record.jdate,
        drug_levels=_find_drug_levels(behavior_text),
        behaviors=_find_behaviors(behavior_text),
        art17_1_applied=_check_art17(jfull, _ART17_1_CITATION),
        art17_2_applied=_check_art17(jfull, _ART17_2_CITATION),
        art59_applied=_check_art59(jfull),
        self_surrender=_any_match(_ART62_PATTERNS, jfull),
        recidivism=_any_match(_ART47_PATTERNS, jfull),
        sentence_months=_extract_sentence_months(main_text),
        detention_days=_extract_detention_days(main_text),
        probation_months=_extract_probation_months(main_text),
        can_convert_to_fine="易科罰金" in main_text,
    )
