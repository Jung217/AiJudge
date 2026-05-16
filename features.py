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
    # Defendant name as captured from 主文 ("" if single-defendant or unparsed).
    # Used for de-duplication and traceability when one record yields multiple
    # CaseFeatures via :func:`extract_features_per_defendant`.
    defendant_name: str = ""

    # Objective
    drug_levels: list[int] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    # 主文-only restriction (the *convicted* charge, not all acts mentioned).
    # Use this for statutory rule lookup — `behaviors` includes facts-narrative
    # acts which are often broader than the actual conviction.
    convicted_behaviors: list[str] = field(default_factory=list)
    convicted_drug_levels: list[int] = field(default_factory=list)
    max_drug_weight_g: Optional[float] = None  # 主文/事實 中最大「淨重 X 公克」
    # Number of distinct defendants named in 主文. >1 means the per-judgment
    # row mixes several people's sentences (and 沒收/§59 flags become
    # ambiguous) — such rows should be dropped from a per-defendant model.
    n_defendants: int = 1
    # Number of 處有期徒刑 verdicts in 主文. >1 → 數罪併罰: `sentence_months`
    # is then the 應執行 aggregate, which a single-count statutory range can't
    # bound (only the 30-year §51 cap applies).
    n_sentence_counts: int = 0
    is_aggregate_sentence: bool = False   # sentence_months came from 應執行
    is_attempt: bool = False              # 主文 含「未遂」(刑法§25Ⅱ 得減輕)
    # 主文中所有「處有期徒刑 X 月」個刑加總(排除 應執行 本身)。對數罪併罰
    # 案件能直接告訴模型「個刑總和」是強 signal — 單罪案件等同 sentence_months。
    sum_individual_months: int = 0

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
    probation_granted: bool = False              # 主文 宣告緩刑 (binary label)
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
    r"不符|未[符能達及]|難認|無從|無法|不適用"
    # 「(雖/並/均/尚/而/仍/亦/卻)無」常出現於駁回語境前
    r"|(?:雖|並|均|尚|而|仍|亦|卻)\s*無"
    r"|無[^。，]{0,60}?之?適用"  # 無...之適用 / 無...規定適用 (放寬到 60)
    r"|不予[適減以]|未予|改口|嗣[後改]|矢口否認"
    # §17Ⅰ:未因而查獲 / 未查獲(其他正犯|共犯)— 這是 §17Ⅰ 駁回鐵證
    r"|未[^。]{0,30}查獲|未因而查獲|並未查獲"
    # 辯護人主張§17但法官未採信 — 駁回慣用法
    r"|辯護人[^。]{0,80}(?:請求|主張|辯)"
    # 「無以」「無從」依據§17減刑
    r"|無\s*以\s*依|無\s*從\s*依"
)
_ART17_APPLY = re.compile(r"減輕|遞減|減其刑")
# Strong reject:在較大窗口(±300)內看到這些 = 該段討論結論是「不適用」,
# 即使本段內有「減輕」字眼也視為駁回(辯護主張被否決等典型情境)。
_ART17_STRONG_REJECT = re.compile(
    r"尚\s*無[^。]{0,80}?(?:規定)?\s*之?\s*適用"
    r"|並\s*無[^。]{0,80}?之\s*適用"
    r"|未\s*因\s*而\s*查\s*獲|並\s*未\s*查\s*獲"
    r"|無\s*以\s*依[^。]{0,60}?減"
    r"|不\s*適\s*用[^。]{0,40}?第\s*(?:十七|17)\s*條"
)
# §59 detection: same citation+window strategy as §17. The bare cite was
# matching釋字 775 boilerplate ("於不符合刑法第59條所定要件之情形下..."),
# rejection clauses ("雖無刑法第59條...適用餘地"), and case-law recitations.
_ART59_CITATION = re.compile(r"刑\s*法\s*第\s*(?:五十九|59)\s*條")
# Recital indicator unique to §59: 釋字 775 explanation references §59 to
# describe what would happen *if* it applied — never an actual application.
_ART59_RECITAL = re.compile(r"釋\s*字\s*第?\s*775|司法院\s*大?\s*法官\s*釋\s*字")
_ART59_APPLY = re.compile(r"減輕|遞減|減其刑|酌減|酌量\s*減")
_ART59_STRONG_REJECT = re.compile(
    r"尚\s*無[^。]{0,80}?(?:規定)?\s*之?\s*適用"
    r"|並\s*無[^。]{0,80}?之\s*適用"
    r"|並\s*非\s*[^。]{0,30}?(?:情堪憫恕|可憫恕|顯可憫恕)"
    r"|難\s*認\s*[^。]{0,40}?(?:情堪憫恕|可憫恕|憫恕)"
    r"|不\s*適\s*用[^。]{0,40}?第\s*(?:五十九|59)\s*條"
)
_ART59_REJECT = re.compile(
    r"不符|未[符能達及]|難認|無從|無法|不適用|並非"
    r"|(?:雖|並|均|尚|而|仍|亦|卻)\s*無"
    r"|無[^。，]{0,60}?之?適用|無\s*[^。，]{0,30}?\s*餘\s*地"
    r"|不予[適減以]|未予"
    # 辯護人主張§59減刑被駁回 慣用句
    r"|辯護人[^。]{0,80}(?:請求|主張|辯)"
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
# Drug net weight extraction (純質淨重/驗餘淨重/淨重 X 公克|公斤). Captures
# numeric value as a string; downstream code converts to grams. Tolerates
# both Arabic ("0.226") and Chinese-decimal ("零點貳貳陸") forms.
_NET_WEIGHT_RE = re.compile(
    r"(?:純\s*質\s*)?(?:驗\s*餘\s*)?淨\s*重\s*(?:約|共\s*計|合\s*計)?\s*"
    rf"(?P<value>[\d{_CN_CHARS}點\.]+?)\s*"
    r"(?P<unit>公\s*斤|公\s*克|克)"
)
# 應執行有期徒刑 (multi-罪併罰 aggregate sentence) — when present, this is
# the actual term the defendant serves and should be preferred over individual
# "處有期徒刑" verdicts. e.g.:
#   "甲犯施用..., 處有期徒刑3月; 又犯..., 處有期徒刑4月. 應執行有期徒刑伍月."
_AGGREGATE_SENTENCE_RE = re.compile(
    r"應\s*執\s*行\s*(?:有\s*期\s*徒\s*刑)?\s*"
    rf"(?:(?P<years>[{_CN_CHARS}\d]+)\s*年)?"
    r"\s*"
    rf"(?:(?P<months>[{_CN_CHARS}\d]+)\s*(?:個)?月)?"
)
# Multi-defendant boundary detector: a Chinese name (1-4 chars; not a
# continuation marker like 又/再) followed by a verb-of-conviction. Used to
# decide whether an 應執行 belongs to the *first* defendant or a later one.
_DEFENDANT_BOUNDARY_RE = re.compile(
    r"。\s*(?![又再而並亦])[一-龥○Ａ-Ｚ\d]{1,5}\s*"
    r"(?:犯|共同|意圖|施用|販賣|轉讓|運輸|製造|持有|幫助)"
)
# Defendant-clause opener: a 1-5 char name (possibly ○-redacted, e.g. 甲○○)
# at the start of 主文 or just after a 句號, followed by a verb-of-conviction.
# Continuation markers (又/再/...) are excluded so a single defendant's
# multiple counts are not miscounted as separate people.
_DEFENDANT_CLAUSE_RE = re.compile(
    r"(?:^|。)\s*(?![又再而並亦另])"
    # Non-greedy {0,4}? so the verb-list lookahead anchors the name boundary
    # — otherwise "羅中彥共同運輸" gets captured as name="羅中彥共同" + verb=運輸.
    r"(?P<name>[一-龥○ＯＡ-Ｚ][一-龥○ＯＡ-Ｚ]{0,4}?)\s*"
    r"(?:犯|共同|意圖|施用|販賣|轉讓|運輸|製造|持有|幫助|教唆|基於|連續)"
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
# Conditional/hypothetical sentence opener — "倘/苟/若 被告/行為人..."
# These describe what *would* happen if the rule applied, not actual finding.
_ART17_CONDITIONAL = re.compile(
    r"(?:倘|苟|若)\s*[被行]"
)
# Case-law citation context — when the cite sits in a sentence explaining
# legal precedent (with 判決意旨 / 裁定意旨 / 判例 references), it's
# typically a procedural rule explanation rather than an actual application
# to this defendant.
_ART17_CASELAW = re.compile(r"判\s*決\s*意\s*旨|裁\s*定\s*意\s*旨|判\s*例")


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
        # Skip hypothetical clauses ("倘被告...", "若行為人...")
        if _ART17_CONDITIONAL.search(window):
            continue
        # Skip case-law-citation contexts (procedural rule explanation)
        if _ART17_CASELAW.search(window):
            continue
        if not _ART17_APPLY.search(window):
            continue
        if _ART17_REJECT.search(window):
            continue
        return True
    return False


def _count_defendants(main_text: str) -> int:
    """Number of distinct defendants named in the 主文.

    Falls back to 1 when no clause opener is recognised (single-defendant
    judgments where 主文 starts with the offense name, or unusual phrasing).
    """
    names = {m.group("name") for m in _DEFENDANT_CLAUSE_RE.finditer(main_text)}
    # Drop spurious captures that are clearly not names (offense fragments etc.)
    names = {n for n in names if 1 <= len(n) <= 5}
    return max(1, len(names))


def _split_main_by_defendant(main_text: str) -> list[tuple[str, str]]:
    """Return ``[(name, slice), ...]`` — one per defendant in 主文 order.

    Each ``slice`` is the chunk of 主文 between this defendant's first clause
    opener and the next one (or end of 主文). 沒收 / 附表 / 應執行刑 phrasing
    typically attaches to the defendant whose preceding 處有期徒刑 it follows
    — that naturally falls inside the slice.

    Single-defendant or unrecognised 主文 → ``[("", main_text)]`` so callers
    can treat the whole text as one defendant. The name is best-effort; for
    rule-based extraction the slice is what matters.
    """
    matches = []
    for m in _DEFENDANT_CLAUSE_RE.finditer(main_text):
        name = m.group("name")
        if 1 <= len(name) <= 5:
            matches.append((name, m.start()))
    # Dedup by name, keep first occurrence of each.
    seen: set[str] = set()
    uniq: list[tuple[str, int]] = []
    for name, pos in matches:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, pos))
    if len(uniq) <= 1:
        return [("", main_text)]
    slices: list[tuple[str, str]] = []
    for i, (name, pos) in enumerate(uniq):
        end = uniq[i + 1][1] if i + 1 < len(uniq) else len(main_text)
        slices.append((name, main_text[pos:end]))
    return slices


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


def _months_from_match(m: re.Match[str]) -> Optional[int]:
    y_raw, mo_raw = m.group("years"), m.group("months")
    if not y_raw and not mo_raw:
        return None
    y = _cn_to_int(y_raw) if y_raw else 0
    mo = _cn_to_int(mo_raw) if mo_raw else 0
    if y is None:
        y = 0
    if mo is None:
        mo = 0
    return (y * 12 + mo) or None


def _parse_decimal(s: str) -> Optional[float]:
    """Parse a number string that may use Arabic digits or Chinese numerals.

    Supports Arabic ("0.226", "16.7", "1,200"), pure Chinese integers
    ("拾陸"), and Chinese decimals ("零點貳貳陸", "拾陸點柒零").
    """
    s = s.strip().replace(",", "")
    if not s:
        return None
    # Try Arabic float
    try:
        return float(s)
    except ValueError:
        pass
    # Chinese decimal: split on 點 (decimal mark)
    int_part, sep, dec_part = s.partition("點")
    int_v = _cn_to_int(int_part) if int_part else 0
    if int_v is None:
        return None
    if not sep:
        return float(int_v)
    # Decimal part: each char is a literal digit (零-玖, 0-9)
    dec_str = ""
    for ch in dec_part:
        if ch.isdigit():
            dec_str += ch
        elif ch in _CN_NUM and _CN_NUM[ch] < 10:
            dec_str += str(_CN_NUM[ch])
        else:
            return None  # unexpected char in decimal
    if not dec_str:
        return float(int_v)
    return float(f"{int_v}.{dec_str}")


def _extract_max_drug_weight_g(text: str) -> Optional[float]:
    """Largest drug net weight (in grams) found in text, normalising 公斤 → 公克."""
    max_g: Optional[float] = None
    for m in _NET_WEIGHT_RE.finditer(text):
        val = _parse_decimal(m.group("value"))
        if val is None or val <= 0:
            continue
        unit = m.group("unit").replace(" ", "")
        if unit.startswith("公斤"):
            val *= 1000.0
        if max_g is None or val > max_g:
            max_g = val
    return max_g


_CHU_VERDICT_RE = re.compile(r"處\s*有\s*期\s*徒\s*刑")
_ATTEMPT_RE = re.compile(r"未\s*遂")


def _sum_individual_sentences(main_text: str) -> int:
    """數罪併罰案件的個刑加總(排除 應執行刑 本身);單罪案件回傳 0。

    避免 label leakage:單罪 sum = sentence_months 直接洩漏標籤,因此只在
    主文有 ≥ 2 筆「處有期徒刑」時回傳加總。對數罪併罰案件,個刑加總是
    強 signal — 模型藉此能直接看到「這個案件有 N 個宣告刑、總和 M 月」,
    不必只看 [0, 30Y] 的 aggregate-only constraint。
    """
    yng = _AGGREGATE_SENTENCE_RE.search(main_text)
    scope = main_text[:yng.start()] if yng else main_text
    months: list[int] = []
    for m in _SENTENCE_RE.finditer(scope):
        v = _months_from_match(m)
        if v is not None:
            months.append(v)
    return sum(months) if len(months) >= 2 else 0


def _extract_sentence_months(main_text: str) -> tuple[Optional[int], bool]:
    """Returns ``(months, is_aggregate)``.

    Prefers the 應執行刑 (數罪併罰 aggregate) over the first 處有期徒刑 verdict;
    ``is_aggregate`` is True when the returned value came from an 應執行 clause.

    Two complications handled here:
      1. Multi-defendant judgments — the first 應執行 in 主文 may belong
         to a later defendant (whose section comes after first defendant's
         single-罪 verdicts). Detect a defendant boundary between first 處
         and the candidate 應執行; if found, prefer first 處.
      2. Partial aggregates — phrases like "前揭得易科罰金之貳罪部分應執行
         有期徒刑X" only sum a subset of verdicts. Skip these and fall back.
    """
    chu_match = _SENTENCE_RE.search(main_text)
    if not chu_match:
        return None, False

    # Find the first 應執行 that isn't a partial-aggregate phrasing.
    yng_match: Optional[re.Match[str]] = None
    for m in _AGGREGATE_SENTENCE_RE.finditer(main_text):
        pre = main_text[max(0, m.start() - 15): m.start()]
        if "部分" in pre or "前揭" in pre:
            continue
        yng_match = m
        break

    if yng_match is None:
        return _months_from_match(chu_match), False

    # If a new-defendant boundary appears between first 處 and 應執行,
    # the 應執行 belongs to a later defendant — fall back to first 處.
    between = main_text[chu_match.end(): yng_match.start()]
    if _DEFENDANT_BOUNDARY_RE.search(between):
        return _months_from_match(chu_match), False
    agg = _months_from_match(yng_match)
    if agg is None:
        return _months_from_match(chu_match), False
    return agg, True


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

def _features_from_main_slice(record: Record, main_slice: str,
                                defendant_name: str,
                                shared_jfull_flags: dict) -> CaseFeatures:
    """Build a CaseFeatures from one defendant's 主文 slice.

    Behaviors / drug levels / sentence label / probation / 易科罰金 / 未遂 /
    aggregate flag are all scoped to the slice (i.e. correct per-defendant).
    Statutory citation flags (§17 / §59 / §62 / §47) are passed in via
    ``shared_jfull_flags`` because they live in 理由 段 with shared phrasing
    across defendants — per-defendant scoping for those is V2.
    """
    sentence_months, is_aggregate = _extract_sentence_months(main_slice)
    return CaseFeatures(
        jid=record.jid,
        jdate=record.jdate,
        defendant_name=defendant_name,
        drug_levels=_find_drug_levels(main_slice),
        behaviors=_find_behaviors(main_slice),
        convicted_behaviors=_find_behaviors(main_slice),
        convicted_drug_levels=_find_drug_levels(main_slice),
        n_defendants=1,   # this is one defendant's row
        n_sentence_counts=len(_CHU_VERDICT_RE.findall(main_slice)),
        is_aggregate_sentence=is_aggregate,
        is_attempt=bool(_ATTEMPT_RE.search(main_slice)),
        sum_individual_months=_sum_individual_sentences(main_slice),
        max_drug_weight_g=_extract_max_drug_weight_g(main_slice),
        art17_1_applied=shared_jfull_flags["art17_1"],
        art17_2_applied=shared_jfull_flags["art17_2"],
        art59_applied=shared_jfull_flags["art59"],
        self_surrender=shared_jfull_flags["self_surrender"],
        recidivism=shared_jfull_flags["recidivism"],
        sentence_months=sentence_months,
        detention_days=_extract_detention_days(main_slice),
        probation_months=_extract_probation_months(main_slice),
        probation_granted=bool(_PROBATION_RE.search(main_slice)),
        can_convert_to_fine="易科罰金" in main_slice,
    )


def extract_features_per_defendant(record: Record) -> list[CaseFeatures]:
    """Return one CaseFeatures per defendant named in 主文.

    Single-defendant judgments → ``[extract_features(record)]``. Multi-defendant
    judgments are split at 主文 clause openers; each defendant gets a row whose
    behaviors / drug levels / sentence label / probation flags come from THEIR
    主文 slice. §17/§59/§62/§47 citations in 理由 段 are still document-wide
    (V1 limitation — per-defendant scoping for those is harder and deferred).
    """
    jfull = record.jfull
    main_text = _extract_section(jfull, "主文",
                                  (r"犯\s*罪\s*事\s*實", r"事\s*實\s*及\s*理\s*由",
                                   r"事\s*實", r"理\s*由"))
    if not main_text:
        return [extract_features(record)]

    slices = _split_main_by_defendant(main_text)
    if len(slices) <= 1:
        # Single-defendant or unparsed — keep legacy whole-document extraction.
        return [extract_features(record)]

    shared = {
        "art17_1": _check_art17(jfull, _ART17_1_CITATION),
        "art17_2": _check_art17(jfull, _ART17_2_CITATION),
        "art59": _check_art59(jfull),
        "self_surrender": _any_match(_ART62_PATTERNS, jfull),
        "recidivism": _any_match(_ART47_PATTERNS, jfull),
    }
    return [
        _features_from_main_slice(record, sl, name, shared)
        for name, sl in slices
        if _SENTENCE_RE.search(sl) or _AGGREGATE_SENTENCE_RE.search(sl)
    ] or [extract_features(record)]


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
    sentence_months, is_aggregate = _extract_sentence_months(main_text)
    # Statutory citations and reduction phrases can appear in 主文, 事實, or 理由
    # depending on judgment style — search the whole document rather than gating
    # on a possibly-misextracted section.
    return CaseFeatures(
        jid=record.jid,
        jdate=record.jdate,
        drug_levels=_find_drug_levels(behavior_text),
        behaviors=_find_behaviors(behavior_text),
        convicted_behaviors=_find_behaviors(main_text),
        convicted_drug_levels=_find_drug_levels(main_text),
        n_defendants=_count_defendants(main_text),
        n_sentence_counts=len(_CHU_VERDICT_RE.findall(main_text)),
        is_aggregate_sentence=is_aggregate,
        is_attempt=bool(_ATTEMPT_RE.search(main_text)),
        sum_individual_months=_sum_individual_sentences(main_text),
        max_drug_weight_g=_extract_max_drug_weight_g(behavior_text),
        art17_1_applied=_check_art17(jfull, _ART17_1_CITATION),
        art17_2_applied=_check_art17(jfull, _ART17_2_CITATION),
        art59_applied=_check_art59(jfull),
        self_surrender=_any_match(_ART62_PATTERNS, jfull),
        recidivism=_any_match(_ART47_PATTERNS, jfull),
        sentence_months=sentence_months,
        detention_days=_extract_detention_days(main_text),
        probation_months=_extract_probation_months(main_text),
        probation_granted=bool(_PROBATION_RE.search(main_text)),
        can_convert_to_fine="易科罰金" in main_text,
    )
