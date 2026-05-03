"""§57 量刑因子 extractor — heuristic.

Strategy:
  1. Find the 量刑審酌 block — typically opens with "爰審酌" / "審酌被告" /
     "兼衡" — and runs until the next 沒收 / paragraph break / 主文 callout.
  2. Apply per-factor regex patterns over the block to detect direction +
     pull a 20–80 char substring of *exact* surrounding text as evidence.
  3. For factors that don't fire, look across the full reason text — the
     judge may discuss 累犯/前案紀錄 (character) outside the §57 block.
  4. Default to absent when no signal — `relation_victim` is absent for
     virtually all drug 施用/持有/販賣 cases.

Substrings are taken verbatim (clip 20–80 chars) so downstream tooling
can re-locate them in the source.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


FACTOR_KEYS = [
    "motive",
    "provocation",
    "means",
    "life_status",
    "character",
    "intellect",
    "relation_victim",
    "duty_breach",
    "harm",
    "post_attitude",
]


# ---------------------------------------------------------------------------
# §57 weighing block locator
# ---------------------------------------------------------------------------

# Open-markers for the §57 weighing block — strong (high precision) and weak.
# Strong markers are unambiguous sentencing-weighing openers. Weak ones can
# trip on 證據能力審酌 (證據適當) — only used as fallback.
_BLOCK_START_STRONG = re.compile(
    r"(?:爰\s*審\s*酌\s*被\s*告|"
    r"爰\s*審\s*酌(?!毒品|該言詞|該書面|證據|當事人)|"
    r"本\s*院\s*審\s*酌\s*被\s*告|"
    r"審\s*酌\s*被\s*告\s*[^。\n]{0,8}?所\s*為|"
    r"審\s*酌\s*被\s*告\s*[^。\n]{0,3}?無視|"
    r"審\s*酌\s*被\s*告\s*[^。\n]{0,3}?漠視|"
    r"審\s*酌\s*被\s*告\s*[^。\n]{0,3}?[前曾]|"
    r"審\s*酌\s*被\s*告\s*[^。\n]{0,3}?(?:輕|未|不|猶)|"
    r"審\s*酌\s*被\s*告\s*構\s*成\s*累\s*犯|"
    r"審\s*酌\s*被\s*告\s*前\s*經|"
    r"考\s*量\s*被\s*告)"
)
_BLOCK_START_WEAK = re.compile(
    r"(?:爰\s*審\s*酌|審\s*酌\s*被\s*告|本\s*院\s*審\s*酌|"
    r"審\s*酌\s*[^。\n]{0,12}?所\s*為|考\s*量\s*被\s*告|"
    r"審\s*酌\s*毒\s*品)"
)
# Hard stops — sections that come after sentencing weighing.
_BLOCK_END = re.compile(
    r"(?:沒\s*收|據\s*上\s*論\s*斷|依\s*刑\s*事\s*訴\s*訟\s*法|"
    r"依\s*刑\s*事訴訟法|附\s*錄\s*法\s*條|本\s*案\s*經\s*檢\s*察\s*官|"
    r"如\s*不\s*服\s*本\s*判\s*決|爰\s*依\s*刑\s*事\s*訴\s*訟\s*法|"
    r"中\s*華\s*民\s*國\s*\d{2,3})"
)


def find_block(reason: str) -> str:
    """Return the §57 weighing block. Falls back to entire reason if absent.

    Skip 量刑審酌 false positives by preferring strong markers; if a weak
    match falls inside a 證據能力 / programmatic recital block (lots of
    "刑事訴訟法第159"s nearby, no 被告 reference), skip it.
    """
    # Try the strong marker first — but iterate, since the first hit may
    # itself fall inside a non-sentencing context (e.g. 量刑後段論證 in
    # 緩刑/累犯加重 blocks). Prefer matches whose downstream 200 chars
    # contain typical sentencing tokens.
    sentencing_signal = re.compile(
        r"(?:量處|處有期徒刑|諭知易科罰金|易科罰金|犯後態度|犯罪動機|"
        r"自陳|自述|自承|戕害|手段|素行|品行|智識程度|教育程度|"
        r"家庭|生活狀況|經濟狀況|坦承犯行|否認|定其應執行|"
        r"非難|非可取|惡習|戒[除絕斷])"
    )
    for m in _BLOCK_START_STRONG.finditer(reason):
        tail = reason[m.start():]
        em = _BLOCK_END.search(tail, pos=10)
        block = tail[: em.start()] if em else tail
        if sentencing_signal.search(block):
            return block
    # Fall back to weak marker.
    for m in _BLOCK_START_WEAK.finditer(reason):
        ctx = reason[max(0, m.start() - 80): m.start() + 240]
        if "刑事訴訟法第159" in ctx or "證據能力" in ctx[:200]:
            continue
        tail = reason[m.start():]
        em = _BLOCK_END.search(tail, pos=10)
        block = tail[: em.start()] if em else tail
        if sentencing_signal.search(block):
            return block
    return reason  # no clear weighing block — search full text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clip_evidence(text: str, m: re.Match[str], min_len: int = 20, max_len: int = 80) -> str:
    """Pull a 20–80 char substring of `text` centered on the regex match.

    The result MUST be an exact substring — no normalization, no edits.
    """
    span = m.end() - m.start()
    pad = max(0, min_len - span) // 2
    start = max(0, m.start() - pad)
    end = min(len(text), m.end() + pad)
    # extend forward to reach min_len
    while (end - start) < min_len and end < len(text):
        end += 1
    while (end - start) < min_len and start > 0:
        start -= 1
    # cap at max_len
    while (end - start) > max_len and end > m.end():
        end -= 1
    while (end - start) > max_len and start < m.start():
        start += 1
    snippet = text[start:end]
    # Strip leading/trailing whitespace fragments to keep it cleaner — but
    # only by trimming the substring boundary, not by rewriting characters.
    # We don't strip because then the substring claim breaks. Instead, retreat
    # boundaries to skip line-leading whitespace blocks for readability.
    return snippet


def first_match(text: str, patterns: list[str]) -> Optional[re.Match[str]]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m
    return None


# ---------------------------------------------------------------------------
# Per-factor patterns
# ---------------------------------------------------------------------------

# motive — drug-case stock phrase: "漠視法令", "輕忽...戕害", "缺乏戒斷決心",
# "未見戒除", "再三施用". These phrase the motive negatively (aggravating).
# Mitigating motives are rare in施用 cases; sometimes "戒癮" / "成癮性" /
# "心理依賴" gets cited as mitigating.
MOTIVE_AGGR = [
    r"漠視[^。]{0,15}?(?:法令|禁制|誡命)",
    r"輕忽[^。]{0,8}?(?:毒品|戕害|健康)",
    r"未\s*見\s*其?\s*戒\s*除",
    r"未\s*能\s*戒[除絕]",
    r"未\s*能\s*記\s*取\s*教\s*訓",
    r"未\s*能\s*體\s*[悟認]",
    r"不\s*知\s*戒\s*[慎除]",
    r"不\s*知\s*悔\s*[改悟]",
    r"猶\s*未\s*能",
    r"再\s*三\s*施\s*用",
    r"屢\s*[次犯]",
    r"自\s*制\s*力\s*薄\s*弱",
    r"缺\s*乏\s*戒[斷除]",
    r"戒\s*毒\s*決\s*心\s*不\s*強",
    r"戒\s*毒\s*意\s*志\s*不\s*堅",
    r"毒\s*癮\s*非\s*淺",
    r"無\s*視\s*[於於][^。]{0,15}?(?:禁\s*令|誡命|嚴禁|告誡|法令)",
    r"故\s*態\s*復\s*[萌發]",
    r"未\s*思\s*悛\s*悔",
    r"竟\s*[未仍]\s*[能思]",
    r"竟\s*再",
    r"執\s*意",
    r"猶\s*再\s*犯",
    r"猶\s*不\s*知",
    r"反\s*覆\s*再\s*犯",
    r"不\s*思\s*悔\s*[改悟]",
    r"輕\s*縱\s*自\s*我",
    r"無\s*視[^。]{0,8}?(?:法律|法令|禁制|誡命|嚴禁|告誡|規範)",
    r"擅\s*自\s*持\s*有",
    r"擅\s*自\s*[施販轉]",
    r"竟\s*[擅持施販]",
    r"非\s*法\s*[施持]",
    r"應\s*予\s*非\s*難",
    r"圖\s*[一謀]?\s*己\s*[私利之]",
    r"圖\s*[利己]",
    r"輕\s*忽",
    r"危\s*害\s*國\s*民",
    r"罔\s*顧[^。]{0,8}?(?:法律|法令|禁制|誡命)",
    r"明\s*知[^。]{0,15}?(?:不可|禁制|為毒品|危害)",
    r"竟\s*為\s*[一私]己",
    r"竟\s*仍",
]
MOTIVE_MIT = [
    r"成\s*癮\s*性",
    r"心\s*理\s*依\s*賴",
    r"一\s*時\s*失\s*慮",
    r"一\s*時\s*[失誤]\s*[念慮]",
]
MOTIVE_NEUT = [
    r"犯\s*罪\s*動\s*機[、，]?\s*目\s*的",
    r"動\s*機[、，]\s*目\s*的[、，]?\s*手\s*段",
    r"持\s*有\s*毒\s*品[^。]{0,4}?(?:動\s*機|目\s*的)",
    r"被\s*告\s*犯\s*罪\s*之?\s*動\s*機",
    r"參\s*酌[^。]{0,4}?犯\s*罪[^。]{0,4}?動\s*機",
]


# means — 平和/手段平和 = mitigating; 殘忍/兇殘 = aggravating
MEANS_MIT = [
    r"手\s*段\s*尚?\s*屬\s*平\s*和",
    r"犯\s*罪\s*手\s*段\s*[尚屬]*\s*平\s*和",
    r"手\s*段\s*平\s*和",
]
MEANS_AGGR = [
    r"手\s*段\s*兇\s*殘",
    r"手\s*段\s*殘\s*忍",
    r"手\s*段\s*惡\s*劣",
]
MEANS_NEUT = [
    r"犯\s*罪\s*[^。]{0,5}?(?:動\s*機|目\s*的)\s*[、，]?\s*手\s*段",
    r"手\s*段\s*[、，]\s*情\s*節",
    r"目\s*的\s*[、，]\s*手\s*段",
    r"動\s*機[、，]\s*目\s*的[、，]\s*手\s*段",
    r"手\s*段\s*[、，]\s*品\s*行",
    r"行\s*為\s*手\s*段",
]


# life_status — 家境/家庭/經濟/生活狀況
LIFE_NEUT = [
    r"家\s*[境庭][^。]{0,8}?(?:勉\s*持|小\s*康|普\s*通|尚\s*可|寬\s*裕|貧寒)",
    r"經\s*濟\s*狀\s*況[^。]{0,15}?(?:勉\s*持|小\s*康|普\s*通|尚\s*可|寬裕|貧寒)",
    r"自\s*[陳述承][^。]{0,10}?(?:家\s*[境庭]|生\s*活\s*狀\s*況|經\s*濟)",
    r"自\s*述[^。]{0,10}?(?:家\s*[境庭]|生\s*活\s*狀\s*況|經\s*濟)",
    r"生\s*活\s*狀\s*況",
    r"家\s*[境庭]\s*狀\s*況",
    r"家\s*庭\s*經\s*濟",
    r"自\s*[陳述承][^。]{0,15}?(?:從事|工作|業|職業|未婚|已婚|離婚|月\s*收|無業)",
    r"職\s*業[、，]\s*經\s*濟",
    r"工\s*作[、，]\s*經\s*濟",
    r"學\s*歷[、，]\s*職\s*業[、，]\s*經\s*濟",
    r"學\s*歷[、，]\s*工\s*作",
    r"教\s*育\s*程\s*度[^。]{0,4}[、，]\s*職\s*業",
    r"自\s*陳\s*業",
    r"暨\s*其\s*生\s*活",
]


# character — 累犯, 前科, 素行, 觀察勒戒
CHAR_AGGR = [
    r"前\s*[科案]\s*紀?\s*錄[^。]{0,15}?(?:不\s*佳|多次|屢|累)",
    r"素\s*行\s*不\s*佳",
    r"累\s*犯",
    r"曾\s*因\s*施\s*用\s*毒\s*品",
    r"前\s*已?\s*因[^。]{0,8}?(?:觀\s*察|勒\s*戒)",
    r"觀\s*察\s*[、,]?\s*勒\s*戒[^。]{0,15}?(?:仍|猶|未)",
    r"屢\s*[次犯][^。]{0,10}?(?:相同|施用|毒品)",
    r"具\s*有\s*特\s*別\s*惡\s*性",
    r"前\s*因.{0,15}?判\s*處",
    r"多\s*次[^。]{0,8}?(?:施用|前科|前案)",
    r"前\s*科\s*紀\s*錄",
    r"前\s*案\s*紀\s*錄",
]
CHAR_MIT = [
    r"素\s*行\s*尚?\s*[佳良]",
    r"無\s*前\s*[科案]",
    r"並\s*無\s*前\s*[科案]",
]
CHAR_NEUT = [
    r"素\s*行",
    r"品\s*行",
]


# intellect — 智識程度 + 教育程度. Stock phrase like "兼衡其智識程度國中畢業" is neutral.
INTEL_NEUT = [
    r"智\s*識\s*程\s*度[^。]{0,15}?(?:國\s*[中小]|高\s*[中職]|大\s*[學專]|專\s*科|碩\s*士|博\s*士|肄\s*業|畢\s*業)",
    r"教\s*育\s*程\s*度[^。]{0,15}?(?:國\s*[中小]|高\s*[中職]|大\s*[學專]|專\s*科|碩\s*士|博\s*士|肄\s*業|畢\s*業)",
    r"自\s*[陳述承][^。]{0,8}?(?:國\s*[中小]|高\s*[中職]|大\s*[學專]|碩\s*士|博\s*士|肄\s*業|畢\s*業)",
    r"自\s*[陳述承][^。]{0,8}?(?:智\s*識\s*程\s*度|教\s*育\s*程\s*度|學\s*歷)",
    r"智\s*識\s*程\s*度",
    r"教\s*育\s*程\s*度",
    r"學\s*歷",
]


# duty_breach — explicit '違反義務' / '違反告誡' phrasing rare in drug cases
DUTY_AGGR = [
    r"違\s*反[^。]{0,8}?(?:義\s*務|告\s*誡|職\s*責)",
    r"漠\s*視[^。]{0,4}?(?:義\s*務|告\s*誡|職\s*責)",
]


# harm — 戕害自身/危害社會
HARM_MIT = [
    # explicit self-harm framing — restricted to "自身/自我/己身/個人"
    r"戕\s*害[^。]{0,4}?自\s*[身己我]",
    r"戕\s*害[^。]{0,4}?個\s*人",
    r"戕\s*害[^。]{0,4}?己\s*身",
    r"戕\s*害\s*本\s*人",
    r"戕\s*害\s*自\s*身\s*健\s*康",
    r"自\s*我\s*身\s*心\s*侵\s*害\s*為\s*主",
    r"自\s*我\s*[身殘戕]\s*害",
    r"自\s*我[^。]{0,4}?身\s*心",
    r"自\s*戕\s*行\s*為",
    r"自\s*殘\s*行\s*為",
    r"侵\s*害[^。]{0,8}?自\s*[身己]",
    r"屬\s*戕\s*害\s*自\s*身",
    r"僅\s*屬\s*戕\s*害",
    r"對\s*他\s*人\s*法\s*益[^。]{0,15}?(?:無|尚無|並無)",
    r"反\s*社\s*會\s*性[^。]{0,4}?(?:程度)?\s*較?\s*低",
    r"非\s*難\s*性\s*[尚較]?\s*低",
    r"對\s*社\s*會[^。]{0,8}?危\s*害[^。]{0,8}?(?:尚\s*非\s*直接|非\s*直接|尚\s*無|尚\s*非)",
    r"未\s*[實侵]\s*[際害][^。]{0,8}?他\s*人",
    r"未\s*侵\s*[害犯][^。]{0,4}?(?:他\s*人|其\s*他)",
    r"未\s*直\s*接\s*危\s*害",
    r"未\s*對\s*他\s*人\s*[造產][^。]{0,4}?(?:危\s*害|損\s*害|傷\s*害)",
    r"非\s*可\s*與\s*侵\s*害\s*他\s*人",
    r"對\s*己\s*身\s*健\s*康",
    r"終\s*究\s*非\s*可",
    r"所\s*生\s*損\s*害\s*尚?\s*非\s*[鉅大重]",
    r"所\s*生\s*危\s*害\s*尚?\s*非",
    r"亦\s*未\s*因\s*此\s*而\s*危\s*害\s*他\s*人",
    r"未\s*危\s*及\s*他\s*人",
    r"持\s*有\s*之?\s*數\s*量\s*尚?\s*微",
    r"數\s*量\s*尚?\s*微",
    r"數\s*量\s*非\s*[鉅大]",
    r"弊\s*害\s*非\s*[鉅大]",
    r"持\s*有[^。]{0,4}?時\s*間\s*短\s*暫",
    r"所\s*生\s*危\s*害\s*並\s*非\s*重\s*大",
    r"所\s*生\s*危\s*害\s*尚?\s*非",
]
HARM_AGGR = [
    r"危\s*害[^。]{0,4}?(?:社\s*會|公\s*共|國\s*家)\s*甚\s*[鉅大]",
    r"危\s*害\s*甚\s*鉅",
    r"造\s*成[^。]{0,8}?(?:社會|家庭)\s*嚴\s*重",
    r"嚴\s*重\s*危\s*害",
    r"助\s*長[^。]{0,12}?(?:不\s*良\s*風\s*氣|毒\s*品|施\s*用|風\s*氣|氾\s*濫|歪\s*風)",
    r"危\s*害[^。]{0,4}?(?:他\s*人|國\s*民)[^。]{0,4}?身\s*心",
    r"戕\s*害\s*他\s*人",
    r"影\s*響\s*社\s*會\s*治\s*安",
    r"毒\s*品\s*流\s*[竄通]",
    r"造\s*成[^。]{0,4}?治\s*安",
    r"危\s*害\s*國\s*民",
    r"危\s*害\s*治\s*安",
]


# post_attitude — 坦承犯行 / 否認 / 矢口
POST_MIT = [
    r"坦\s*[承認]\s*[全所]?\s*犯\s*行",
    r"坦\s*[承認]\s*不\s*諱",
    r"坦\s*[承認]\s*犯\s*罪",
    r"坦\s*[承認][^。]{0,4}?犯\s*行",
    r"自\s*白\s*犯\s*行",
    r"態\s*度\s*[尚良]\s*[好佳]",
    r"態\s*度\s*良\s*好",
    r"見\s*悔\s*意",
    r"悔\s*悟",
    r"知\s*[所有]?\s*悔\s*[悟改過]",
    r"犯\s*後\s*坦\s*[承認]",
    r"於\s*警\s*詢\s*[、,及和]?\s*偵\s*查[^。]{0,15}?坦\s*[承認]",
    r"自\s*首",
    r"認\s*[罪錯]",
]
POST_AGGR = [
    r"否\s*認\s*犯\s*行",
    r"矢\s*口\s*否\s*認",
    r"飾\s*詞\s*否\s*認",
    r"無\s*悔\s*意",
    r"毫\s*無\s*悔\s*[意改]",
    r"態\s*度\s*[不甚]\s*[佳好]",
    r"態\s*度\s*惡\s*劣",
    r"狡\s*辯",
]


# provocation — 受刺激 / 為...所激 / 因...憤而. Rare in drug cases.
PROVOC_MIT = [
    r"受[^。]{0,4}?刺\s*激",
    r"因[^。]{0,8}?所\s*激",
    r"激\s*於\s*義\s*憤",
    r"一\s*時\s*失\s*慮",
]


# ---------------------------------------------------------------------------
# Per-factor extraction
# ---------------------------------------------------------------------------

def detect_motive(block: str) -> tuple[str, str]:
    m = first_match(block, MOTIVE_AGGR)
    if m:
        return "aggravating", clip_evidence(block, m)
    m = first_match(block, MOTIVE_MIT)
    if m:
        return "mitigating", clip_evidence(block, m)
    m = first_match(block, MOTIVE_NEUT)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


def detect_provocation(block: str) -> tuple[str, str]:
    m = first_match(block, PROVOC_MIT)
    if m:
        return "mitigating", clip_evidence(block, m)
    return "absent", ""


def detect_means(block: str) -> tuple[str, str]:
    m = first_match(block, MEANS_MIT)
    if m:
        return "mitigating", clip_evidence(block, m)
    m = first_match(block, MEANS_AGGR)
    if m:
        return "aggravating", clip_evidence(block, m)
    m = first_match(block, MEANS_NEUT)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


def detect_life_status(block: str) -> tuple[str, str]:
    m = first_match(block, LIFE_NEUT)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


def detect_character(block: str, full_reason: str) -> tuple[str, str]:
    # search both block and full_reason — judges discuss prior records
    # extensively in 論罪科刑 ㈡ section before the §57 block.
    for source in (block, full_reason):
        m = first_match(source, CHAR_AGGR)
        if m:
            return "aggravating", clip_evidence(source, m)
    for source in (block, full_reason):
        m = first_match(source, CHAR_MIT)
        if m:
            return "mitigating", clip_evidence(source, m)
    m = first_match(block, CHAR_NEUT)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


def detect_intellect(block: str) -> tuple[str, str]:
    m = first_match(block, INTEL_NEUT)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


def detect_relation_victim(block: str) -> tuple[str, str]:
    # Drug 施用/持有 cases have no victim; rare for販賣 to discuss relation.
    # If present, default to neutral if mentioned at all.
    m = re.search(r"(?:被害人|告訴人)[^。]{0,15}?(?:關係|為)", block)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


def detect_duty_breach(block: str) -> tuple[str, str]:
    m = first_match(block, DUTY_AGGR)
    if m:
        return "aggravating", clip_evidence(block, m)
    return "absent", ""


def detect_harm(block: str) -> tuple[str, str]:
    # Strong-aggravating signals (販賣/轉讓 cases where harm to others dominates):
    # 戕害他人 / 助長社會 / 影響社會治安 take priority over generic MIT phrases.
    strong_aggr = re.search(
        r"戕\s*害\s*他\s*人|助\s*長[^。]{0,12}?(?:不\s*良\s*風\s*氣|風\s*氣|氾\s*濫|歪\s*風)|"
        r"影\s*響\s*社\s*會\s*治\s*安|危\s*害\s*他\s*人\s*身\s*心",
        block,
    )
    if strong_aggr:
        return "aggravating", clip_evidence(block, strong_aggr)
    m = first_match(block, HARM_MIT)
    if m:
        return "mitigating", clip_evidence(block, m)
    m = first_match(block, HARM_AGGR)
    if m:
        return "aggravating", clip_evidence(block, m)
    return "absent", ""


def detect_post_attitude(block: str) -> tuple[str, str]:
    m = first_match(block, POST_MIT)
    if m:
        return "mitigating", clip_evidence(block, m)
    m = first_match(block, POST_AGGR)
    if m:
        return "aggravating", clip_evidence(block, m)
    # generic 犯後態度 mention
    m = re.search(r"犯\s*後\s*態\s*度", block)
    if m:
        return "neutral", clip_evidence(block, m)
    return "absent", ""


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def extract_factors(reason: str) -> dict[str, dict[str, str]]:
    block = find_block(reason)
    factors: dict[str, dict[str, str]] = {}

    direction, ev = detect_motive(block)
    factors["motive"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_provocation(block)
    factors["provocation"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_means(block)
    factors["means"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_life_status(block)
    factors["life_status"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_character(block, reason)
    factors["character"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_intellect(block)
    factors["intellect"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_relation_victim(block)
    factors["relation_victim"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_duty_breach(block)
    factors["duty_breach"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_harm(block)
    factors["harm"] = {"direction": direction, "evidence": ev}

    direction, ev = detect_post_attitude(block)
    factors["post_attitude"] = {"direction": direction, "evidence": ev}

    return factors


def main() -> int:
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    n = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            obj = json.loads(line)
            factors = extract_factors(obj["reason"])
            row = {"jid": obj["jid"], "factors": factors}
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst.flush()
            n += 1
    print(f"wrote {n} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
