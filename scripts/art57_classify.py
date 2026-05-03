"""§57 sentencing-factor classifier for Taiwan drug-case judgments.

Strategy:
  1. Extract the reasoning section from jfull.
  2. Within the reasoning, locate the §57 enumeration block ("audit block")
     where the judge lists the sentencing factors. Anchors:
     start: "審酌被告" / "爰審酌" / "兼衡" / "兼酌" / "考量" (preceded by 爰/惟/併/暨)
     end:   "量處(如主文)" / "處有期徒刑" / "宣告(刑)"
  3. Pattern-match each §57 factor on the audit block (NOT whole reason),
     to avoid false positives from 累犯/自首/§59 explanations.
  4. Window-extract a 20-80 char evidence substring centred on the cue.

Direction reflects the JUDGE's framing, not crime severity.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import _extract_section


# ---------------------------------------------------------------------------
# Reason extraction
# ---------------------------------------------------------------------------

def extract_reason(jfull: str) -> str:
    """Extract reasoning section with multiple fallbacks.

    End-pattern strategy: the only reliable end marker is the date+stamp at
    the END of the judgment ("中華民國 XXX 年 XX 月 XX 日" + 法官 / 書記官).
    Generic 附表 markers fail because real judgments reference 附表一、附表二
    inside the reasoning text. We use a stricter end pattern that requires
    full date+month+day.
    """
    END_PATS = (
        # End-of-judgment date stamp followed by 法官 or 書記官 within ~80 chars
        r"中\s*華\s*民\s*國\s*\d{2,3}\s*[　\s]*年\s*\d{1,2}\s*[　\s]*月\s*\d{1,2}\s*[　\s]*日"
        r"(?:[\s\S]{0,200}?(?:法\s*官|書\s*記\s*官))",
        # Plain end-of-judgment date as last resort
        r"中\s*華\s*民\s*國\s*\d{2,3}\s*[　\s]*年\s*\d{1,2}\s*[　\s]*月\s*\d{1,2}\s*[　\s]*日"
        r"\s*[　\s]*\n\s*\S{0,30}?(?:法\s*官|簡易庭|刑事第)",
        # 附錄/附件
        r"附\s*錄(?:本案)?\s*論\s*罪",
    )
    reason = _extract_section(jfull, "事實及理由", END_PATS)
    if len(reason) >= 50:
        return reason
    reason = _extract_section(jfull, "理由", END_PATS)
    if len(reason) >= 50:
        return reason
    reason = _extract_section(jfull, "論罪科刑", END_PATS)
    if len(reason) >= 50:
        return reason
    return jfull


# ---------------------------------------------------------------------------
# Audit-block extraction — the §57 enumeration
# ---------------------------------------------------------------------------

# Common 量刑 block openers (in priority order):
#   - 爰以行為人責任為基礎，審酌
#   - 爰以(...)責任為基礎，審酌
#   - 爰審酌
#   - (其它句首) 審酌被告
_AUDIT_OPEN_RES = [
    re.compile(r"爰\s*[以].{0,20}?責任\s*[為基礎之]+\s*[，,].{0,5}?審\s*酌"),
    re.compile(r"爰\s*[審衡考].{0,2}?[酌量酌]"),
    re.compile(r"(?:[\(（一二三四五六七八九十]+[\)）]|[一二三四五六七八九十]\s*[、,．])\s*爰?\s*審\s*酌\s*被\s*告"),
    # 通常判決 multi-defendant cases use "本院審酌" / "兼衡其等" structure;
    # the §57 enumeration appears AFTER 累犯/§59/§17 reasoning; identify it by
    # cues like "犯罪動機、目的、手段" or "智識程度" or "犯後態度"
    re.compile(r"審\s*酌\s*[^。]{0,30}?(?:犯罪[之]?動機|犯罪手段|犯後態度|前案紀錄|無視於|漠視)"),
    re.compile(r"審\s*酌\s*被\s*告(?!.{0,5}?(?:前案|事實|證據|構成累犯))"),
    re.compile(r"考\s*量\s*被\s*告(?:無|不|有|曾)"),
    # For 通常判決 with structured 兼衡 phrasing (no爰):
    re.compile(r"兼\s*衡\s*其?[等]?(?:[^。]{0,30}?(?:犯罪[之]?動機|犯後態度|參與情節))"),
]
# End markers for the audit block (in order tried).
_AUDIT_END_RES = [
    re.compile(r"量\s*處(?:如\s*)?(?:主\s*文|[本宣]\s*告)?"),
    re.compile(r"分\s*別\s*量\s*處"),
    re.compile(r"處\s*以\s*主\s*文"),
    re.compile(r"科\s*處\s*如\s*主\s*文"),
    re.compile(r"以\s*資\s*[懲儆警]"),
    re.compile(r"附\s*錄"),
    re.compile(r"沒\s*收\s*部\s*分"),
    re.compile(r"^\s*[三四五六七八九十]\s*[、,．]\s*[沒緩依如本據據以]"),
]


def extract_audit_block(reason: str) -> str:
    """Return the §57 enumeration block, or the whole reason if no block found."""
    start_idx = -1
    for r in _AUDIT_OPEN_RES:
        m = r.search(reason)
        if m:
            start_idx = m.start()
            break
    if start_idx < 0:
        return reason  # fallback
    tail = reason[start_idx:]
    end_off = len(tail)
    for r in _AUDIT_END_RES:
        m = r.search(tail, 1)
        if m and m.start() < end_off:
            end_off = m.start()
    return tail[:end_off + 30]  # +30 to keep "量處主文所示之刑" inside if useful


# ---------------------------------------------------------------------------
# Window helpers — extract 20-80 char substring centred on a regex match
# ---------------------------------------------------------------------------

_PUNCT_BOUNDARY = "，。；：、,;:!?"
_WS_AND_PUNCT = " \t\n\r" + _PUNCT_BOUNDARY


def _window(text: str, start: int, end: int, target_len: int = 50) -> str:
    """Return a 20-80 char substring of text covering [start, end].

    Anchors to nearest punctuation when reasonable.
    """
    pad = max(0, (target_len - (end - start)) // 2)
    win_start = max(0, start - pad)
    win_end = min(len(text), end + pad)

    # Try to align left edge to a punctuation boundary just before win_start
    left_search = text[max(0, win_start - 20):win_start]
    last_punct = -1
    for i in range(len(left_search) - 1, -1, -1):
        if left_search[i] in _PUNCT_BOUNDARY:
            last_punct = i
            break
    if last_punct >= 0:
        new_left = max(0, win_start - 20) + last_punct + 1
        if start - new_left <= target_len + 20:
            win_start = new_left

    # Snap right edge to next punctuation if reasonably close
    right_search = text[end:min(len(text), win_end + 20)]
    for i, ch in enumerate(right_search):
        if ch in _PUNCT_BOUNDARY:
            new_end = end + i
            if new_end - start <= target_len + 30:
                win_end = new_end
                break

    snippet = text[win_start:win_end]
    # Strip whitespace and full-width spaces but preserve content
    snippet = snippet.strip(" \t\n\r　")
    # Replace internal whitespace runs with single space for compactness
    snippet = re.sub(r"[\s　]+", "", snippet)
    if len(snippet) > 80:
        snippet = snippet[:80]
    if len(snippet) < 20:
        # Expand symmetrically up to 80 chars
        ws = win_start
        we = win_end
        while len(snippet) < 20 and (ws > 0 or we < len(text)):
            if ws > 0:
                ws -= 1
            if we < len(text):
                we += 1
            snippet = re.sub(r"[\s　]+", "", text[ws:we])
            if ws == 0 and we == len(text):
                break
        if len(snippet) > 80:
            snippet = snippet[:80]
    return snippet


def _find_window(text: str, pattern: str, target_len: int = 50) -> Optional[str]:
    m = re.search(pattern, text)
    if not m:
        return None
    return _window(text, m.start(), m.end(), target_len)


def _ensure_substring(snippet: str, original: str) -> str:
    """If snippet (whitespace-stripped) doesn't appear verbatim in original,
    fall back to taking a contiguous window from original around the snippet's
    matching position.
    """
    if not snippet:
        return ""
    if snippet in original:
        return snippet
    # Try to find a contiguous window in original that matches snippet's
    # whitespace-stripped form.
    stripped_orig = re.sub(r"[\s　]+", "", original)
    if snippet not in stripped_orig:
        return ""
    # Map back: find the substring in original whose stripped version starts
    # where snippet starts in stripped_orig.
    target_pos = stripped_orig.index(snippet)
    target_end = target_pos + len(snippet)
    # Walk original char by char, counting non-whitespace chars
    ws_re = re.compile(r"[\s　]")
    count = 0
    orig_start = -1
    orig_end = -1
    for i, ch in enumerate(original):
        if not ws_re.match(ch):
            if count == target_pos and orig_start < 0:
                orig_start = i
            count += 1
            if count == target_end:
                orig_end = i + 1
                break
    if orig_start >= 0 and orig_end > orig_start:
        result = original[orig_start:orig_end]
        if len(result) > 80:
            # trim from right, end at last non-ws char
            result = result[:80]
        return result
    return ""


# ---------------------------------------------------------------------------
# Per-factor classifiers — operate on audit_block, with reason as fallback
# ---------------------------------------------------------------------------


def classify_motive(audit: str, reason: str) -> tuple[str, str]:
    """動機目的 — usually neutral as factual statement.

    Aggravating: profit motive (sales cases) — judge frames as 圖私利/牟利.
    """
    full = audit
    # Aggravating sales-motive cues
    for p in (
        r"圖\s*一\s*己\s*[之]?\s*[私利]+",
        r"圖\s*[謀牟取]+\s*(?:暴?利|私?利)",
        r"為\s*[供賺]?\s*營\s*利",
        r"獲\s*取(?:不法)?\s*利\s*益",
        r"圖\s*賺(?:取)?\s*差\s*價",
        r"以\s*營\s*利\s*[為之]",
        r"為\s*謀[個取]人?\s*[私之]\s*利",
        r"僅\s*圖\s*一\s*己",
        r"為\s*謀\s*[取賺]?\s*[暴利之私]+",
        r"竟\s*為\s*[謀取]\s*個\s*人\s*[私之]?\s*利",
    ):
        ev = _find_window(full, p, 60)
        if ev:
            return "aggravating", ev
    # Neutral motive listing
    for p in (
        r"動\s*機(?:[、，,]\s*目\s*的)?",
        r"犯\s*罪[之]?\s*動\s*機",
        r"目\s*的\s*在?[供於解癮戒減毒]+",
        r"動\s*機\s*[在於為]+\s*解\s*癮",
    ):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    if "動機" in full or "目的" in full:
        ev = _find_window(full, r"動\s*機|目\s*的", 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_provocation(audit: str, reason: str) -> tuple[str, str]:
    """受刺激 — provocation. Rare in drug cases.

    When judge lists "所受之刺激" as part of the §57 template, treat as neutral
    (mere mention without finding). Genuine provocation cues are mitigating.
    """
    full = audit
    # Genuine mitigating provocation language
    for p in (r"挑\s*釁", r"因\s*[何之].{0,5}?刺\s*激", r"被害人.{0,5}?[挑刺激釁]"):
        ev = _find_window(full, p, 50)
        if ev:
            return "mitigating", ev
    # Templated mention: "所受之刺激" in factor list
    for p in (r"所\s*受\s*之?\s*[剌刺]\s*激", r"受\s*[何之]?\s*[剌刺]\s*激"):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_means(audit: str, reason: str) -> tuple[str, str]:
    """手段 — usually 平和 (mitigating)."""
    full = audit
    # Aggravating
    for p in (r"手\s*段\s*[兇殘暴]+", r"殘\s*忍", r"兇\s*殘",
              r"以\s*[強暴脅迫]+\s*手\s*段"):
        ev = _find_window(full, p, 50)
        if ev:
            return "aggravating", ev
    # Mitigating
    for p in (
        r"手\s*段\s*(?:尚屬|尚稱)?\s*平\s*和",
        r"手\s*段\s*非\s*暴\s*力",
        r"犯\s*罪\s*手\s*段\s*[尚屬]+\s*平\s*和",
    ):
        ev = _find_window(full, p, 50)
        if ev:
            return "mitigating", ev
    # Neutral mention (factor-list phrasing)
    for p in (r"犯\s*罪[之]?\s*手\s*段", r"手\s*段[、，,]\s*情\s*節",
              r"手\s*段(?:[、，,]\s*分?工?角色)?"):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_life_status(audit: str, reason: str) -> tuple[str, str]:
    """生活狀況 — neutral factual statement."""
    full = audit
    # Common life status phrases (all neutral)
    for p in (
        r"家\s*[境境]?(?:庭)?(?:經濟)?(?:狀況)?(?:勉持|小康|貧寒|富裕|尚可|清寒|普通)",
        r"經\s*濟\s*狀\s*況\s*(?:勉持|小康|貧寒|富裕|尚可|清寒|普通)",
        r"家\s*庭\s*[暨及](?:經濟)?(?:狀況)?",
        r"家\s*庭\s*經\s*濟",
        r"家\s*境[\s之]?[勉持小康貧寒尚]+",
        r"自\s*[陳述].{0,10}?(?:已婚|未婚|離婚|月入|月薪|從事|業[工農商醫]+)",
        r"從\s*事[^。，]{0,15}?(?:業|工作)",
        r"月\s*[薪入]\s*\d",
        r"家\s*中(?:無人)?[經需].{0,10}?扶\s*養",
        r"已\s*離\s*婚",
        r"未\s*婚",
        r"小\s*康(?:之)?(?:家庭)?",
        r"勉\s*持",
        r"生\s*活\s*狀\s*況",
        r"曾\s*從\s*事",
        r"陳\s*述[^。]{0,10}?(?:學歷|家庭)",
        r"待\s*業\s*中",
    ):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_character(audit: str, reason: str) -> tuple[str, str]:
    """品行/素行/前科 — drug priors framed as aggravating; clean record mitigating."""
    full = audit
    # Mitigating: clean record / good character
    for p in (
        r"前\s*無\s*犯\s*罪\s*紀\s*錄",
        r"未\s*曾\s*因\s*故\s*意\s*犯\s*罪",
        r"前\s*未\s*曾\s*[因受犯]",
        r"素\s*行\s*[良佳]+",
        r"素\s*行\s*尚\s*[佳良可]",
        r"無\s*前\s*科",
        r"並\s*無\s*前\s*科",
        r"無\s*犯\s*罪\s*前\s*科",
    ):
        ev = _find_window(full, p, 50)
        if ev:
            return "mitigating", ev
    # Aggravating: prior drug record framed as adverse
    for p in (
        r"前\s*案\s*紀\s*錄.{0,80}?(?:猶未能|仍未能|不知悔改|惡性|顯難|足認其自制力)",
        r"自\s*制\s*力\s*薄\s*弱",
        r"反\s*省[之心]+\s*不\s*足",
        r"反\s*省\s*之\s*心\s*不\s*足",
        r"前\s*科\s*累\s*累",
        r"屢\s*[犯次]+",
        r"惡\s*性\s*[非難重大]+",
        r"惡\s*性\s*非\s*輕",
        r"不\s*可\s*謂\s*無\s*惡\s*性",
        r"曾\s*因\s*施\s*用\s*毒\s*品.{0,80}?(?:猶未能|仍未能|不知悔改)",
        r"前\s*因.{0,20}?案\s*件.{0,40}?(?:猶未能|仍未能|不知悔改)",
        r"戒\s*毒\s*意\s*志\s*不\s*堅",
        r"毒\s*癮\s*非\s*淺",
        r"刑\s*罰\s*反\s*應[力性]+\s*[薄顯]?\s*弱",
        r"未\s*因\s*[先前]+.{0,20}?(?:遭查獲|遭判處|執行).{0,30}?(?:改弦更張|悔過改新)",
        r"並\s*未\s*改\s*弦\s*更\s*張",
        r"未\s*能\s*悔\s*過\s*改\s*新",
        r"故\s*態\s*復\s*萌",
        r"再\s*犯",
        r"觀\s*察\s*、?\s*勒\s*戒.{0,30}?(?:仍未|猶未|未能|不思|不知|未戒|再犯|再施用|又犯)",
        r"勒\s*戒\s*後.{0,30}?(?:仍未|猶未|未能|不思|不知)",
        r"戒\s*斷\s*決\s*心\s*[不薄欠]",
        r"再\s*三\s*施\s*用",
        r"屢\s*次\s*施\s*用",
        r"未\s*戒\s*除\s*毒\s*癮",
        r"不\s*思\s*悔\s*改",
        r"猶\s*再\s*[犯施]",
        r"竟\s*再\s*[犯施]",
        r"不\s*知\s*戒\s*除",
        r"一\s*再\s*施\s*用",
        r"竟\s*不\s*知\s*戒",
        r"前\s*已\s*因.{0,30}?(?:猶|仍|不|竟)",
        r"猶\s*未\s*能\s*深\s*切\s*體\s*認",
    ):
        ev = _find_window(full, p, 60)
        if ev:
            return "aggravating", ev
    # Neutral mention
    for p in (
        r"前\s*案\s*紀\s*錄\s*表",
        r"素\s*行(?:狀況)?",
        r"前\s*科\s*素\s*行",
        r"被\s*告\s*前\s*有",
        r"前\s*案",
    ):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_intellect(audit: str, reason: str) -> tuple[str, str]:
    """智識程度 — neutral factual statement."""
    full = audit
    for p in (
        r"智\s*識\s*程\s*度[^，。]{0,30}?(?:畢業|肄業)",
        r"(?:高中|高職|國中|國小|大學|碩士|博士|大專|專科|研究所|國民小學)\s*(?:畢業|肄業)",
        r"教\s*育\s*程\s*度[^，。]{0,30}?(?:畢業|肄業)",
        r"自\s*[陳述][^，。]{0,15}?(?:高中|高職|國中|國小|大學|碩士|博士|專科)",
        r"智\s*識\s*程\s*度",
        r"教\s*育\s*程\s*度",
        r"(?:畢業|肄業)\s*(?:之?\s*智\s*識\s*程\s*度)?",
        r"陳\s*述[^。]{0,5}?學\s*歷",
        r"自\s*述[^。]{0,5}?學\s*歷",
        r"學\s*歷[^。]{0,10}?(?:、|為)",
    ):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_relation_victim(audit: str, reason: str) -> tuple[str, str]:
    """與被害人關係 — almost always 'absent' in drug cases (no victim)."""
    full = audit
    if "被害人" in full:
        for p in (r"與\s*被\s*害\s*人[之]?\s*關\s*係",
                  r"與\s*被\s*害\s*人[^，。]{0,20}?(?:朋友|親屬|家人|無|關係)"):
            ev = _find_window(full, p, 50)
            if ev:
                return "neutral", ev
    return "absent", ""


def classify_duty_breach(audit: str, reason: str) -> tuple[str, str]:
    """違反義務之程度 — rare in drug cases."""
    full = audit
    for p in (r"違\s*反.{0,10}?義\s*務(?:之?\s*程\s*度)?", r"義\s*務\s*違\s*反"):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_harm(audit: str, reason: str) -> tuple[str, str]:
    """危害 — drug self-use: 戕害自身 (mitigating). Sales/transfer: 危害社會 (aggravating).

    NB: aggravating cues are matched FIRST. 戕害自身 is mitigating, but
    戕害國民/他人 is aggravating. Specificity matters — we check
    aggravating-戕害 patterns before mitigating-戕害 patterns.
    """
    full = audit
    # Aggravating — must check before mitigating (戕害自身 vs 戕害國民/他人)
    for p in (
        r"戕\s*害\s*國\s*民",
        r"戕\s*害\s*他\s*人",
        r"危\s*害\s*社\s*會[治安秩序善良風俗]+",
        r"對\s*社\s*會\s*[深具造成危害]+",
        r"嚴\s*重\s*危\s*害",
        r"危\s*害\s*[甚至重大匪非]+\s*[淺鉅]?",
        r"助\s*長\s*毒\s*品[流通蔓延氾濫風氣濫用]+",
        r"助\s*長\s*濫\s*用",
        r"影\s*響\s*社\s*會[秩序治安善良風俗]+",
        r"敗\s*壞\s*社\s*會",
        r"流\s*毒\s*甚\s*廣",
        r"危\s*害\s*非\s*淺",
        r"危\s*害\s*匪\s*淺",
        r"造\s*成\s*毒\s*品[之]?\s*流\s*通",
        r"足\s*以\s*戕\s*害",
        r"危\s*害[甚社會國家公共非]+",
        r"危\s*害[國民健康公]+",
        r"傷\s*害\s*社\s*會",
    ):
        ev = _find_window(full, p, 60)
        if ev:
            return "aggravating", ev
    # Mitigating
    for p in (
        r"戕\s*害\s*自\s*[身己]",
        r"戕\s*害\s*其\s*[個自身己]",
        r"戕\s*害\s*個\s*人",
        r"未\s*直\s*接\s*侵\s*害\s*他\s*人\s*法\s*益",
        r"以\s*自\s*我\s*身\s*心\s*侵\s*害\s*為\s*主",
        r"反\s*社\s*會\s*性[之]?\s*程\s*度\s*較\s*低",
        r"自\s*戕\s*行\s*為",
        r"未\s*對\s*他\s*人\s*造\s*成\s*危\s*害",
        r"對\s*他\s*人\s*法\s*益\s*尚\s*無\s*重\s*大\s*明\s*顯",
        r"未\s*流\s*傳\s*於\s*眾",
        r"非\s*難\s*性\s*較\s*低",
        r"所\s*生\s*損\s*害\s*尚\s*非\s*[鉅大重]+",
        r"戕\s*害[其自個][身心個人己]+",
    ):
        ev = _find_window(full, p, 60)
        if ev:
            return "mitigating", ev
    # Neutral
    for p in (r"危\s*害", r"損\s*害", r"所\s*生[之]?\s*危\s*害"):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_post_attitude(audit: str, reason: str) -> tuple[str, str]:
    """犯後態度 — 坦承 (mitigating), 否認 (aggravating)."""
    full = audit
    # Aggravating: 否認 / 矢口否認 / 無悔意
    for p in (
        r"矢\s*口\s*否\s*認",
        r"否\s*認\s*犯\s*行(?!.{0,10}?(?:不諱|然))",
        r"飾\s*詞\s*狡\s*辯",
        r"態\s*度\s*[不欠]\s*[佳良]",
        r"無\s*[悔愧]\s*意",
        r"毫\s*無\s*悔\s*意",
        r"未\s*見\s*悔\s*[意悟]+",
    ):
        ev = _find_window(full, p, 60)
        if ev:
            return "aggravating", ev
    # Mitigating: 坦承 / 已見悔意 / 態度良好
    for p in (
        r"坦\s*[承認]\s*犯\s*行",
        r"承\s*認\s*犯\s*行",
        r"坦\s*承\s*(?:不諱|本案)",
        r"坦\s*[認承]\s*不\s*諱",
        r"自\s*白\s*犯\s*行",
        r"已\s*見\s*悔\s*意",
        r"態\s*度\s*[尚良好可佳]+",
        r"知\s*所\s*悔\s*悟",
        r"確\s*有\s*悔\s*意",
        r"態\s*度\s*尚\s*可",
        r"態\s*度\s*良\s*好",
        r"犯\s*[後罪]\s*(?:坦承|認罪|自白|承認)",
        r"犯\s*罪\s*後\s*[之]?(?:坦承|認罪|承認)",
        r"於\s*[偵警審][中時]?\s*(?:坦承|自白|認罪|承認)\s*[犯本不]",
        r"被\s*告\s*[認坦]\s*罪",
    ):
        ev = _find_window(full, p, 60)
        if ev:
            return "mitigating", ev
    # Neutral mention
    for p in (r"犯\s*後\s*態\s*度", r"犯\s*罪\s*後[之]?\s*態\s*度"):
        ev = _find_window(full, p, 50)
        if ev:
            return "neutral", ev
    return "absent", ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FACTORS = (
    ("motive", classify_motive),
    ("provocation", classify_provocation),
    ("means", classify_means),
    ("life_status", classify_life_status),
    ("character", classify_character),
    ("intellect", classify_intellect),
    ("relation_victim", classify_relation_victim),
    ("duty_breach", classify_duty_breach),
    ("harm", classify_harm),
    ("post_attitude", classify_post_attitude),
)


def _normalize(text: str) -> str:
    """Collapse all whitespace (incl. full-width 　 and newlines) for matching."""
    return re.sub(r"[\s　]+", "", text)


def classify_case(reason: str) -> dict:
    norm_reason = _normalize(reason)
    audit_raw = extract_audit_block(reason)
    audit = _normalize(audit_raw)
    out = {}
    for key, fn in FACTORS:
        direction, evidence = fn(audit, norm_reason)
        # Validate: evidence must be substring of normalized reason
        if evidence:
            evidence = _normalize(evidence)
            if evidence not in norm_reason:
                evidence = ""
                direction = "absent"
        # Final length clamp
        if evidence and len(evidence) > 80:
            evidence = evidence[:80]
        if evidence and len(evidence) < 20:
            # Try to expand by finding the position in norm_reason and grabbing a wider window
            pos = norm_reason.find(evidence)
            if pos >= 0:
                ws = max(0, pos - (40 - len(evidence) // 2))
                we = min(len(norm_reason), pos + len(evidence) + (40 - len(evidence) // 2))
                evidence = norm_reason[ws:we][:80]
        out[key] = {"direction": direction, "evidence": evidence}
    return out


def main() -> None:
    src = Path(r"C:\Users\alex2\Desktop\vsCode\AiJudge\data\filtered\keelung_drug_all.jsonl")
    out = Path(r"/tmp/art57_batch_2.jsonl")
    start, end = 320, 640

    with src.open(encoding="utf-8") as fin:
        rows = [json.loads(l) for l in fin]
    sub = rows[start:end]
    out.parent.mkdir(parents=True, exist_ok=True)

    empty_reason: list[str] = []
    written = 0
    with out.open("w", encoding="utf-8") as fout:
        for d in sub:
            jfull = d.get("jfull", "")
            jid = d.get("jid", "")
            reason = extract_reason(jfull)
            if len(reason) < 50:
                empty_reason.append(jid)
            factors = classify_case(reason)
            row = {"jid": jid, "factors": factors}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

    print(f"wrote {written} rows to {out}")
    if empty_reason:
        print(f"empty/short reason cases ({len(empty_reason)}):")
        for j in empty_reason:
            print("  ", j)
    else:
        print("no empty/short reason cases")


if __name__ == "__main__":
    main()
