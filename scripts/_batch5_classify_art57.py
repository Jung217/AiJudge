"""§57 factor classifier for batch 5 (idx 1280-1597 inclusive).

Reads /tmp/art57_batch_5_reasons.jsonl, classifies 10 factors per case
based on stereotyped phrasing in Taiwan drug judgments, writes
/tmp/art57_batch_5.jsonl (one JSON object per line).

Strategy: drug judgments use a small set of stock phrases for each §57
factor. We pattern-match against those phrases and return:
  - direction: mitigating / aggravating / neutral / absent
  - evidence: 20-80 char EXACT substring from reason text
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

INPUT = Path("/tmp/art57_batch_5_reasons.jsonl")
OUTPUT = Path("/tmp/art57_batch_5.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trim_ev(text: str, start: int, end: int, *, target: int = 50) -> str:
    """Return a 20-80 char exact substring centered on [start:end].

    Expands or contracts to reach roughly `target` length while preserving
    that the result is a substring of `text`.
    """
    n = len(text)
    s, e = max(0, start), min(n, end)
    cur = e - s
    # Pad symmetrically up to target
    while cur < 25 and (s > 0 or e < n):
        if s > 0:
            s -= 1
            cur += 1
        if cur >= 25:
            break
        if e < n:
            e += 1
            cur += 1
    # Cap at 80
    if e - s > 80:
        e = s + 80
    # Strip leading/trailing whitespace within bounds without going
    # outside; trimming whitespace ALSO must remain a valid substring,
    # which it always does (substr of substr).
    sub = text[s:e]
    return sub


def find_evidence(text: str, *patterns: str, target: int = 50) -> Optional[str]:
    """Return first matching evidence substring or None."""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            mid = (m.start() + m.end()) // 2
            ev = trim_ev(text, mid - target // 2, mid + target // 2, target=target)
            if 20 <= len(ev) <= 80:
                return ev
            # Otherwise expand
            return trim_ev(text, m.start(), m.end(), target=target)
    return None


def evidence_around(text: str, anchor_pat: str, *, before: int = 25, after: int = 25) -> Optional[str]:
    m = re.search(anchor_pat, text)
    if not m:
        return None
    s = max(0, m.start() - before)
    e = min(len(text), m.end() + after)
    sub = text[s:e]
    if len(sub) > 80:
        sub = sub[:80]
    return sub if len(sub) >= 20 else None


# ---------------------------------------------------------------------------
# Quanxing paragraph extractor
# ---------------------------------------------------------------------------

def extract_quanxing(reason: str) -> str:
    """Best-effort extraction of the sentencing-rationale paragraph.

    Falls back to the full reason if no anchor is found.
    """
    # Search for typical anchors of the 量刑 paragraph
    anchors = [
        r"爰\s*[以審]\s*[行審]",
        r"爰\s*審\s*酌",
        r"爰\s*以\s*行\s*為\s*人",
        r"審\s*酌\s*被\s*告",
        r"本院依刑法第57條",
    ]
    earliest = -1
    for pat in anchors:
        m = re.search(pat, reason)
        if m and (earliest < 0 or m.start() < earliest):
            earliest = m.start()
    if earliest < 0:
        return reason
    # Walk forward to find end (量處如主文 / 諭知 / next 三/四 numbered section)
    tail = reason[earliest:]
    end_anchors = [
        r"以資懲儆",
        r"以示懲儆",
        r"以示警惕",
        r"資以懲儆",
        r"以資警惕",
        r"用以鼓勵",
        r"標準\s*。",
        r"如易科罰金之折算\n?\s*標準",
        r"\n\s*(?:三|四|五|六)\s*、\s*(?:沒收|據|依|扣案|據上)",
        r"\n\s*[（(]\s*[三四]\s*[）)]\s*沒收",
    ]
    end = len(tail)
    for pat in end_anchors:
        m = re.search(pat, tail)
        if m and 200 < m.end() < end + 200:
            end = min(end, m.end() + 50)
    end = min(end, 2000)
    return tail[:end]


# ---------------------------------------------------------------------------
# Factor classifiers — one per §57 factor.
# Each returns (direction, evidence)
# ---------------------------------------------------------------------------

def cls_motive(q: str, r: str) -> tuple[str, str]:
    # Drug cases routinely cite 動機/目的 as part of stock list. Without an
    # explicit framing it's neutral fact.
    pats = [
        r"基於施用毒品之動機[^，。]{0,40}",
        r"基於販賣[^。]{0,15}毒品[^。]{0,20}犯意",
        r"意圖營利",
        r"犯罪\s*動機[、，]?\s*目的",
        r"犯罪[之的]?動機[、，]?\s*目的",
        r"動\s*機[、，]\s*目\s*的",
        r"動\s*機\s*[、，]?\s*手\s*段",
        r"其犯罪[之的]?動機",
        r"動機[及與]目的",
        r"犯罪[之的]?動機",
        r"動\s*機",
    ]
    for pat in pats:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=40)
            # Profit motive (販賣) is aggravating, simple drug use is neutral
            if "營利" in ev or "販賣" in ev or "意圖" in ev:
                return ("aggravating", ev)
            return ("neutral", ev)
    return ("absent", "")


def cls_provocation(q: str, r: str) -> tuple[str, str]:
    # 受刺激: rarely mentioned in drug cases; often part of stock §57 list
    pats = [
        r"所\s*受\s*[之]?\s*刺\s*激",
        r"受刺激",
    ]
    for pat in pats:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=40)
            return ("neutral", ev)
    return ("absent", "")


def cls_means(q: str, r: str) -> tuple[str, str]:
    # 手段: 平和 = mitigating; 兇殘 = aggravating; mere mention = neutral
    pats_mit = [
        r"犯罪手段尚屬平和",
        r"犯罪手段[尚屬非]?平和",
        r"手段尚屬平和",
        r"以\s*[網路通訊]{1,8}\s*交易",  # neutral
    ]
    for pat in pats_mit[:3]:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=40)
            return ("mitigating", ev)
    pats_agg = [
        r"手段[兇殘]",
        r"手段惡劣",
    ]
    for pat in pats_agg:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=40)
            return ("aggravating", ev)
    pats_neu = [
        r"犯罪[之的]?[動機目的手段，、]+",
        r"手\s*段[、，]",
        r"所\s*用\s*之\s*手\s*段",
        r"犯罪[之的]?手段",
        r"動機[、，]\s*目的[、，]\s*手段",
        r"目\s*的[、，]\s*手\s*段",
        r"動機[及與]方法",
        r"手段",
    ]
    for pat in pats_neu:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=40)
            return ("neutral", ev)
    return ("absent", "")


def cls_life_status(q: str, r: str) -> tuple[str, str]:
    # 生活狀況/家庭經濟: usually 自陳 ... 勉持/小康/貧寒/良好
    # Also: 需扶養 (mitigating empathy), 經濟貧寒 (mitigating)
    pats_mit = [
        r"需\s*扶\s*養[^。]{0,30}",
        r"須\s*扶\s*養[^。]{0,30}",
        r"扶\s*養[^。]{0,15}(?:母親|外公|父親|妻|子女|幼)",
        r"需\s*照\s*顧[^。]{0,30}",
        r"經濟貧[寒困]",
        r"家境清寒",
        r"獨力扶養",
    ]
    for pat in pats_mit:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("mitigating", ev)
    pats_neu = [
        r"家庭經濟狀況[勉小普良中富][持康通好裕]",
        r"家境[勉小普]\s*[持康通]",
        r"家\s*境\s*勉\s*持",
        r"勉強維持\s*之?家庭經濟",
        r"自[述陳][^。]{0,30}家庭經濟",
        r"家庭經濟狀況",
        r"家\s*庭\s*經\s*濟",
        r"經濟狀況",
        r"家庭狀況",
        r"生\s*活\s*狀\s*況",
        r"從事[^。]{0,15}業[^。]{0,5}[，、。]",
        r"業[工農服商粗園美調無待]",
        r"自陳[^。]{0,15}從事",
        r"業\s*[工農粗服]",
        r"未婚無子女",
        r"職業",
        r"家境",
    ]
    for pat in pats_neu:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("neutral", ev)
    return ("absent", "")


def cls_character(q: str, r: str) -> tuple[str, str]:
    # 品行/前科素行: 前科 always there in drug cases — typically
    # framed with "猶未能深切體認" → aggravating tone
    # Strong 累犯/再犯 phrasing is aggravating
    pats_agg = [
        r"猶未能深切體認",
        r"惡習已深",
        r"再犯性極高",
        r"自制力薄弱",
        r"戒毒[意決]志不[堅強]",
        r"戒毒決心不[堅強足]",
        r"未能戒除毒癮",
        r"足見戒毒意志不堅",
        r"故態復萌",
        r"屢次再犯",
        r"無法戒除毒癮",
        r"不[知思]戒除",
        r"不[知思]悔改",
        r"未能體悟施用毒品",
        r"再三施用",
        r"再為[^。]{0,15}本案犯行",
        r"再為本案",
        r"前案執行產生警惕",
        r"未因前案[^。]{0,15}警惕",
        r"對於刑罰[之的]?反應力",
        r"前案紀錄[^。]{0,30}再犯",
        r"前有[^。]{0,20}前科",
        r"前因施用毒品[^。]{0,20}執行完畢",
        r"曾[經有]\s*[因觀][^。]{0,15}觀察",
        r"曾經觀察、勒戒",
        r"曾經觀察勒戒",
        r"曾有觀察勒戒",
        r"前[已有][^。]{0,15}觀察",
        r"輕\s*忽\s*毒\s*品",
        r"未[見能]戒除[惡習毒癮]",
        r"未[見能]\s*[徹其]?戒除[^。]{0,15}[惡毒]",
        r"未\s*徹\s*底\s*戒\s*除",
        r"未認清毒品",
        r"猶未認清",
        r"業受[^。]{0,15}寬典",
        r"竟仍[基於再]",
        r"素行不[佳良]",
        r"惡性非輕",
    ]
    for pat in pats_agg:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("aggravating", ev)
    # 素行良好 / 無前科 → mitigating
    pats_mit = [
        r"素行良好",
        r"無[相同類]前科",
        r"未曾犯罪",
        r"並無前科",
    ]
    for pat in pats_mit:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("mitigating", ev)
    pats_neu = [
        r"[前]?[科素][行]?[（(]?[參見有][^。]{0,15}前案紀錄表",
        r"前科素行",
        r"前\s*案\s*紀\s*錄",
        r"素\s*行[、，（(]",
        r"法院前案紀錄表",
        r"前科\s*[、，]",
        r"素行",
    ]
    for pat in pats_neu:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("neutral", ev)
    return ("absent", "")


def cls_intellect(q: str, r: str) -> tuple[str, str]:
    pats = [
        r"國\s*[小中]\s*(?:畢業|肄業)",
        r"國民?\s*中\s*[畢肄]業",
        r"高\s*[中職]\s*(?:畢業|肄業)",
        r"高中肄[學業]",
        r"五專[^。]{0,8}[畢肄]業",
        r"專科[^。]{0,8}[畢肄]業",
        r"大學[^。]{0,8}[畢肄]業",
        r"研究所[^。]{0,8}[畢肄]業",
        r"國\s*中\s*畢",
        r"教育程度[^。]{0,20}",
        r"智\s*識\s*程\s*度",
        r"學\s*歷[（(][^)]{0,25}[）)]",
        r"自[述陳][^。]{0,15}智識",
        r"自[述陳][^。]{0,15}學歷",
        r"學\s*歷",
        r"智識正常",
    ]
    for pat in pats:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("neutral", ev)
    return ("absent", "")


def cls_relation_victim(q: str, r: str) -> tuple[str, str]:
    # In solo drug cases (施用/持有/販賣) there's no victim. Default absent.
    # Only for 妨害公務/傷害 etc. would victim relation appear.
    pats = [
        r"與被害人[^。]{0,15}關係",
        r"被害人[^。]{0,15}關係",
    ]
    for pat in pats:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("neutral", ev)
    return ("absent", "")


def cls_duty_breach(q: str, r: str) -> tuple[str, str]:
    # 違反義務之程度: rarely cited in drug cases. Sometimes 漠視/無視 法令禁制
    pats_agg = [
        r"漠視法令禁制",
        r"漠視國家[^。]{0,20}禁令",
        r"無視[^。]{0,15}禁令",
        r"無視國家對於杜絕毒品",
    ]
    for pat in pats_agg:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("aggravating", ev)
    return ("absent", "")


def cls_harm(q: str, r: str) -> tuple[str, str]:
    # 戕害自身 = mitigating (limited harm)
    # 危害社會 / 損害甚鉅 = aggravating
    pats_mit = [
        r"僅[有]?[戕殘]\s*害[自其][身己我]",
        r"[戕殘]\s*害\s*[自其]\s*[身己我]",
        r"[戕殘]\s*害\s*[其]?\s*個\s*人\s*身[心健]",
        r"[戕殘傷]\s*害\s*[自其]?\s*[個]?\s*人\s*[個]?\s*健\s*康",
        r"[殘戕]\s*害\s*[自其]\s*身",
        r"戕\s*害\s*自[己我]身",
        r"戕\s*害\s*自\s*身",
        r"戕\s*害\s*自\s*我\s*身\s*心",
        r"未[直]?\s*[接造]?\s*[成及]?\s*危\s*害\s*他\s*人",
        r"未[有對直]+\s*危[害及]",
        r"未\s*及\s*於\s*他\s*人",
        r"未\s*侵\s*犯\s*其\s*他\s*法\s*益",
        r"尚\s*未[直造]?\s*[接成]?\s*[影危]\s*響?\s*他\s*人",
        r"自\s*我\s*身\s*心\s*侵\s*害",
        r"反\s*社\s*會\s*性[之的]?\s*程\s*度\s*較\s*低",
        r"所\s*生[之]?\s*[危損]?\s*[害損]?\s*尚[非無]?\s*重\s*大",
        r"所\s*生\s*損\s*害\s*尚\s*非\s*鉅\s*大",
        r"未\s*流\s*傳\s*於\s*眾",
        r"對\s*他\s*人\s*法\s*益\s*尚\s*無\s*重\s*大",
        r"對\s*社\s*會\s*[公]?\s*共\s*[秩安]?\s*[序安]?",
        r"自\s*戕\s*行\s*為",
        r"自[傷殘戕]\s*行\s*為",
        r"屬\s*戕\s*害\s*自[己我]",
        r"乃\s*戕\s*害",
        r"屬\s*自\s*戕",
        r"自\s*我\s*戕\s*害",
        r"戕\s*害\s*其\s*個\s*人",
    ]
    for pat in pats_mit:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("mitigating", ev)
    pats_agg = [
        r"危害(?:甚|非)\s*[鉅劇大重]",
        r"嚴重戕害國民",
        r"流毒所及",
        r"危害社會非輕",
        r"危及他人",
        r"製造社會風氣[、，]\s*治安",
        r"惡化社會治安",
        r"助長毒品[之的]?蔓延",
        r"助長社會[上]?施用毒品",
        r"所生[之]?危害[非甚]\s*[輕鉅]",
        r"危害社會治安",
        r"影響社會[秩治]?[序安]",
        r"直接戕害國民身心",
        r"間接危害社會治安",
        r"敗壞社會",
        r"製造[社毒]",
        r"危害甚[深鉅大重]",
        r"傷害自身健康甚鉅",
    ]
    for pat in pats_agg:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("aggravating", ev)
    pats_neu = [
        r"所\s*生\s*[之]?\s*危\s*害",
        r"所\s*生\s*[之]?\s*損\s*害",
        r"危害",
    ]
    for pat in pats_neu:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("neutral", ev)
    return ("absent", "")


def cls_post_attitude(q: str, r: str) -> tuple[str, str]:
    # 坦承 / 坦認 = mitigating
    # 否認 / 矢口否認 = aggravating
    # 部分坦承 = neutral-ish, but usually treated mitigating
    pats_agg = [
        r"否\s*認\s*犯\s*行",
        r"矢\s*口\s*否\s*認",
        r"無[悔反][意省]",
        r"後\s*改\s*為\s*否\s*認",
        r"飾\s*詞\s*狡\s*辯",
        r"否\s*認\s*施\s*用",
        r"否\s*認\s*本\s*案",
        r"否\s*認\s*犯\s*罪",
        r"否\s*認[^。]{0,15}態\s*度",
        r"於\s*警\s*詢\s*之\s*否\s*認",
        r"警\s*詢\s*時[^。]{0,10}否\s*認",
        r"否\s*認[^。]{0,15}犯\s*行",
    ]
    for pat in pats_agg:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("aggravating", ev)
    pats_mit = [
        r"坦\s*承\s*犯\s*行[^。]{0,25}",
        r"始\s*終\s*坦[認承]",
        r"坦\s*[承認][^。]{0,15}犯[行罪]",
        r"坦\s*[承認]\s*犯[行罪]",
        r"犯\s*後\s*態\s*度\s*尚?[佳可良好]",
        r"態\s*度\s*尚[可佳良]",
        r"見\s*悔\s*意",
        r"已\s*坦[認承]",
        r"終\s*能\s*坦",
        r"自\s*白\s*[犯不]?[行諱]",
        r"認\s*罪\s*態\s*度",
        r"坦\s*白",
        r"尚\s*見\s*悔\s*意",
        r"自\s*[首白]",
        r"自\s*首",
        r"坦\s*[承認]",
        r"犯\s*後[態度]?[尚]?[佳可良]",
    ]
    for pat in pats_mit:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("mitigating", ev)
    pats_neu = [
        r"犯\s*後\s*[態度]",
        r"犯[罪後]\s*態\s*度",
        r"犯\s*後\s*之\s*態\s*度",
        r"犯\s*罪\s*後\s*之?\s*態\s*度",
    ]
    for pat in pats_neu:
        m = re.search(pat, q)
        if m:
            ev = trim_ev(q, m.start(), m.end(), target=50)
            return ("neutral", ev)
    return ("absent", "")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

CLASSIFIERS = [
    ("motive", cls_motive),
    ("provocation", cls_provocation),
    ("means", cls_means),
    ("life_status", cls_life_status),
    ("character", cls_character),
    ("intellect", cls_intellect),
    ("relation_victim", cls_relation_victim),
    ("duty_breach", cls_duty_breach),
    ("harm", cls_harm),
    ("post_attitude", cls_post_attitude),
]


def classify_case(reason: str) -> dict:
    q = extract_quanxing(reason)
    factors = {}
    for name, fn in CLASSIFIERS:
        direction, evidence = fn(q, reason)
        # Clamp evidence to 20-80 chars
        if direction != "absent":
            if len(evidence) < 20:
                # Try expanding from full reason
                if evidence in reason:
                    idx = reason.find(evidence)
                    s = max(0, idx - 5)
                    e = min(len(reason), idx + len(evidence) + 25)
                    candidate = reason[s:e]
                    if len(candidate) >= 20 and candidate in reason:
                        evidence = candidate
            if len(evidence) > 80:
                evidence = evidence[:80]
            # Sanity: evidence MUST be substring of reason (not just q)
            if evidence and evidence not in reason:
                # try q membership
                if evidence not in q:
                    evidence = ""
                    direction = "absent"
        else:
            evidence = ""
        factors[name] = {"direction": direction, "evidence": evidence}
    return factors


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:
        n = 0
        for line in fin:
            obj = json.loads(line)
            factors = classify_case(obj["reason"])
            out = {"jid": obj["jid"], "factors": factors}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
    print(f"wrote {n} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
