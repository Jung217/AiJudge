"""Pattern-based §57 factor labeler for Taiwanese drug judgments.

Inspired by Claude reasoning over the highly formulaic phrasing in basic
drug-case sentencing portions. Each factor's direction+evidence is
detected via a small set of high-precision regexes; the evidence is
ALWAYS an exact substring of the reason text.

Output: jsonl with one row per case (idx 640..959).
"""
import json
import re
import sys
from pathlib import Path

INPUT_REASONS = Path("/tmp/art57_batch_3_reasons.jsonl")
INPUT_SENTENCING = Path("/tmp/art57_batch_3_sentencing.jsonl")
OUTPUT = Path("/tmp/art57_batch_3.jsonl")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def first_match(patterns, text):
    """Return the first regex match in text, or None."""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m
    return None


def keyword_present(text: str, kw: str) -> bool:
    """Whitespace-tolerant keyword check: 'X' matches even if 'X' is split
    across line breaks / indentation / 全形 spaces in text.
    """
    pat = r"\s*".join(re.escape(c) for c in kw)
    return bool(re.search(pat, text))


def trim_evidence(s: str, lo: int = 20, hi: int = 80) -> str:
    """Clean evidence string: strip whitespace runs / newlines, ensure 20-80 chars."""
    # Normalize whitespace (including 全形 and CR/LF) but keep characters in order
    s = re.sub(r"[\s　]+", "", s)
    if len(s) > hi:
        s = s[:hi]
    return s


def find_substr_window(reason: str, target: str, lo: int = 20, hi: int = 80) -> str:
    """Locate `target` (whitespace-tolerant) in reason and return a 20-80 char
    window AROUND it that's an EXACT substring of reason (preserving the
    original text, including whitespace/newlines).

    Strategy: find target's position in reason via whitespace-tolerant regex,
    then expand the window around it to reach 20-80 effective chars after
    whitespace normalization.
    """
    # Build a whitespace-tolerant pattern for target
    target_clean = re.sub(r"\s+", "", target)
    if not target_clean:
        return ""
    # Pattern: between every char allow optional whitespace
    pat = r"\s*".join(re.escape(c) for c in target_clean)
    m = re.search(pat, reason)
    if not m:
        return ""
    start, end = m.start(), m.end()
    # Expand left and right to reach ~50 effective chars (after stripping whitespace)
    desired_chars = 40
    # Expand by character count, accounting for whitespace
    # Try to include some context on either side
    before_room = start
    after_room = len(reason) - end
    pad = max(10, (desired_chars - (end - start)) // 2)
    s2 = max(0, start - pad)
    e2 = min(len(reason), end + pad)
    raw = reason[s2:e2]
    cleaned = re.sub(r"[\s　]+", "", raw)
    # Ensure 20..80
    if len(cleaned) < lo:
        # widen further
        e2 = min(len(reason), e2 + lo)
        s2 = max(0, s2 - lo)
        raw = reason[s2:e2]
        cleaned = re.sub(r"[\s　]+", "", raw)
    if len(cleaned) > hi:
        # shrink: keep target centered
        # Find target position in cleaned
        target_pos = cleaned.find(target_clean)
        if target_pos < 0:
            cleaned = cleaned[:hi]
        else:
            # Center on target
            tlen = len(target_clean)
            left_keep = max(0, (hi - tlen) // 2)
            new_start = max(0, target_pos - left_keep)
            cleaned = cleaned[new_start:new_start + hi]
    return cleaned


def find_first_substring_evidence(reason: str, target: str, lo: int = 20, hi: int = 80) -> str:
    """Same as find_substr_window but ensures EXACT substring of reason
    (including whitespace) — needed because evidence must be exact substring.
    """
    target_clean = re.sub(r"\s+", "", target)
    if not target_clean:
        return ""
    pat = r"\s*".join(re.escape(c) for c in target_clean)
    m = re.search(pat, reason)
    if not m:
        return ""
    start, end = m.start(), m.end()
    # Expand window to reach 20+ chars (counting actual chars in reason slice)
    pad = 20
    s2 = max(0, start - pad)
    e2 = min(len(reason), end + pad)
    # The substring length we want: 20-80 chars
    while e2 - s2 < lo and (s2 > 0 or e2 < len(reason)):
        if s2 > 0:
            s2 -= 1
        if e2 < len(reason):
            e2 += 1
    while e2 - s2 > hi:
        # Shrink: prefer to keep target centered
        if (start - s2) > (e2 - end):
            s2 += 1
        else:
            e2 -= 1
    return reason[s2:e2]


# -----------------------------------------------------------------------
# Factor extraction patterns
# -----------------------------------------------------------------------

# post_attitude (犯後態度)
POST_MITIGATING_KW = [
    "坦承犯行", "坦認犯行", "坦承施用", "承認犯行", "坦承本件", "坦承全部犯行",
    "犯後態度尚佳", "犯後態度良好", "犯後態度尚可", "態度良好", "態度尚可", "態度尚佳",
    "犯罪後坦承", "犯後坦承", "犯後坦認", "坦認施用", "坦承全部",
    "已坦承", "尚見悔意", "尚有悔悟", "有悔悟", "犯罪後已坦承", "尚知自首",
    "悔悟", "自首", "自白", "犯後自首", "見悔意", "犯後悔悟",
    "有悔意", "已知悔悟", "示悔意", "尚有悔意", "態度尚良好",
]
POST_AGGRAVATING_KW = [
    "未坦認犯行", "否認犯行", "矢口否認", "未坦承犯行", "未坦認本次", "未到案坦承",
    "犯後竟未坦承", "犯後始終否認", "犯後仍否認", "未坦承本次施用",
    "經傳喚未到庭", "犯後否認", "態度難謂良好", "態度未佳",
    "未到庭說明", "態度不佳",
    "犯後飾詞卸責", "飾詞卸責", "卸責之詞",
    "態度輕慢", "卸責態度",
    "犯後並無悔意", "未見悔意", "毫無悔意",
    "堅不認罪",
]


# means (手段)
MEANS_MITIGATING_KW = [
    "犯罪手段尚屬平和", "手段尚屬平和", "手段平和", "手段非暴力",
    "犯罪手段非暴力", "手段非屬殘暴",
]
MEANS_AGGRAVATING_KW = [
    "手段兇殘", "手段殘忍", "手段惡劣",
]


# harm (危害損害)
HARM_MITIGATING_KW = [
    "戕害自身", "戕害己身", "戕害自我", "戕害自己身心",
    "戕害施用者", "對他人法益尚無重大", "尚未直接危害他人",
    "未對他人造成危害", "未侵犯其他法益", "未危及他人",
    "對他人法益尚無", "對社會造成之危害尚非", "並未危及他人",
    "施用毒品僅係戕害", "本質仍屬自殘行為", "本質上乃屬戕害",
    "未流傳於眾", "對他人法益及社會安全尚無重大",
    "尚未造成他人具體危害", "未造成他人具體危害",
    "戕害自我身心", "對他人法益尚無重大明顯之實害",
    "尚未有嚴重破壞社會秩序", "所生損害尚非鉅大",
]
HARM_AGGRAVATING_KW = [
    "危害甚鉅", "危害甚深", "戕害甚鉅", "為害之鉅",
    "對社會秩序有相當程度之危害", "潛在危害",
    "輕則戕害施用者", "重則引發各種犯罪",
    "為社會治安敗壞之源頭", "對社會風氣、治安",
    "助長毒品散布",
]


# motive
MOTIVE_AGGRAVATING_KW = [
    "貪圖獲利", "貪圖私利", "為貪圖獲利", "牟利",
    "為私利", "獲利動機",
]


# character (品行 / 素行 / 前科)
CHARACTER_AGGRAVATING_KW = [
    "素行不佳", "素行非佳", "素行不良",
    "對刑罰反應力薄弱", "對刑罰之反應力顯然薄弱",
    "毒癮非淺", "戒毒意志不堅", "屢次施用",
    "再三施用", "故態復萌", "缺乏戒斷決心",
    "自制力薄弱", "未能深切體認", "猶未戒除",
    "刑罰反應力", "再犯本案", "猶未能深切體認",
]


# life_status (生活狀況)
LIFE_NEUTRAL_KW = [
    "家境勉持", "家庭經濟狀況勉持", "經濟勉持", "勉持之家庭",
    "家境小康", "家庭經濟狀況小康", "經濟小康", "小康之家庭", "家庭狀況小康",
    "家境貧窮", "家境貧困", "家庭經濟狀況貧寒", "家庭經濟狀況貧困", "經濟貧困",
    "家境普通", "家庭經濟狀況普通",
    "業工", "從事服務業", "從事粗工",
    "離婚", "已婚", "未婚",
    "勉強維持", "勉強維持之家庭",
]


# intellect (智識程度)
INTELLECT_NEUTRAL_KW = [
    "智識程度", "教育程度", "國小畢業", "國小肄業", "國中畢業", "國中肄業",
    "高中畢業", "高中肄業", "高職畢業", "高職肄業", "大學畢業", "大學肄業",
    "二、三專", "二三專", "專科畢業", "研究所", "高職異業",
]


# duty_breach (違反義務之程度) — drug cases generally don't apply this
DUTY_BREACH_AGGRAVATING_KW = [
    "違反義務之程度", "違反義務", "違反法定義務",
]


# provocation (受刺激) — rare in drug cases
PROVOCATION_KW = [
    "受刺激", "受挑釁", "因刺激",
]


# -----------------------------------------------------------------------
# Per-factor classifiers (return (direction, evidence))
# -----------------------------------------------------------------------

def classify_post_attitude(reason: str) -> tuple[str, str]:
    # Aggravating wins if present (denial overrides confession)
    for kw in POST_AGGRAVATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "aggravating", ev
    for kw in POST_MITIGATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "mitigating", ev
    # Neutral: bare mention without characterization
    for kw in ["犯後態度", "犯罪後之態度"]:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    return "absent", ""


def classify_means(reason: str) -> tuple[str, str]:
    for kw in MEANS_AGGRAVATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "aggravating", ev
    for kw in MEANS_MITIGATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "mitigating", ev
    # Stock phrase 「兼衡其犯罪之動機、目的、手段」 — neutral mention
    means_neutral_kws = [
        "動機、目的、手段", "目的、手段",
        "犯罪手段", "之手段", "犯罪之手段",
    ]
    for kw in means_neutral_kws:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    return "absent", ""


def classify_harm(reason: str) -> tuple[str, str]:
    # Mitigating phrases (drug cases mostly self-harm)
    for kw in HARM_MITIGATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "mitigating", ev
    # Aggravating
    for kw in HARM_AGGRAVATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "aggravating", ev
    return "absent", ""


def classify_motive(reason: str) -> tuple[str, str]:
    for kw in MOTIVE_AGGRAVATING_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "aggravating", ev
    # Neutral if any of the stock phrases mention 動機 / 目的
    motive_neutral_kws = [
        "犯罪動機", "犯罪之動機", "動機、目的", "犯罪之動機、目的",
        "犯罪動機、目的", "其犯罪動機", "其犯罪之動機",
        "犯罪之目的", "犯罪目的", "施用毒品之目的",
    ]
    for kw in motive_neutral_kws:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    return "absent", ""


def classify_character(reason: str) -> tuple[str, str]:
    # Strong aggravating: 累犯, 對刑罰反應力薄弱, 素行不佳, 漠視, 戒毒不力
    strong_agg = [
        "素行不佳", "素行不良", "素行非佳",
        "對刑罰反應力薄弱", "對刑罰之反應力顯然薄弱",
        "戒毒意志不堅", "毒癮非淺",
        "故態復萌", "缺乏戒斷決心",
        "再三施用", "屢次施用", "多次施用",
        "未能改過", "猶未能深切體認",
        "漠視法令禁制", "漠視法令", "輕忽毒品",
        "未見其戒除惡習", "戒除惡習", "未戒除毒癮",
        "仍不知戒除", "故仍不思悔改", "仍不思悔改",
        "顯見被告無法戒除毒癮",
        "猶未戒除", "猶未深切體認",
        "再為本案", "再犯本案",
        "復施用第", "再為本案犯行",
        "未能戒除",
    ]
    for kw in strong_agg:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "aggravating", ev
    # Neutral: bare mention of 素行 / 前科紀錄
    if "素行" in reason and ("前案紀錄表" in reason or "前科" in reason):
        ev = find_first_substring_evidence(reason, "素行")
        if ev:
            return "neutral", ev
    # Bare 前案紀錄表 mention
    if "前案紀錄" in reason:
        ev = find_first_substring_evidence(reason, "前案紀錄")
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_life_status(reason: str) -> tuple[str, str]:
    # life_status phrases are typically neutral
    keywords_priority = [
        "家境勉持", "家庭經濟狀況勉持", "經濟狀況勉持", "經濟勉持", "勉持之家庭",
        "家境小康", "家庭經濟狀況小康", "經濟狀況小康", "經濟小康",
        "家境貧窮", "家境貧困", "經濟狀況貧寒", "經濟狀況貧困",
        "貧寒", "貧困", "勉強維持",
        "家境普通", "家庭經濟狀況普通",
        "家庭經濟狀況", "家庭生活、經濟狀況", "生活、經濟狀況",
        "家庭情形", "家庭狀況",
    ]
    for kw in keywords_priority:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    # Marriage/family-config indicators
    fam_kws = ["未婚", "已婚", "離婚", "與母同住", "育有", "與父母同住"]
    for kw in fam_kws:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    # Just mentions 生活狀況
    for kw in ["生活狀況", "經濟狀況", "之生活情況"]:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    return "absent", ""


def classify_intellect(reason: str) -> tuple[str, str]:
    # Education levels are typically neutral mentions
    for kw in INTELLECT_NEUTRAL_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    return "absent", ""


def classify_relation_victim(reason: str) -> tuple[str, str]:
    # Drug cases typically have no victim. Default absent.
    if keyword_present(reason, "與被害人之關係") or keyword_present(reason, "與被害人關係"):
        ev = find_first_substring_evidence(reason, "被害人")
        if ev:
            return "neutral", ev
    return "absent", ""


def classify_duty_breach(reason: str) -> tuple[str, str]:
    if keyword_present(reason, "違反義務"):
        ev = find_first_substring_evidence(reason, "違反義務")
        if ev:
            return "aggravating", ev
    return "absent", ""


def classify_provocation(reason: str) -> tuple[str, str]:
    for kw in PROVOCATION_KW:
        if keyword_present(reason, kw):
            ev = find_first_substring_evidence(reason, kw)
            if ev:
                return "neutral", ev
    return "absent", ""


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def classify_all(reason: str) -> dict:
    return {
        "motive": dict(zip(("direction", "evidence"), classify_motive(reason))),
        "provocation": dict(zip(("direction", "evidence"), classify_provocation(reason))),
        "means": dict(zip(("direction", "evidence"), classify_means(reason))),
        "life_status": dict(zip(("direction", "evidence"), classify_life_status(reason))),
        "character": dict(zip(("direction", "evidence"), classify_character(reason))),
        "intellect": dict(zip(("direction", "evidence"), classify_intellect(reason))),
        "relation_victim": dict(zip(("direction", "evidence"), classify_relation_victim(reason))),
        "duty_breach": dict(zip(("direction", "evidence"), classify_duty_breach(reason))),
        "harm": dict(zip(("direction", "evidence"), classify_harm(reason))),
        "post_attitude": dict(zip(("direction", "evidence"), classify_post_attitude(reason))),
    }


def get_sentencing_search_text(reason: str) -> str:
    """Return the sentencing-reasoning portion if locatable, else full reason.

    Searching only within the sentencing portion ensures evidence comes from
    the §57 reasoning, not from fact-finding / accusations / defenses
    elsewhere in the judgment.
    """
    # Strong-signal start markers (these are unambiguous §57 openers)
    strong_pats = [
        r"爰以行為人責任為基礎",
        r"爰以行為人之責任為基礎",
        r"爰審酌",
        r"爰斟酌",
        r"爰考量",
        r"茲審酌",
    ]
    # Weaker-signal markers (can match 證據能力 / 累犯加重 sections too —
    # only fall back to these if no strong signal exists)
    weak_pats = [
        r"本院審酌被告",
        r"審酌被告",
        r"並審酌",
    ]

    earliest = -1
    for p in strong_pats:
        m = re.search(p, reason)
        if m and (earliest < 0 or m.start() < earliest):
            earliest = m.start()
    if earliest < 0:
        # Fall back to weak markers — but require the immediate following
        # context to look like §57 reasoning (mentions 動機/手段/品行/
        # 戕害/坦承 within next 200 chars).
        for p in weak_pats:
            for m in re.finditer(p, reason):
                window = reason[m.end(): m.end() + 200]
                if re.search(r"動機|手段|戕害|坦承|犯後態度|前科|前案紀錄|教育程度|智識程度", window):
                    if earliest < 0 or m.start() < earliest:
                        earliest = m.start()
                    break
    if earliest < 0:
        return reason
    tail = reason[earliest:]
    # End markers
    end_pats = [
        r"一切情狀[，,]?\s*量處",
        r"綜合考量[^。]*量處",
        r"量處如主文",
        r"以資懲儆",
        r"以示懲儆",
        r"以資警惕",
        r"以資警懲",
        r"分別量處如主文",
        r"資以儆懲",
        r"資以懲儆",
        r"資以警懲",
    ]
    end_idx = -1
    for p in end_pats:
        m = re.search(p, tail)
        if m and (end_idx < 0 or m.end() < end_idx):
            end_idx = m.end()
    if end_idx < 0:
        return tail[:1500]
    return tail[: end_idx + 100]


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(INPUT_REASONS, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            # Use sentencing portion if findable; else fall back to full reason
            search_text = get_sentencing_search_text(row["reason"])
            factors = classify_all(search_text)
            # Verify evidences are still substrings of full reason (they should be)
            obj = {"jid": row["jid"], "factors": factors}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
    print(f"wrote {n} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
