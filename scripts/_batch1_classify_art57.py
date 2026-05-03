"""Classify §57 sentencing factors for cases 0-319 of keelung_drug_all.jsonl.

Output: /tmp/art57_batch_1.jsonl (one JSON object per line, 320 rows).

Strategy
--------
Taiwan drug judgments use a highly stereotyped "爰以行為人之責任為基礎,審酌..."
量刑審酌 paragraph that enumerates the §57 factors by name. We:

  1. Extract the reason section (compound 事實及理由 / 理由 / 論罪科刑 fallback).
  2. Locate the 量刑審酌 paragraph by scoring candidate "爰..." blocks for §57
     keyword density and picking the highest-scoring block.
  3. Apply pattern-based classification per factor — the boilerplate phrasings
     are stable across hundreds of cases, so rules give high precision.
  4. Pick a 20-80 char substring of the reason text as evidence (must be exact
     substring; we use anchored span around the matched phrase).

Each row contains all 10 factors with {direction, evidence}. evidence is
empty for absent factors. For the reason-missing case (idx=202), all factors
are emitted as `absent`/empty.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import _extract_section  # noqa: E402

INPUT = Path(r"C:\Users\alex2\Desktop\vsCode\AiJudge\data\filtered\keelung_drug_all.jsonl")
OUTPUT = Path("/tmp/art57_batch_1.jsonl")
START, END = 0, 320

FACTOR_KEYS = [
    "motive", "provocation", "means", "life_status", "character",
    "intellect", "relation_victim", "duty_breach", "harm", "post_attitude",
]


# -- Section extraction -----------------------------------------------------

def extract_reason(jfull: str) -> str:
    """Extract the reasoning section, with cascading fallbacks.

    For 訴/重訴 cases the §57 paragraph can sit far below the start marker
    and the default end patterns (附..表 / 書記官) cut it off. We use
    less-aggressive end patterns for the 理由 marker — only stop at the
    final 中華民國 ddd date or 書記官 line."""
    text = _extract_section(
        jfull, "事實及理由",
        (r"中\s*華\s*民\s*國\s*\d{3}年", r"書\s*記\s*官"),
    )
    if len(text) >= 50:
        return text
    text = _extract_section(
        jfull, "理由",
        (r"書\s*記\s*官",),
    )
    if len(text) >= 50:
        return text
    text = _extract_section(
        jfull, "論罪科刑",
        (r"書\s*記\s*官",),
    )
    if len(text) >= 50:
        return text
    return ""


def find_sentencing_block(reason: str) -> str:
    """Return the §57 量刑審酌 paragraph (highest-scoring 爰... block)."""
    candidates: list[tuple[int, str]] = []
    for m in re.finditer(r"爰[^。]*?(?:審酌|衡|斟酌|為基礎|考量)", reason):
        start = m.start()
        rest = reason[start:start + 3500]
        end = -1
        liang = re.search(r"量處", rest)
        if liang:
            after = rest.find("。", liang.end())
            if after > 0:
                end = after + 1
        if end < 0:
            # fallback: take up to 1500 chars or two periods
            end = min(len(rest), 1500)
        block = rest[:end]
        score = 0
        for kw in (
            "智識", "生活狀況", "動機", "目的", "手段", "態度",
            "品行", "素行", "一切情狀", "危害", "戕害", "坦承", "否認",
            "教育程度", "家境", "勉持", "家庭",
        ):
            if kw in block:
                score += 1
        candidates.append((score, block))
    if not candidates:
        # fallback: search for any 審酌/衡酌 paragraph
        for m in re.finditer(r"審酌|衡酌", reason):
            start = max(0, m.start() - 5)
            rest = reason[start:start + 1500]
            score = sum(1 for kw in ("智識", "生活狀況", "動機", "手段",
                                     "態度", "品行", "戕害") if kw in rest)
            if score >= 2:
                candidates.append((score, rest))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# -- Evidence-substring helper ---------------------------------------------

def normalize_for_match(s: str) -> str:
    """Remove common whitespace artifacts (newlines, spaces) for substring search."""
    return re.sub(r"[\s　]+", "", s)


def find_substring(reason: str, anchor: str, pre: int = 8, post: int = 32,
                   target_len: int = 40) -> str:
    """Find 'anchor' substring in reason and return a 20-80 char window around it.

    Tolerates the fact that judgment text contains line wraps; we search the
    whitespace-stripped reason but return original substring with surroundings.
    """
    if not anchor:
        return ""
    # Try direct substring first
    idx = reason.find(anchor)
    if idx >= 0:
        start = max(0, idx - pre)
        end = min(len(reason), idx + len(anchor) + post)
        sub = reason[start:end]
        # Trim to target_len if too long
        if len(sub) > 80:
            sub = sub[:80]
        return sub
    # Whitespace-tolerant search
    norm = normalize_for_match(reason)
    norm_anchor = normalize_for_match(anchor)
    n_idx = norm.find(norm_anchor)
    if n_idx < 0:
        return ""
    # Map norm index back to original index by counting
    orig_count = 0
    norm_count = 0
    start_orig = -1
    end_orig = -1
    for i, ch in enumerate(reason):
        if norm_count == n_idx and start_orig < 0:
            start_orig = i
        if norm_count == n_idx + len(norm_anchor):
            end_orig = i
            break
        if not re.match(r"[\s　]", ch):
            norm_count += 1
        orig_count += 1
    if end_orig < 0:
        end_orig = len(reason)
    if start_orig < 0:
        return ""
    s2 = max(0, start_orig - pre)
    e2 = min(len(reason), end_orig + post)
    sub = reason[s2:e2]
    if len(sub) > 80:
        sub = sub[:80]
    return sub


def expand_window(reason: str, idx: int, length: int,
                  min_len: int = 20, max_len: int = 80) -> str:
    """Expand a [idx, idx+length] span symmetrically to be in [min_len, max_len]."""
    start = idx
    end = idx + length
    while end - start < min_len and (start > 0 or end < len(reason)):
        if start > 0:
            start -= 1
        if end - start < min_len and end < len(reason):
            end += 1
    if end - start > max_len:
        end = start + max_len
    return reason[start:end]


def find_phrase_window(reason: str, phrase: str,
                       min_len: int = 20, max_len: int = 80) -> str:
    """Locate phrase in reason (whitespace-tolerant) and return a window of length [min_len, max_len]."""
    if not phrase:
        return ""
    idx = reason.find(phrase)
    if idx >= 0:
        return expand_window(reason, idx, len(phrase), min_len, max_len)
    # whitespace-tolerant fallback: build regex with optional whitespace between chars
    pat = r"[\s　]*".join(re.escape(c) for c in phrase)
    m = re.search(pat, reason)
    if m:
        return expand_window(reason, m.start(), m.end() - m.start(), min_len, max_len)
    return ""


def phrase_in_text(text: str, phrase: str) -> bool:
    """Whitespace-tolerant substring check (handles 智\\r\\n    識 -> 智識)."""
    if not phrase:
        return False
    if phrase in text:
        return True
    pat = r"[\s　]*".join(re.escape(c) for c in phrase)
    return re.search(pat, text) is not None


# -- §57 classifier ---------------------------------------------------------

def classify(jfull: str, reason: str, block: str) -> dict[str, dict]:
    """Apply rule-based classification per factor.

    For each factor we look for stereotyped phrasings in `block` (the 量刑
    paragraph). If not found there, fall back to `reason`. Direction follows
    the judge's framing (mitigating = favorable to defendant).
    """
    R = block if block else reason  # primary search space
    ALL = reason  # fallback

    factors: dict[str, dict] = {k: {"direction": "absent", "evidence": ""}
                                 for k in FACTOR_KEYS}

    def search_phrases(phrases: list[str], directions: list[str],
                       windows: int = 80) -> tuple[str, str]:
        """For each phrase (in order), check R then ALL. Return (direction, evidence) on first hit."""
        for ph, dr in zip(phrases, directions):
            if phrase_in_text(R, ph):
                return dr, find_phrase_window(R, ph, max_len=windows)
            if phrase_in_text(ALL, ph):
                return dr, find_phrase_window(ALL, ph, max_len=windows)
        return "absent", ""

    # ---------- post_attitude (most informative; check first) ----------
    pa_agg_phrases = [
        "否認犯行", "矢口否認", "飾詞狡辯", "犯後態度不佳", "態度不佳",
        "毫無悔意", "無悔意", "態度惡劣", "百般狡辯", "拒不認罪",
        "態度欠佳", "否認運輸", "否認販賣", "否認施用", "否認持有",
        "否認轉讓",
    ]
    pa_mit_phrases = [
        "坦承犯行", "坦承施用", "坦承全部犯行", "犯後態度良好", "態度良好",
        "犯後坦承", "坦認不諱", "坦承不諱", "見悔意", "尚見悔意",
        "犯後尚見悔意", "態度尚可", "犯後態度尚可", "犯後自首",
        "自首犯行", "供出毒品來源", "悔悟", "犯後坦白", "坦承販賣",
        "坦承運輸", "坦承轉讓", "坦承持有",
    ]
    pa_dir = "absent"
    pa_evi = ""
    for ph in pa_agg_phrases:
        if phrase_in_text(R, ph):
            pa_dir = "aggravating"
            pa_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            pa_dir = "aggravating"
            pa_evi = find_phrase_window(ALL, ph)
            break
    if pa_dir == "absent":
        for ph in pa_mit_phrases:
            if phrase_in_text(R, ph):
                pa_dir = "mitigating"
                pa_evi = find_phrase_window(R, ph)
                break
            if phrase_in_text(ALL, ph):
                pa_dir = "mitigating"
                pa_evi = find_phrase_window(ALL, ph)
                break
    factors["post_attitude"] = {"direction": pa_dir, "evidence": pa_evi}

    # ---------- intellect (智識程度 / 教育程度) ----------
    intel_phrases_neutral = [
        "智識程度", "教育程度", "學歷", "國中畢業", "高中畢業",
        "高中肄業", "國小畢業", "大學畢業",
    ]
    intel_dir = "absent"
    intel_evi = ""
    for ph in intel_phrases_neutral:
        if phrase_in_text(R, ph):
            intel_dir = "neutral"
            intel_evi = find_phrase_window(R, ph, max_len=70)
            break
        if phrase_in_text(ALL, ph):
            intel_dir = "neutral"
            intel_evi = find_phrase_window(ALL, ph, max_len=70)
            break
    factors["intellect"] = {"direction": intel_dir, "evidence": intel_evi}

    # ---------- life_status (生活狀況 / 家境 / 經濟 / 婚姻) ----------
    ls_phrases = [
        "生活狀況", "家境勉持", "家境貧寒", "家境小康", "家境普通",
        "經濟狀況", "家庭狀況", "經濟勉持", "家庭經濟", "家境清寒",
        "勉持", "勉強維持", "家境不好", "家境尚可",
    ]
    ls_dir = "absent"
    ls_evi = ""
    for ph in ls_phrases:
        if phrase_in_text(R, ph):
            ls_dir = "neutral"
            ls_evi = find_phrase_window(R, ph, max_len=70)
            break
        if phrase_in_text(ALL, ph):
            ls_dir = "neutral"
            ls_evi = find_phrase_window(ALL, ph, max_len=70)
            break
    factors["life_status"] = {"direction": ls_dir, "evidence": ls_evi}

    # ---------- harm (危害損害) ----------
    harm_mit_phrases = [
        "僅屬戕害自身", "戕害自身之行為", "戕害自己身心健康",
        "戕害自我身心健康", "戕害其個人身心健康", "未侵犯其他法益",
        "未直接危害他人", "未造成他人具體危害", "尚未造成他人具體危害",
        "反社會性之程度較低", "戕害自我", "以自戕身心健康為主",
        "戕害自己", "以自戕為主", "戕害自身", "戕害己身",
        "對己身健康戕害",
    ]
    harm_agg_phrases = [
        "危害甚鉅", "危害社會", "危害國民身心健康", "危害國民之身心健康",
        "毒品造成諸多社會問題", "對社會治安造成", "對社會治安仍造成潛在危險",
        "危害匪淺", "為害匪淺", "戕害國民身心", "毒品擴散之危害",
        "對社會造成之負擔",
    ]
    harm_dir = "absent"
    harm_evi = ""
    for ph in harm_mit_phrases:
        if phrase_in_text(R, ph):
            harm_dir = "mitigating"
            harm_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            harm_dir = "mitigating"
            harm_evi = find_phrase_window(ALL, ph)
            break
    if harm_dir == "absent":
        for ph in harm_agg_phrases:
            if phrase_in_text(R, ph):
                harm_dir = "aggravating"
                harm_evi = find_phrase_window(R, ph)
                break
            if phrase_in_text(ALL, ph):
                harm_dir = "aggravating"
                harm_evi = find_phrase_window(ALL, ph)
                break
    factors["harm"] = {"direction": harm_dir, "evidence": harm_evi}

    # ---------- character (品行 / 素行 / 前科) ----------
    char_agg_phrases = [
        "猶不知戒慎", "再次漠視法令", "再三施用毒品", "未能戒除",
        "仍不思悔改", "猶未能", "對於刑罰反應力薄弱", "刑罰反應力薄弱",
        "缺乏戒斷決心", "顯然缺乏戒斷決心", "猶不知悔改", "猶不知警惕",
        "猶不思悔改", "前已因施用毒品", "再犯本罪", "再三施用",
        "再為本案犯行", "其根絕毒害之意志不堅",
    ]
    char_neutral_phrases = [
        "前科、素行", "素行資料", "其素行", "品行", "前案紀錄表",
    ]
    char_dir = "absent"
    char_evi = ""
    for ph in char_agg_phrases:
        if phrase_in_text(R, ph):
            char_dir = "aggravating"
            char_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            char_dir = "aggravating"
            char_evi = find_phrase_window(ALL, ph)
            break
    if char_dir == "absent":
        for ph in char_neutral_phrases:
            if phrase_in_text(R, ph):
                char_dir = "neutral"
                char_evi = find_phrase_window(R, ph)
                break
            if phrase_in_text(ALL, ph):
                char_dir = "neutral"
                char_evi = find_phrase_window(ALL, ph)
                break
    factors["character"] = {"direction": char_dir, "evidence": char_evi}

    # ---------- motive (動機、目的) ----------
    motive_agg_phrases = [
        "意圖牟利", "牟取暴利", "為牟取私利", "貪圖",
        "為一己私利",
    ]
    motive_neutral_phrases = [
        "犯罪之動機", "犯罪動機", "動機、目的", "動機目的",
        "犯罪動機", "之動機、目的", "其動機",
    ]
    m_dir = "absent"
    m_evi = ""
    for ph in motive_agg_phrases:
        if phrase_in_text(R, ph):
            m_dir = "aggravating"
            m_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            m_dir = "aggravating"
            m_evi = find_phrase_window(ALL, ph)
            break
    if m_dir == "absent":
        for ph in motive_neutral_phrases:
            if phrase_in_text(R, ph):
                m_dir = "neutral"
                m_evi = find_phrase_window(R, ph)
                break
            if phrase_in_text(ALL, ph):
                m_dir = "neutral"
                m_evi = find_phrase_window(ALL, ph)
                break
    factors["motive"] = {"direction": m_dir, "evidence": m_evi}

    # ---------- means (手段) ----------
    means_mit_phrases = [
        "犯罪手段尚屬平和", "手段尚屬平和", "手段平和", "情節非鉅",
        "手段尚非殘酷", "犯罪手段平和", "手段非鉅",
    ]
    means_agg_phrases = [
        "手段兇殘", "手段惡劣", "暴力相向", "兇狠", "手段卑劣",
    ]
    means_neutral_phrases = [
        "犯罪之手段", "目的、手段", "手段、",
    ]
    means_dir = "absent"
    means_evi = ""
    for ph in means_mit_phrases:
        if phrase_in_text(R, ph):
            means_dir = "mitigating"
            means_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            means_dir = "mitigating"
            means_evi = find_phrase_window(ALL, ph)
            break
    if means_dir == "absent":
        for ph in means_agg_phrases:
            if phrase_in_text(R, ph):
                means_dir = "aggravating"
                means_evi = find_phrase_window(R, ph)
                break
            if phrase_in_text(ALL, ph):
                means_dir = "aggravating"
                means_evi = find_phrase_window(ALL, ph)
                break
    if means_dir == "absent":
        for ph in means_neutral_phrases:
            if phrase_in_text(R, ph):
                means_dir = "neutral"
                means_evi = find_phrase_window(R, ph)
                break
            if phrase_in_text(ALL, ph):
                means_dir = "neutral"
                means_evi = find_phrase_window(ALL, ph)
                break
    factors["means"] = {"direction": means_dir, "evidence": means_evi}

    # ---------- provocation (受刺激) ----------
    prov_phrases = ["所受刺激", "犯罪時所受之刺激", "受刺激"]
    prov_dir = "absent"
    prov_evi = ""
    for ph in prov_phrases:
        if phrase_in_text(R, ph):
            prov_dir = "neutral"
            prov_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            prov_dir = "neutral"
            prov_evi = find_phrase_window(ALL, ph)
            break
    factors["provocation"] = {"direction": prov_dir, "evidence": prov_evi}

    # ---------- relation_victim (與被害人關係) ----------
    rv_phrases = ["與被害人", "被害人之關係", "與買家", "與買受人"]
    rv_dir = "absent"
    rv_evi = ""
    for ph in rv_phrases:
        if phrase_in_text(R, ph):
            rv_dir = "neutral"
            rv_evi = find_phrase_window(R, ph)
            break
    factors["relation_victim"] = {"direction": rv_dir, "evidence": rv_evi}

    # ---------- duty_breach (違反義務之程度) ----------
    db_phrases = [
        "違反義務", "違反義務之程度", "義務之違反",
    ]
    db_dir = "absent"
    db_evi = ""
    for ph in db_phrases:
        if phrase_in_text(R, ph):
            db_dir = "neutral"
            db_evi = find_phrase_window(R, ph)
            break
        if phrase_in_text(ALL, ph):
            db_dir = "neutral"
            db_evi = find_phrase_window(ALL, ph)
            break
    factors["duty_breach"] = {"direction": db_dir, "evidence": db_evi}

    # Constraint: every evidence must be a substring of the original full text
    # If we picked from `block` but block is a substring of reason, evidence is
    # already a substring of reason (and hence of jfull). Verify length 20-80.
    for k, v in factors.items():
        evi = v["evidence"]
        if not evi:
            continue
        # Verify substring of jfull (reason is a substring of jfull)
        if evi not in jfull:
            # try to find equivalent in jfull (whitespace-tolerant)
            if evi in reason:
                # reason is substring of jfull, so evi must be in jfull too
                pass
            else:
                # something went wrong - clear evidence
                v["evidence"] = ""
                continue
        # Trim/pad to 20-80 char range
        if len(evi) < 20:
            # locate in reason and re-expand
            idx = reason.find(evi)
            if idx >= 0:
                v["evidence"] = expand_window(reason, idx, len(evi), 20, 80)
        elif len(evi) > 80:
            v["evidence"] = evi[:80]
    return factors


# -- Main loop --------------------------------------------------------------

def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    empty_reason_jids: list[str] = []
    pa_dist: dict[str, int] = {"mitigating": 0, "aggravating": 0,
                                "neutral": 0, "absent": 0}
    samples: list[dict] = []

    with INPUT.open(encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i < START:
                continue
            if i >= END:
                break
            row = json.loads(line)
            jid = row.get("jid", "")
            jfull = row.get("jfull", "")
            jcase = row.get("jcase", "")

            reason = extract_reason(jfull)
            if not reason or len(reason) < 50:
                empty_reason_jids.append(jid)
                # Emit absent factors
                factors = {k: {"direction": "absent", "evidence": ""}
                            for k in FACTOR_KEYS}
            else:
                block = find_sentencing_block(reason)
                factors = classify(jfull, reason, block)

            obj = {"jid": jid, "factors": factors}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            pa_dist[factors["post_attitude"]["direction"]] = (
                pa_dist[factors["post_attitude"]["direction"]] + 1
            )
            written += 1
            if i in (0, 8, 9, 102, 200) and len(samples) < 5:
                samples.append({"idx": i, "jcase": jcase, "obj": obj})

    print(f"wrote {written} rows to {OUTPUT}")
    print(f"empty-reason cases: {len(empty_reason_jids)} -> {empty_reason_jids}")
    print(f"post_attitude distribution: {pa_dist}")
    print("\n--- sample rows ---")
    for s in samples:
        print(f"idx={s['idx']} jcase={s['jcase']}")
        print(json.dumps(s["obj"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
