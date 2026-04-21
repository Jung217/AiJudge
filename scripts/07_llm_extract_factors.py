"""Extract 刑法第57條十款量刑因子 via Claude API — **deferred scaffold**.

The 10 factors are inherently **subjective quality judgments** (犯後態度、品行、
智識程度、生活狀況 …) that regex cannot reliably catch. This script defers that
extraction to Claude. Run only after:
  1. `pip install anthropic` + set `ANTHROPIC_API_KEY` env var
  2. Human-labeled 100-case validation set exists (see scripts/05 + 06)
  3. Prompt calibrated against validation set (iterate until F1 > 0.8 per field)

Design:
  - **Prompt caching**: the lengthy system prompt (§57 definitions + extraction
    rules) is the same across all judgments, so caching collapses ~2000 tokens
    of overhead per call into 0.1× cost after the first hit.
  - **Structured output**: JSON schema with one boolean + optional evidence
    string per factor. Temperature 0 for reproducibility.
  - **Model**: claude-haiku-4-5 for cost (approx 100 cases ≈ USD 0.50 with cache);
    bump to claude-sonnet-4-6 if factor-level F1 < 0.7 on validation.

Usage (when enabled):
    python scripts/07_llm_extract_factors.py                # 全部 cases
    python scripts/07_llm_extract_factors.py --limit 20     # smoke test
    python scripts/07_llm_extract_factors.py --jid KLDM,115,訴,5,20260205,2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- 刑法第57條十款 ----------------------------------------------------------

ART57_FACTORS = [
    ("motive",          "犯罪之動機、目的"),
    ("provocation",     "犯罪時所受之刺激"),
    ("means",           "犯罪之手段"),
    ("life_status",     "犯人之生活狀況"),
    ("character",       "犯人之品行"),
    ("intellect",       "犯人之智識程度"),
    ("relation_victim", "犯人與被害人之關係"),
    ("duty_breach",     "犯人違反義務之程度"),
    ("harm",            "犯罪所生之危險或損害"),
    ("post_attitude",   "犯罪後之態度"),
]


@dataclass
class Art57Extraction:
    jid: str
    # One {judgment direction, evidence} per factor.
    # direction: "mitigating" (有利被告) / "aggravating" (不利) / "neutral" / "absent"
    factors: dict[str, dict] = field(default_factory=dict)
    raw_response: str = ""
    model: str = ""
    cache_hit: Optional[bool] = None


# -- Prompt (cached, ~2000 tokens) ------------------------------------------

SYSTEM_PROMPT = """你是中華民國刑事量刑分析專家。從判決書「理由」段落中，識別法官在量刑時具體提及的**刑法第57條十款**因子。

十款因子：
1. motive          犯罪之動機、目的
2. provocation     犯罪時所受之刺激
3. means           犯罪之手段
4. life_status     犯人之生活狀況
5. character       犯人之品行
6. intellect       犯人之智識程度
7. relation_victim 犯人與被害人之關係（對無被害人之毒品案多不適用）
8. duty_breach     犯人違反義務之程度
9. harm            犯罪所生之危險或損害
10. post_attitude  犯罪後之態度（自白、坦承、悔悟、和解、賠償皆屬之）

每款的 direction 判斷：
- "mitigating"：法官認為有利被告（如坦承犯行、態度良好）
- "aggravating"：法官認為不利被告（如手段兇殘、態度輕慢）
- "neutral"：法官提及但未明顯偏向
- "absent"：判決理由未實質論及該款

evidence 欄位：貼出判決理由中支持 direction 判斷的**原文片段**（30-80 字），不可臆測、不可改寫。absent 時為空字串。

輸出嚴格 JSON（無註解、無多餘空白），格式：
{
  "motive": {"direction": "...", "evidence": "..."},
  ...
}"""

USER_TEMPLATE = """判決 JID: {jid}
判決案由: {title}

理由段落：
{reason_text}"""


# -- Extraction client (not executed until user enables) --------------------

def extract_one(client, jid: str, title: str, reason_text: str,
                model: str = "claude-haiku-4-5") -> Art57Extraction:
    """One API call per judgment. Uses prompt-cache on system prompt."""
    import anthropic  # noqa: F401  (type hint only; raises if uninstalled)

    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": USER_TEMPLATE.format(
                jid=jid, title=title, reason_text=reason_text[:6000]
            )}
        ],
    )

    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    try:
        factors = json.loads(text)
    except json.JSONDecodeError:
        # Try to locate first {...} block
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        factors = json.loads(m.group(0)) if m else {}

    cache_hit = bool(getattr(resp.usage, "cache_read_input_tokens", 0))
    return Art57Extraction(
        jid=jid, factors=factors, raw_response=text,
        model=model, cache_hit=cache_hit,
    )


def _load_reason_text(jfull: str) -> str:
    """Minimal re-implementation — mirrors features._extract_section logic."""
    import re
    for marker in ("事實及理由", "理由"):
        pat = r"\s*".join(re.escape(c) for c in marker)
        m = re.search(pat, jfull)
        if not m:
            continue
        tail = jfull[m.end():]
        end_m = re.search(r"中\s*華\s*民\s*國\s*\d{3}", tail)
        return tail[:end_m.start()] if end_m else tail[:6000]
    return jfull[:6000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/keelung_drug.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/art57_factors.jsonl"))
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--jid", help="process a single JID")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: set ANTHROPIC_API_KEY env var first", file=sys.stderr)
        return 1
    try:
        import anthropic
    except ImportError:
        print("error: pip install anthropic", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip JIDs already in out
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["jid"])
            except (json.JSONDecodeError, KeyError):
                continue

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    if args.jid:
        rows = [r for r in rows if r["jid"] == args.jid]

    processed = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for r in rows:
            if r["jid"] in done:
                continue
            if args.limit and processed >= args.limit:
                break
            reason = _load_reason_text(r["jfull"])
            ext = extract_one(client, r["jid"], r["jtitle"], reason, model=args.model)
            fh.write(json.dumps(asdict(ext), ensure_ascii=False) + "\n")
            fh.flush()
            processed += 1
            marker = "c" if ext.cache_hit else "·"
            print(f"[{processed}] {marker} {r['jid']}  "
                  f"factors={sum(1 for v in ext.factors.values() if v.get('direction') and v.get('direction') != 'absent')}")

    print(f"done: {processed} new extractions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
