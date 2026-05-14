"""Extract 刑法第57條十款量刑因子 via Claude API.

Reads judgment JSONL (`data/filtered/keelung_drug_all.jsonl`), sends each case's
量刑論述 to Claude Haiku 4.5 with a cached system prompt, parses strict JSON,
and appends one extraction per line to `data/processed/art57_factors.jsonl`.

The trainer (`scripts/04_train_baseline.py::_load_art57`) consumes this file
and one-hot-encodes each factor's direction as `a57_<factor>_mit/agg/neu`.

Usage:
    python scripts/07_llm_extract_factors.py --limit 5             # smoke
    python scripts/07_llm_extract_factors.py --jid KLDM,114,訴,5,...  # single
    python scripts/07_llm_extract_factors.py                       # full run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Anthropic public pricing for claude-haiku-4-5 (USD per million tokens),
# captured 2026-05-14. Update here if Anthropic changes their rates.
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00
PRICE_CACHE_WRITE_PER_MTOK = 1.25  # 5-minute ephemeral cache write surcharge
PRICE_CACHE_READ_PER_MTOK = 0.10

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


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


SYSTEM_PROMPT = """你是中華民國刑事量刑分析專家。你的任務是閱讀地方法院刑事判決書的「理由（科刑）」段落，並就**刑法第57條十款**量刑因子，逐款判斷法官在量刑審酌時的傾向。

刑法第57條原文（科刑審酌事項）：
「科刑時應以行為人之責任為基礎，並審酌一切情狀，尤應注意下列事項，為科刑輕重之標準：
 一、犯罪之動機、目的。
 二、犯罪時所受之刺激。
 三、犯罪之手段。
 四、犯罪行為人之生活狀況。
 五、犯罪行為人之品行。
 六、犯罪行為人之智識程度。
 七、犯罪行為人與被害人之關係。
 八、犯罪行為人違反義務之程度。
 九、犯罪所生之危險或損害。
 十、犯罪後之態度。」

十款判斷對應的 JSON key：
1. motive          犯罪之動機、目的（為己施用 vs. 為營利販賣 vs. 受迫不得已）
2. provocation     犯罪時所受之刺激（被挑釁、被脅迫、毒癮發作壓力）
3. means           犯罪之手段（單純交付 vs. 持械、誘騙、層轉販售）
4. life_status     犯人之生活狀況（家庭、經濟、是否單親扶養、失業）
5. character       犯人之品行（前科紀錄、素行）
6. intellect       犯人之智識程度（學歷、職業、認知能力）
7. relation_victim 犯人與被害人之關係（毒品案多半無被害人，常見 absent；若提及共犯間之利害關係或對家屬影響可填入）
8. duty_breach     犯人違反義務之程度（販賣對社會散布毒害、運輸數量、利得規模）
9. harm            犯罪所生之危險或損害（純質淨重、流通數量、查獲後是否已散布）
10. post_attitude  犯罪後之態度（自白、坦承、否認、悔悟、和解、賠償、配合調查）

每款的 direction 值（四選一）：
- "mitigating"   判決理由明確將該款列為**有利被告**之事由（例如：坦承犯行、無前科、家境清寒、為扶養家人犯罪、手段平和）
- "aggravating"  判決理由明確將該款列為**不利被告**之事由（例如：手段惡劣、否認犯行、前科累累、純質淨重高、層轉販售規模大）
- "neutral"      判決理由有提到該款（如「智識程度為高職畢業」「生活狀況勉持」），但未明示偏向有利或不利
- "absent"       判決理由完全沒有實質論及該款

evidence 欄位規則：
- 必須是判決理由中的**原文片段**，最長 80 字
- 不得改寫、不得臆測、不得補字
- direction = "absent" 時 evidence 為空字串
- 若同一段話同時涉及多款，可重複引用該段落

輸出格式（嚴格 JSON，無註解、無 markdown 圍欄、無多餘文字）：
{
  "motive":          {"direction": "...", "evidence": "..."},
  "provocation":     {"direction": "...", "evidence": "..."},
  "means":           {"direction": "...", "evidence": "..."},
  "life_status":     {"direction": "...", "evidence": "..."},
  "character":       {"direction": "...", "evidence": "..."},
  "intellect":       {"direction": "...", "evidence": "..."},
  "relation_victim": {"direction": "...", "evidence": "..."},
  "duty_breach":     {"direction": "...", "evidence": "..."},
  "harm":            {"direction": "...", "evidence": "..."},
  "post_attitude":   {"direction": "...", "evidence": "..."}
}

注意：
- 嚴格輸出十個 key，不可缺漏，不可新增
- direction 只能是四個列舉值之一
- 若判決理由極短或非典型（如協商判決），多數欄位填 "absent"
"""


USER_TEMPLATE = """JID: {jid}
案由: {title}

判決理由（科刑段落）：
{reason_text}"""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ExtractResult:
    jid: str
    factors: dict[str, dict[str, str]] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None


def _extract_reason_text(jfull: str, cap: int = 50_000) -> str:
    """Trim to the 量刑論述. Cut at 主文 onwards then up to date stamp.

    Find the *last* 中華民國...年...月...日 stamp (the signature/date footer);
    earlier 中華民國 mentions (e.g. citing a prior judgment date) won't trigger.
    """
    main_idx = jfull.find("主文")
    tail = jfull[main_idx:] if main_idx >= 0 else jfull
    stamps = list(re.finditer(r"中\s*華\s*民\s*國\s*\d{2,3}\s*年", tail))
    if stamps:
        tail = tail[: stamps[-1].start()]
    return tail[:cap]


def _parse_json_strict(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip optional ```json ... ``` fences if the model ignored instructions.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _validate_factors(obj: Any) -> dict[str, dict[str, str]]:
    if not isinstance(obj, dict):
        raise ValueError("response root is not a JSON object")
    valid_dirs = {"mitigating", "aggravating", "neutral", "absent"}
    out: dict[str, dict[str, str]] = {}
    for name, _ in ART57_FACTORS:
        v = obj.get(name)
        if not isinstance(v, dict):
            raise ValueError(f"missing factor: {name}")
        direction = v.get("direction")
        if direction not in valid_dirs:
            raise ValueError(f"factor {name}: bad direction {direction!r}")
        evidence = v.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = str(evidence)
        out[name] = {"direction": direction, "evidence": evidence[:120]}
    return out


def call_api(client: Any, jid: str, title: str, reason_text: str, model: str,
             max_retries: int = 3) -> ExtractResult:
    """Single Claude call with retry on transient failure. Never raises."""
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
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
                        jid=jid, title=title, reason_text=reason_text,
                    )},
                ],
            )
            latency = time.time() - t0
            text = "".join(getattr(b, "text", "") for b in resp.content)
            u = resp.usage
            usage = Usage(
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            )
            try:
                parsed = _parse_json_strict(text)
                factors = _validate_factors(parsed)
            except (ValueError, json.JSONDecodeError) as e:
                return ExtractResult(jid=jid, usage=usage, latency_s=latency,
                                      raw_response=text, error=f"parse: {e}")
            return ExtractResult(jid=jid, factors=factors, usage=usage,
                                  latency_s=latency, raw_response=text)
        except Exception as e:  # network / 429 / 5xx
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return ExtractResult(jid=jid, error=last_err or "unknown api error")


def _usd_cost(u: Usage) -> float:
    return (
        u.input_tokens * PRICE_INPUT_PER_MTOK / 1_000_000
        + u.output_tokens * PRICE_OUTPUT_PER_MTOK / 1_000_000
        + u.cache_write_tokens * PRICE_CACHE_WRITE_PER_MTOK / 1_000_000
        + u.cache_read_tokens * PRICE_CACHE_READ_PER_MTOK / 1_000_000
    )


def _load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["jid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_extraction(
    client: Any,
    rows: list[dict[str, Any]],
    out_path: Path,
    failures_path: Path,
    model: str,
    max_concurrency: int,
    limit: int = 0,
) -> dict[str, Any]:
    done = _load_done(out_path)
    pending = [r for r in rows if r["jid"] not in done]
    if limit:
        pending = pending[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    totals = Usage()
    latencies: list[float] = []
    n_ok = 0
    n_fail = 0

    def work(row: dict[str, Any]) -> ExtractResult:
        reason = _extract_reason_text(row["jfull"])
        return call_api(client, row["jid"], row.get("jtitle", ""), reason, model)

    with out_path.open("a", encoding="utf-8") as out_fh, \
         failures_path.open("a", encoding="utf-8") as fail_fh, \
         ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futures = {ex.submit(work, r): r for r in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with write_lock:
                totals.input_tokens += res.usage.input_tokens
                totals.output_tokens += res.usage.output_tokens
                totals.cache_read_tokens += res.usage.cache_read_tokens
                totals.cache_write_tokens += res.usage.cache_write_tokens
                if res.latency_s:
                    latencies.append(res.latency_s)
                if res.error or not res.factors:
                    n_fail += 1
                    fail_fh.write(json.dumps({
                        "jid": res.jid,
                        "error": res.error,
                        "raw_response": res.raw_response[:2000],
                    }, ensure_ascii=False) + "\n")
                    fail_fh.flush()
                else:
                    n_ok += 1
                    out_fh.write(json.dumps({
                        "jid": res.jid,
                        "factors": res.factors,
                        "input_tokens": res.usage.input_tokens,
                        "output_tokens": res.usage.output_tokens,
                        "cache_read_tokens": res.usage.cache_read_tokens,
                        "cache_write_tokens": res.usage.cache_write_tokens,
                    }, ensure_ascii=False) + "\n")
                    out_fh.flush()
                running_cost = _usd_cost(totals)
                cache_pct = (totals.cache_read_tokens
                             / max(1, totals.input_tokens + totals.cache_read_tokens) * 100)
                print(
                    f"[{i}/{len(pending)}] {res.jid}  "
                    f"{'OK' if not res.error else 'FAIL'}  "
                    f"in={res.usage.input_tokens} out={res.usage.output_tokens} "
                    f"cr={res.usage.cache_read_tokens}  "
                    f"lat={res.latency_s:.2f}s  "
                    f"total cost=${running_cost:.4f}  cache={cache_pct:.0f}%"
                )

    return {
        "ok": n_ok,
        "failed": n_fail,
        "totals": totals,
        "cost_usd": _usd_cost(totals),
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "n_pending": len(pending),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/keelung_drug_all.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/art57_factors.jsonl"))
    ap.add_argument("--failures", type=Path,
                    default=Path("data/processed/art57_failures.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--jid", help="process a single JID")
    ap.add_argument("--max-concurrency", type=int, default=4)
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("錯誤：請先設定 ANTHROPIC_API_KEY 環境變數。", file=sys.stderr)
        return 1
    try:
        import anthropic
    except ImportError:
        print("錯誤：請先 `pip install anthropic`。", file=sys.stderr)
        return 1

    if not args.inp.exists():
        print(f"錯誤：輸入檔案不存在：{args.inp}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    if args.jid:
        rows = [r for r in rows if r["jid"] == args.jid]
        if not rows:
            print(f"錯誤：找不到 jid={args.jid}", file=sys.stderr)
            return 1

    client = anthropic.Anthropic()
    summary = run_extraction(
        client=client,
        rows=rows,
        out_path=args.out,
        failures_path=args.failures,
        model=args.model,
        max_concurrency=args.max_concurrency,
        limit=args.limit,
    )

    t = summary["totals"]
    print("\n=== 完成 ===")
    print(f"成功：{summary['ok']}    失敗：{summary['failed']}")
    print(f"輸入 tokens：{t.input_tokens}  (cache_read={t.cache_read_tokens}, "
          f"cache_write={t.cache_write_tokens})")
    print(f"輸出 tokens：{t.output_tokens}")
    print(f"平均延遲：{summary['avg_latency_s']:.2f}s")
    print(f"總成本：${summary['cost_usd']:.4f} USD")
    if summary["ok"] > 0:
        per_case = summary["cost_usd"] / summary["ok"]
        print(f"每件平均：${per_case:.5f}    "
              f"推估 5,809 件全量：${per_case * 5809:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
