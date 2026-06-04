"""Build / update the error registry (plan §6.3, §7.3) from walk-forward outliers.

`09_inspect_outliers.py` dumps the worst-residual cases to
`data/processed/outliers.csv`. This script turns that raw dump into a
*persistent, append-only* error registry for manual review:

  - each outlier is pre-classified into one of the known residual buckets
    (plan §13.3 末 / web §168) via heuristics, so the reviewer starts from a
    suspected cause instead of a blank cell;
  - human columns (`verdict_category`, `action`, `reviewer`, `reviewed_date`,
    `notes`) are preserved across re-runs — keyed by JID — so refreshing the
    outliers after a model change never clobbers prior review work;
  - cases that were reviewed once but later dropped out of the top-K (model
    improved) are kept, marked `dropped_from_topk`, so the registry is an
    audit trail, not a moving snapshot.

The registry doubles as the governance "偏誤登記" plan §7.3 calls for.

Run:
    python scripts/09_inspect_outliers.py --top 200          # refresh outliers
    python scripts/10_error_registry.py                      # build / update
    python scripts/10_error_registry.py --top 50             # only worst 50
"""
from __future__ import annotations

import argparse
import urllib.parse
from pathlib import Path

import pandas as pd

# ── Error taxonomy ──────────────────────────────────────────────────────────
# Aligned with plan §13.3 末 known-hard cases and web index.md §168. The
# suspected_category is a *heuristic guess*; the reviewer confirms or overrides
# it in verdict_category.
TAXONOMY = {
    "aggregate_51":      "數罪併罰 / 應執行刑 — 單罪刑度區間無法約束合併刑(§51),"
                         "此列 y_true 是合併後總刑,本就不該被單罪模型預測。",
    "missing_reduction": "重罪實判遠低於預測 — 疑漏抓 §59/§17 減刑,或多被告判決"
                         "中他人之減刑討論被全文偵測誤算到本案。",
    "missing_feature":   "重罪實判遠高於預測 — 決定刑度的加重因子(純質淨重、共犯"
                         "角色、下游規模、犯次)尚未結構化抽取。",
    "rule_table_gap":    "持有純質淨重加重型(§11Ⅴ/Ⅵ)未完整建表,法定刑度區間偏低。",
    "label_or_variance": "抽取無明顯錯誤 — 疑為標籤雜訊(OCR / 附件刑度)或法官個案"
                         "裁量之天然變異,模型未必有錯。",
}

FLAG_COLS = ["art17_1", "art17_2", "art59", "self_surrender",
             "is_attempt", "recidivism", "summary", "is_aggregate_sentence"]

# Columns the human fills in. Preserved verbatim across re-runs (keyed by JID).
HUMAN_COLS = ["verdict_category", "action", "reviewer", "reviewed_date", "notes"]

# Behaviors heavy enough that a large residual signals a structural extraction
# gap rather than ordinary sentencing variance.
_HEAVY = {"販賣", "運輸", "製造", "意圖販賣而持有", "轉讓"}

# |y_pred − y_true| beyond this (months) on a heavy crime is read as a
# systematic miss (missed reduction / missed aggravator), not label noise.
_HEAVY_RESIDUAL_MONTHS = 6

# Reviewer-friendly column order for the registry CSV (remaining columns, e.g.
# the raw flag/feature columns carried from outliers.csv, follow in place).
COLUMN_ORDER = [
    "jid", "jdate", "court", "convicted_behaviors", "convicted_levels",
    "weight_g", "y_true", "y_pred_p50", "y_pred_p25", "y_pred_p75",
    "signed_residual", "abs_residual", "in_band", "flags",
    "suspected_category", "verdict_category", "action", "reviewer",
    "reviewed_date", "notes", "suspected_reason", "dropped_from_topk", "jurl",
]


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _primary_behavior(s) -> str:
    if not isinstance(s, str):       # NaN / float when behaviors column is empty
        return ""
    return s.split(",")[0].strip()


def suspect(row: pd.Series) -> str:
    """Heuristic pre-classification → one of TAXONOMY keys."""
    behavior = _primary_behavior(row.get("convicted_behaviors", ""))
    signed_residual = float(row.get("signed_residual", 0.0))     # y_pred − y_true
    n_counts = _as_int(row.get("n_sentence_counts"), default=1)
    is_aggregate = _as_int(row.get("is_aggregate_sentence"))
    weight_g = float(row.get("weight_g", -1) or -1)

    if is_aggregate or n_counts > 1:
        return "aggregate_51"
    if behavior in _HEAVY and signed_residual > _HEAVY_RESIDUAL_MONTHS:
        return "missing_reduction"           # model far above actual
    if behavior in _HEAVY and signed_residual < -_HEAVY_RESIDUAL_MONTHS:
        return "missing_feature"             # model far below actual
    if behavior == "持有" and weight_g > 0:
        return "rule_table_gap"
    return "label_or_variance"


def flags_str(row: pd.Series) -> str:
    return ",".join(c for c in FLAG_COLS if _as_int(row.get(c)) == 1)


def judgment_url(jid: str) -> str:
    """Best-effort 司法院裁判書系統 URL. The JID's case-type char may be mangled
    (full-width corruption, see project_parsing_gotchas) — the reviewer can fix
    it in the system's search box if the deep link 404s."""
    if not jid:
        return ""
    return ("https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id="
            + urllib.parse.quote(str(jid)))


def build(outliers: pd.DataFrame, top: int | None) -> pd.DataFrame:
    """Annotate the raw outliers with heuristic classification + blank human
    columns, ready to seed (or merge into) the registry."""
    df = outliers.copy()
    if top is not None:
        df = df.head(top)
    df["suspected_category"] = df.apply(suspect, axis=1)
    df["suspected_reason"] = df["suspected_category"].map(TAXONOMY)
    df["flags"] = df.apply(flags_str, axis=1)
    df["jurl"] = df["jid"].map(judgment_url)
    for col in HUMAN_COLS:
        df[col] = ""
    return df


def merge_with_prior(fresh: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Carry human review columns from a prior registry onto the refreshed rows
    (keyed by JID), and keep previously-reviewed cases that fell out of the
    current top-K as an audit trail."""
    prior = prior.fillna("")
    prior_human = prior.set_index("jid")
    merged = fresh.set_index("jid")
    for col in HUMAN_COLS:
        if col in prior_human.columns:
            merged[col] = prior_human[col].reindex(merged.index).fillna("")
    merged = merged.reset_index()
    merged["dropped_from_topk"] = 0

    fell_out = prior[~prior["jid"].isin(set(merged["jid"]))].copy()
    if len(fell_out):
        fell_out["dropped_from_topk"] = 1
        return pd.concat([merged, fell_out], ignore_index=True)
    return merged


def order_columns(registry: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in COLUMN_ORDER if c in registry.columns] + \
              [c for c in registry.columns if c not in COLUMN_ORDER]
    return registry[ordered].fillna("")


def _short_label(category: str) -> str:
    return TAXONOMY.get(category, "").split("—")[0].strip().split(",")[0]


def print_summary(registry: pd.DataFrame, out_path: Path) -> None:
    n = len(registry)
    reviewed = int((registry["verdict_category"].astype(str).str.len() > 0).sum())
    print(f"寫出錯誤登記 → {out_path}")
    print(f"  總計 {n} 件,已覆核 {reviewed} 件,待覆核 {n - reviewed} 件")
    if "dropped_from_topk" in registry.columns:
        dropped = int((registry["dropped_from_topk"] == 1).sum())
        if dropped:
            print(f"  其中 {dropped} 件已非當前 top-K(模型改善後掉出,保留審計軌跡)")

    print("\n── 啟發式預分類分布 ──")
    for cat, cnt in registry["suspected_category"].value_counts().items():
        print(f"  {cat:<18} {cnt:>3}   {_short_label(cat)}")

    if reviewed:
        print("\n── 人工覆核確認分布(verdict_category) ──")
        confirmed = registry["verdict_category"].astype(str).str.len() > 0
        vc = registry.loc[confirmed, "verdict_category"].value_counts()
        for cat, cnt in vc.items():
            print(f"  {cat:<18} {cnt:>3}")

    print("\n下一步:用 Excel/編輯器打開,逐列填 verdict_category("
          + " / ".join(TAXONOMY.keys()) + ")、action、reviewer、reviewed_date。")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outliers", type=Path,
                    default=Path("data/processed/outliers.csv"))
    ap.add_argument("--registry", type=Path,
                    default=Path("data/processed/error_registry.csv"))
    ap.add_argument("--top", type=int, default=None,
                    help="only register the worst N outliers (default: all)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.outliers.exists():
        print(f"找不到 {args.outliers} — 請先跑:\n"
              f"  python scripts/09_inspect_outliers.py --top 200")
        return 1

    outliers = pd.read_csv(args.outliers, encoding="utf-8-sig")
    fresh = build(outliers, args.top)

    if args.registry.exists():
        prior = pd.read_csv(args.registry, encoding="utf-8-sig")
        registry = merge_with_prior(fresh, prior)
    else:
        registry = fresh.copy()
        registry["dropped_from_topk"] = 0

    registry = order_columns(registry)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.registry, index=False, encoding="utf-8-sig")

    print_summary(registry, args.registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
