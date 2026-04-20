"""Statutory constraints — the rule engine (hard constraints).

The ML model's output must be clipped to the ranges defined here. The table
below must be maintained in sync with 毒品危害防制條例 and 刑法 amendments.

Current coverage is partial — see TODO. Must be verified by a legal expert
before production use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MAX_FIXED_TERM_MONTHS = 12 * 30       # 有期徒刑上限 30 年
MAX_AGGREGATE_TERM_MONTHS = 12 * 30   # 定應執行刑上限 30 年


@dataclass
class SentencingConstraint:
    min_months: float
    max_months: float
    includes_life: bool = False
    includes_capital: bool = False


# key: (behavior, drug_level, law_version_year)
# TODO v2: complete the table with 2020-01-15 and 2023-05-30 versions.
#   Source: 毒品危害防制條例 §4–§11、刑法 §10、§47。
#   For §4 Ⅰ (第一級販賣/運輸/製造) there is NO 有期徒刑 option in the statute
#   itself (only 死刑/無期徒刑); a fixed term can only arise from §17 reductions.
_PENALTY_TABLE: dict[tuple[str, int, int], SentencingConstraint] = {
    # 毒品條例 §4 Ⅱ 販賣第二級毒品：無期徒刑或 10 年以上有期徒刑
    ("販賣", 2, 2020): SentencingConstraint(10 * 12, MAX_FIXED_TERM_MONTHS, includes_life=True),
    # 毒品條例 §4 Ⅲ 販賣第三級毒品：7 年以上
    ("販賣", 3, 2020): SentencingConstraint(7 * 12, MAX_FIXED_TERM_MONTHS),
    # 毒品條例 §4 Ⅳ 販賣第四級毒品：5 年以上 12 年以下
    ("販賣", 4, 2020): SentencingConstraint(5 * 12, 12 * 12),
    # 毒品條例 §10 Ⅰ 施用第一級毒品：6 月以上 5 年以下
    ("施用", 1, 2020): SentencingConstraint(6, 5 * 12),
    # 毒品條例 §10 Ⅱ 施用第二級毒品：3 年以下
    ("施用", 2, 2020): SentencingConstraint(0, 3 * 12),
}


def base_range(
    behavior: str,
    drug_level: int,
    law_version: int = 2020,
) -> Optional[SentencingConstraint]:
    """Look up statutory min/max for the given (behavior, drug_level, law_version)."""
    return _PENALTY_TABLE.get((behavior, drug_level, law_version))


def apply_reductions(
    base: SentencingConstraint,
    art17_1: bool = False,
    art17_2: bool = False,
    art59: bool = False,
    recidivism: bool = False,
) -> SentencingConstraint:
    """Apply statutory reductions and recidivism enhancement.

    刑法§70: each 減輕其刑 halves the range (both min and max).
    Multiple reductions compound multiplicatively per §70.
    """
    lo, hi = base.min_months, base.max_months

    reduction_factor = 1.0
    for flag in (art17_1, art17_2, art59):
        if flag:
            reduction_factor *= 0.5
    lo *= reduction_factor
    hi *= reduction_factor

    if recidivism:
        lo = min(lo * 1.5, MAX_FIXED_TERM_MONTHS)
        hi = min(hi * 1.5, MAX_FIXED_TERM_MONTHS)

    # Once reduced, life/capital options do not auto-apply
    any_reduction = art17_1 or art17_2 or art59
    return SentencingConstraint(
        min_months=lo,
        max_months=hi,
        includes_life=base.includes_life and not any_reduction,
        includes_capital=base.includes_capital and not any_reduction,
    )


def clip_prediction(pred_months: float, constraint: SentencingConstraint) -> float:
    """Clip a raw ML prediction to the legally valid range."""
    return max(constraint.min_months, min(pred_months, constraint.max_months))


def aggregate_sentence_bounds(individual_months: list[float]) -> tuple[float, float]:
    """刑法§51 合併定刑上下限。

    下限 = 個刑中之最長期
    上限 = min(個刑合計, 30 年)
    """
    if not individual_months:
        return 0.0, 0.0
    return max(individual_months), min(sum(individual_months), MAX_AGGREGATE_TERM_MONTHS)
