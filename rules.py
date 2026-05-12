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
    # 毒品條例 §4 Ⅰ 販賣/運輸/製造第一級毒品：死刑/無期徒刑/15年以上
    # No statutory 有期徒刑 floor below 15Y; fixed-term reachable only via §17 reduction.
    ("販賣", 1, 2020): SentencingConstraint(15 * 12, MAX_FIXED_TERM_MONTHS,
                                              includes_life=True, includes_capital=True),
    ("運輸", 1, 2020): SentencingConstraint(15 * 12, MAX_FIXED_TERM_MONTHS,
                                              includes_life=True, includes_capital=True),
    ("製造", 1, 2020): SentencingConstraint(15 * 12, MAX_FIXED_TERM_MONTHS,
                                              includes_life=True, includes_capital=True),
    # 毒品條例 §4 Ⅱ 第二級毒品：無期徒刑或 10 年以上有期徒刑
    ("販賣", 2, 2020): SentencingConstraint(10 * 12, MAX_FIXED_TERM_MONTHS, includes_life=True),
    ("運輸", 2, 2020): SentencingConstraint(10 * 12, MAX_FIXED_TERM_MONTHS, includes_life=True),
    ("製造", 2, 2020): SentencingConstraint(10 * 12, MAX_FIXED_TERM_MONTHS, includes_life=True),
    # 毒品條例 §4 Ⅲ 第三級毒品：7 年以上
    ("販賣", 3, 2020): SentencingConstraint(7 * 12, MAX_FIXED_TERM_MONTHS),
    ("運輸", 3, 2020): SentencingConstraint(7 * 12, MAX_FIXED_TERM_MONTHS),
    ("製造", 3, 2020): SentencingConstraint(7 * 12, MAX_FIXED_TERM_MONTHS),
    # 毒品條例 §4 Ⅳ 第四級毒品：5–12 年
    ("販賣", 4, 2020): SentencingConstraint(5 * 12, 12 * 12),
    ("運輸", 4, 2020): SentencingConstraint(5 * 12, 12 * 12),
    ("製造", 4, 2020): SentencingConstraint(5 * 12, 12 * 12),
    # 毒品條例 §5 意圖販賣而持有
    ("意圖販賣而持有", 1, 2020): SentencingConstraint(10 * 12, MAX_FIXED_TERM_MONTHS, includes_life=True),
    ("意圖販賣而持有", 2, 2020): SentencingConstraint(5 * 12, 12 * 12),
    ("意圖販賣而持有", 3, 2020): SentencingConstraint(3 * 12, 10 * 12),
    ("意圖販賣而持有", 4, 2020): SentencingConstraint(1 * 12, 7 * 12),
    # 毒品條例 §8 轉讓
    ("轉讓", 1, 2020): SentencingConstraint(1 * 12, 7 * 12),
    ("轉讓", 2, 2020): SentencingConstraint(6, 5 * 12),
    ("轉讓", 3, 2020): SentencingConstraint(0, 3 * 12),
    ("轉讓", 4, 2020): SentencingConstraint(0, 1 * 12),
    # 毒品條例 §10 施用
    ("施用", 1, 2020): SentencingConstraint(6, 5 * 12),
    ("施用", 2, 2020): SentencingConstraint(0, 3 * 12),
    # 毒品條例 §11 持有 — 第三/四級「純質淨重 ≥ 5 公克」加重型 (§11Ⅴ/Ⅵ)
    # not included here (would require 純質淨重 split). The general (low-amount)
    # rules are listed; weight-triggered enhancement is handled by the caller.
    ("持有", 1, 2020): SentencingConstraint(0, 3 * 12),       # §11Ⅰ 3年以下
    ("持有", 2, 2020): SentencingConstraint(0, 2 * 12),       # §11Ⅱ 2年以下
    ("持有", 3, 2020): SentencingConstraint(0, 1 * 12),       # §11Ⅲ 1年以下
    ("持有", 4, 2020): SentencingConstraint(0, 1 * 12),       # §11Ⅳ 1年以下
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

    刑法§66: 有期徒刑減輕其刑「至二分之一」— the reduced range has both
    lower and upper bounds scaled by 1/2.
    刑法§70: multiple reductions compound (each ×1/2).
    刑法§47 + 釋字775: 累犯加重「至二分之一」enhances the *upper* bound only;
    the lower bound stays untouched to allow proportionality (otherwise
    a reduced §17 case combined with 累犯 would have an impossibly high floor).
    """
    lo, hi = base.min_months, base.max_months

    reduction_factor = 1.0
    for flag in (art17_1, art17_2, art59):
        if flag:
            reduction_factor *= 0.5
    lo *= reduction_factor
    hi *= reduction_factor

    if recidivism:
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


# Floating-point slack for boundary comparisons (reductions can yield .5-month
# fractions; statutory bounds are integer months).
_BOUND_EPS = 1e-6


def is_within(months: float, constraint: SentencingConstraint) -> bool:
    """Whether a sentence (in months) lies inside the statutory range."""
    return (constraint.min_months - _BOUND_EPS
            <= months
            <= constraint.max_months + _BOUND_EPS)


def violation_rate(
    sentences_and_constraints: list[tuple[float, Optional[SentencingConstraint]]],
) -> tuple[float, int, int]:
    """法定刑越界率 over a batch.

    Cases with no applicable constraint (``None``) are excluded from the
    denominator — they cannot be judged in or out of range.

    Returns ``(rate, n_violations, n_with_constraint)``.
    """
    checked = [(m, c) for m, c in sentences_and_constraints if c is not None]
    if not checked:
        return 0.0, 0, 0
    n_viol = sum(1 for m, c in checked if not is_within(m, c))
    return n_viol / len(checked), n_viol, len(checked)


def aggregate_sentence_bounds(individual_months: list[float]) -> tuple[float, float]:
    """刑法§51 合併定刑上下限。

    下限 = 個刑中之最長期
    上限 = min(個刑合計, 30 年)
    """
    if not individual_months:
        return 0.0, 0.0
    return max(individual_months), min(sum(individual_months), MAX_AGGREGATE_TERM_MONTHS)


# Severity ranking for picking the "primary" behavior in multi-act cases.
# Judges typically apply the highest-tier statute (高度行為吸收低度行為);
# §4 (販賣/運輸/製造) > §5 (意圖販賣而持有) > §8 (轉讓) > §10 (施用) > §11 (持有).
_BEHAVIOR_SEVERITY: dict[str, int] = {
    "販賣": 5, "運輸": 5, "製造": 5,
    "意圖販賣而持有": 4,
    "轉讓": 3,
    "施用": 2,
    "持有": 1,
}


def binding_constraint(
    behaviors: set[str],
    drug_levels: set[int],
    art17_1: bool = False,
    art17_2: bool = False,
    art59: bool = False,
    recidivism: bool = False,
    law_version: int = 2020,
) -> Optional[SentencingConstraint]:
    """Strictest applicable sentencing constraint for a case.

    Picks the most-severe behavior (e.g. 販賣 over 持有) and the lowest
    drug level (e.g. 第一級 over 第三級), looks up base range, then applies
    statutory reductions / 累犯 enhancement. Returns None if the
    (behavior, level) tuple is not in the penalty table.
    """
    if not behaviors or not drug_levels:
        return None
    primary = max(behaviors, key=lambda b: _BEHAVIOR_SEVERITY.get(b, 0))
    primary_level = min(drug_levels)
    base = base_range(primary, primary_level, law_version)
    if base is None:
        return None
    return apply_reductions(
        base, art17_1=art17_1, art17_2=art17_2,
        art59=art59, recidivism=recidivism,
    )
