"""Layer-4 statutory constraint engine."""
import pytest

from models import constrain_sentence
from rules import (
    MAX_AGGREGATE_TERM_MONTHS,
    MAX_FIXED_TERM_MONTHS,
    SentencingConstraint,
    aggregate_only_constraint,
    aggregate_sentence_bounds,
    apply_reductions,
    base_range,
    binding_constraint,
    clip_prediction,
    is_within,
    violation_rate,
)


def test_base_range_lookup():
    c = base_range("施用", 2, 2020)
    assert c is not None
    assert (c.min_months, c.max_months) == (0, 36)
    assert base_range("販賣", 2, 2020).includes_life is True
    assert base_range("nonsense", 9, 2020) is None


def test_single_reduction_halves_both_bounds():
    base = SentencingConstraint(120, 360)          # 10–30 年
    r = apply_reductions(base, art17_2=True)
    assert (r.min_months, r.max_months) == (60, 180)


def test_reductions_compound_per_art70():
    base = SentencingConstraint(120, 360)
    r = apply_reductions(base, art17_1=True, art59=True)   # ×1/2 ×1/2
    assert (r.min_months, r.max_months) == (30, 90)


def test_attempt_counts_as_a_reduction():
    base = SentencingConstraint(120, 360)
    r = apply_reductions(base, attempt=True)               # §25Ⅱ ×1/2
    assert (r.min_months, r.max_months) == (60, 180)
    r2 = apply_reductions(base, art17_2=True, attempt=True)  # ×1/2 ×1/2
    assert (r2.min_months, r2.max_months) == (30, 90)


def test_recidivism_raises_only_upper_bound_and_caps_at_30y():
    base = SentencingConstraint(60, 12 * 30)       # upper already 30Y
    r = apply_reductions(base, recidivism=True)
    assert r.min_months == 60
    assert r.max_months == MAX_FIXED_TERM_MONTHS   # 1.5× clipped back to 30Y

    base2 = SentencingConstraint(12, 60)
    r2 = apply_reductions(base2, recidivism=True)
    assert (r2.min_months, r2.max_months) == (12, 90)


def test_reduction_drops_life_and_capital():
    base = base_range("販賣", 1, 2020)
    assert base.includes_capital and base.includes_life
    r = apply_reductions(base, art17_1=True)
    assert not r.includes_capital and not r.includes_life


def test_clip_prediction():
    c = SentencingConstraint(6, 60)
    assert clip_prediction(3, c) == 6
    assert clip_prediction(100, c) == 60
    assert clip_prediction(24, c) == 24


def test_is_within_tolerates_half_month_fractions():
    c = apply_reductions(SentencingConstraint(7, 63), art59=True)  # → (3.5, 31.5)
    assert is_within(3.5, c)
    assert is_within(31.5, c)
    assert not is_within(3.0, c)
    assert not is_within(32.0, c)


def test_aggregate_sentence_bounds_art51():
    assert aggregate_sentence_bounds([12, 8, 6]) == (12, 26)
    assert aggregate_sentence_bounds([]) == (0.0, 0.0)
    # 上限 capped at 30 年
    lo, hi = aggregate_sentence_bounds([200, 200])
    assert (lo, hi) == (200, MAX_FIXED_TERM_MONTHS)


def test_binding_constraint_picks_most_severe_behavior_and_lowest_level():
    # 販賣 (§4) dominates 持有 (§11); 第一級 dominates 第二級.
    c = binding_constraint({"持有", "販賣"}, {2, 1})
    assert c is not None
    assert c.min_months == base_range("販賣", 1, 2020).min_months


def test_binding_constraint_none_when_unmapped_or_empty():
    assert binding_constraint(set(), {2}) is None
    assert binding_constraint({"施用"}, set()) is None
    assert binding_constraint({"持有"}, {9}) is None  # level 9 not in table


def test_binding_constraint_summary_halves_only_the_floor():
    plain = binding_constraint({"施用"}, {1})              # §10Ⅰ [6, 60]
    summ = binding_constraint({"施用"}, {1}, summary=True)
    assert (plain.min_months, plain.max_months) == (6, 60)
    assert (summ.min_months, summ.max_months) == (3, 60)
    # no effect when the floor is already 0
    assert binding_constraint({"施用"}, {2}, summary=True).min_months == 0


def test_binding_constraint_passes_attempt_through():
    base = binding_constraint({"販賣"}, {3})               # §4Ⅲ [84, 360]
    att = binding_constraint({"販賣"}, {3}, attempt=True)
    assert (att.min_months, att.max_months) == (base.min_months / 2,
                                                base.max_months / 2)


def test_aggregate_only_constraint():
    c = aggregate_only_constraint()
    assert (c.min_months, c.max_months) == (0.0, MAX_AGGREGATE_TERM_MONTHS)
    assert is_within(0, c) and is_within(MAX_AGGREGATE_TERM_MONTHS, c)
    assert not is_within(MAX_AGGREGATE_TERM_MONTHS + 1, c)


def test_violation_rate_excludes_unmapped():
    c = SentencingConstraint(6, 60)
    rate, n_viol, n_chk = violation_rate([(3, c), (24, c), (100, c), (999, None)])
    assert n_chk == 3          # the None case dropped from the denominator
    assert n_viol == 2         # 3 and 100 both out of [6, 60]; 24 is in
    assert rate == pytest.approx(2 / 3)
    assert violation_rate([]) == (0.0, 0, 0)
    assert violation_rate([(5, None)]) == (0.0, 0, 0)


def test_constrain_sentence_clips_all_quantiles_into_range():
    c = SentencingConstraint(12, 60)
    out = constrain_sentence({"p25": 4, "p50": 30, "p75": 200}, c)
    assert out["rule_applied"] is True
    assert out["clipped"] is True
    assert out["p25_months"] == 12
    assert out["p50_months"] == 30
    assert out["p75_months"] == 60
    assert all(is_within(out[f"{q}_months"], c) for q in ("p25", "p50", "p75"))
    assert out["raw"] == {"p25": 4, "p50": 30, "p75": 200}


def test_constrain_sentence_keeps_quantiles_monotone():
    # raw quantiles inverted by clipping at the floor must stay non-decreasing
    c = SentencingConstraint(24, 360)
    out = constrain_sentence({"p25": 1, "p50": 2, "p75": 3}, c)
    assert out["p25_months"] <= out["p50_months"] <= out["p75_months"] == 24


def test_constrain_sentence_passthrough_without_constraint():
    out = constrain_sentence({"p50": 7.5}, None)
    assert out["rule_applied"] is False
    assert out["clipped"] is False
    assert out["p50_months"] == 7.5
    assert out["constraint"] is None
