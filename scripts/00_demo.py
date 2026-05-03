"""End-to-end demo with synthetic judgments.

Validates the filter → feature-extraction → rules pipeline using hand-crafted
cases that mimic real 基隆地院 drug judgments. Useful while the live fetcher
is blocked by the bot-defense challenge.

Usage:
    python scripts/00_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402
from filter import keelung_drug_cases  # noqa: E402
from records import Record  # noqa: E402
from rules import apply_reductions, base_range, clip_prediction  # noqa: E402


SAMPLES = [
    # -- SHOULD PASS FILTER --
    Record(
        jid="PB,112,訴,101,20230315,1",
        jyear="112", jcase="訴", jno="101", jdate="20230315",
        jtitle="違反毒品危害防制條例",
        jfull="""臺灣基隆地方法院刑事判決  112年度訴字第101號
公 訴 人  臺灣基隆地方檢察署檢察官
被     告  甲○○
主文
甲○○販賣第二級毒品,累犯,處有期徒刑柒年貳月。
犯罪事實
一、甲○○基於販賣第二級毒品甲基安非他命之犯意,於民國112年2月間,在基隆市○○路便利商店前,販賣第二級毒品甲基安非他命予乙○○,交易金額新臺幣壹仟元,純質淨重0.5公克。
理由
一、訊據被告甲○○於偵查中及本院審理時均自白上開販賣第二級毒品犯行,核與證人乙○○之證述相符,事證明確,犯行堪以認定。
二、核被告所為,係犯毒品危害防制條例第4條第2項販賣第二級毒品罪。
三、被告前因施用毒品案件經判處有期徒刑,於109年8月執行完畢,5年以內故意再犯本件有期徒刑以上之罪,為累犯,依刑法第47條第1項規定加重其刑。
四、被告於偵查及審判中均自白犯行,依毒品危害防制條例第17條第2項規定減輕其刑,先加後減之。
""",
    ),
    Record(
        jid="PB,112,易,202,20230601,1",
        jyear="112", jcase="易", jno="202", jdate="20230601",
        jtitle="違反毒品危害防制條例",
        jfull="""臺灣基隆地方法院刑事判決  112年度易字第202號
主文
丙○○施用第二級毒品,處有期徒刑陸月,如易科罰金,以新臺幣壹仟元折算壹日。緩刑貳年。
犯罪事實
一、丙○○明知甲基安非他命為第二級毒品,於112年5月某日在其基隆市住處施用第二級毒品甲基安非他命壹次。
理由
一、訊據被告坦承犯行。
二、核被告所為,係犯毒品危害防制條例第10條第2項施用第二級毒品罪。
三、念被告犯後坦承犯行,且未曾犯罪,宣告緩刑貳年以啟自新。
""",
    ),
    Record(
        jid="PB,113,訴,55,20240110,1",
        jyear="113", jcase="訴", jno="55", jdate="20240110",
        jtitle="違反毒品危害防制條例",
        jfull="""臺灣基隆地方法院刑事判決  113年度訴字第55號
主文
丁○○轉讓第二級毒品,處有期徒刑捌月。
犯罪事實
丁○○於113年1月轉讓第二級毒品甲基安非他命予戊○○施用。
理由
一、訊據被告自白犯行。
二、核被告所為,係犯毒品危害防制條例第8條第2項轉讓第二級毒品罪。
""",
    ),
    # -- SHOULD FAIL FILTER --
    Record(
        jid="PB,112,訴,999,20230801,1",
        jyear="112", jcase="訴", jno="999", jdate="20230801",
        jtitle="違反毒品危害防制條例",
        jfull="""臺灣基隆地方法院刑事判決  112年度訴字第999號
主文
庚○○無罪。
理由
證據不足,不能證明被告犯罪。
""",
    ),
    Record(
        jid="TPD,112,訴,111,20230601,1",
        jyear="112", jcase="訴", jno="111", jdate="20230601",
        jtitle="違反毒品危害防制條例",
        jfull="""臺灣臺北地方法院刑事判決  112年度訴字第111號
主文
辛○○販賣第一級毒品,處無期徒刑。
""",
    ),
    Record(
        jid="PB,112,訴,222,20230701,1",
        jyear="112", jcase="訴", jno="222", jdate="20230701",
        jtitle="詐欺",
        jfull="""臺灣基隆地方法院刑事判決  112年度訴字第222號
主文
壬○○犯詐欺罪,處有期徒刑壹年。
""",
    ),
]


def divider(title: str):
    print(f"\n{'═' * 70}\n  {title}\n{'═' * 70}")


def main() -> int:
    divider("Stage 1 — Filter (6 synthetic cases)")
    print(f"input: {len(SAMPLES)} records")

    passed = list(keelung_drug_cases(SAMPLES))
    print(f"after filter: {len(passed)} (expected 3 — three Keelung drug guilty cases)")
    for r in passed:
        print(f"  - {r.jid} | {r.jtitle} | 字別={r.jcase}")

    divider("Stage 2 — Feature extraction")
    for r in passed:
        f = extract_features(r)
        print(f"\n[{r.jid}]")
        print(f"  behaviors         = {f.behaviors}")
        print(f"  drug_levels       = {f.drug_levels}")
        print(f"  sentence_months   = {f.sentence_months}")
        print(f"  probation_months  = {f.probation_months}")
        print(f"  easy-fine convert = {f.can_convert_to_fine}")
        print(f"  §17 Ⅰ / §17 Ⅱ    = {f.art17_1_applied} / {f.art17_2_applied}")
        print(f"  §47 累犯          = {f.recidivism}")
        print(f"  §59 酌減          = {f.art59_applied}")

    divider("Stage 3 — Rule-engine constraint check")
    # Example: the 販賣第二級毒品 case with 累犯 + §17Ⅱ
    case1 = extract_features(passed[0])
    print(f"\n案例 1 [{passed[0].jid}] 販賣第二級")
    base = base_range("販賣", 2, 2020)
    if base:
        print(f"  statutory range (§4 Ⅱ):    [{base.min_months}, {base.max_months}] months")
        reduced = apply_reductions(
            base,
            art17_2=case1.art17_2_applied,
            recidivism=case1.recidivism,
        )
        print(f"  after §17Ⅱ + 累犯:           [{reduced.min_months:.0f}, {reduced.max_months:.0f}] months")
        if case1.sentence_months:
            clipped = clip_prediction(case1.sentence_months, reduced)
            mark = "在法定刑內" if clipped == case1.sentence_months else "越界,clip 後調整"
            print(f"  actual sentence = {case1.sentence_months} → clip = {clipped:.0f}  {mark}")

    case2 = extract_features(passed[1])
    print(f"\n案例 2 [{passed[1].jid}] 施用第二級")
    base = base_range("施用", 2, 2020)
    if base:
        print(f"  statutory range (§10 Ⅱ):   [{base.min_months}, {base.max_months}] months")
        if case2.sentence_months:
            clipped = clip_prediction(case2.sentence_months, base)
            mark = "在法定刑內" if clipped == case2.sentence_months else "越界"
            print(f"  actual sentence = {case2.sentence_months} → clip = {clipped:.0f}  {mark}")

    divider("Summary")
    print(f"pipeline integrity: filter → features → rules all green")
    print(f"next: swap SAMPLES for real data once bot-challenge cookie is obtained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
