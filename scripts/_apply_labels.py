"""Apply LLM-generated ground-truth labels to sample_100.csv.

Each entry in OVERRIDES is {jid: {gt_field: value}}. Fields not listed
keep their --prefill value (i.e. agree with auto_*).
NOTES annotations are appended to the notes column.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

CSV_PATH = Path("data/labeling/sample_100.csv")

# Ground-truth overrides (only where I disagree with auto_*).
# Each value is the corrected gt_* cell; missing fields keep auto value.
OVERRIDES: dict[str, dict[str, str]] = {
    # --- 1-30 ---
    "KLDM,112,基簡,353,20230428,1": {"gt_art59": "0"},  # #4: §59 boilerplate FP
    "KLDM,113,基原簡,42,20240614,1": {"gt_sentence_months": "3"},  # #8: 應執行3月
    "KLDM,113,訴,210,20250711,1": {  # #14: multi-defendant, levels FP, 應執行24月
        "gt_drug_levels": "3",
        "gt_sentence_months": "24",
        # §17Ⅰ IS applied to 甲○○ ("足認被告甲○○有供出毒品來源因而查獲正犯之情事")
        # §59 IS applied to 乙○○ ("爰依刑法第59條之規定，予以減輕其刑")
    },
    "KLDM,114,基簡,1129,20251217,1": {"gt_sentence_months": "9"},  # #16: 應執行9月
    "KLDM,113,基簡,415,20240628,1": {"gt_behaviors": "施用,持有"},  # #17
    "KLDM,113,基簡,1034,20240918,1": {"gt_sentence_months": "3"},  # #18
    "KLDM,113,訴,49,20240606,1": {"gt_art17_2": "1"},  # #19: §17Ⅱ FN by auto
    "KLDM,113,基簡,917,20240822,1": {"gt_art59": "0"},  # #20
    "KLDM,113,基原簡,50,20240726,1": {"gt_art59": "0"},  # #21
    "KLDM,113,訴,122,20241004,1": {"gt_can_convert_to_fine": "1"},  # #22
    "KLDM,113,基簡,1413,20241224,1": {  # #24
        "gt_behaviors": "持有",
        "gt_art59": "0",
    },
    "KLDM,112,基簡,1281,20240205,1": {"gt_art59": "0"},  # #27
    "KLDM,112,基簡,225,20230427,1": {"gt_art59": "0"},  # #28
    "KLDM,114,基簡,13,20250203,1": {"gt_can_convert_to_fine": "1"},  # #29
    "KLDM,112,基簡,401,20230510,1": {"gt_art59": "0"},  # #30
    # --- 31-60 ---
    "KLDM,112,原訴,13,20240131,1": {  # #31
        "gt_drug_levels": "2",
        "gt_sentence_months": "42",
        "gt_behaviors": "販賣,持有",
        "gt_recidivism": "0",
    },
    "KLDM,113,基簡,1384,20241217,1": {"gt_recidivism": "0"},  # #38
    "KLDM,113,基原簡,12,20240130,1": {"gt_sentence_months": "4"},  # #39
    "KLDM,112,訴,353,20240220,1": {  # #40
        "gt_drug_levels": "2",
        "gt_sentence_months": "62",
        "gt_behaviors": "販賣,轉讓,持有",
        "gt_art17_1": "0",
    },
    "KLDM,112,基簡,995,20230928,1": {"gt_sentence_months": "5"},  # #42
    "KLDM,112,基簡,1105,20231031,1": {  # #49
        "gt_behaviors": "施用,持有",
        "gt_art59": "0",
    },
    "KLDM,113,基簡,252,20240301,1": {"gt_sentence_months": "7"},  # #50
    "KLDM,113,基簡,932,20240919,1": {"gt_art59": "0"},  # #51
    "KLDM,112,易,577,20231221,1": {  # #54
        "gt_sentence_months": "18",
        "gt_art59": "0",  # judgment: "復無任何符合刑法第59條"
    },
    "KLDM,112,訴,70,20230606,1": {"gt_sentence_months": "9"},  # #55
    "KLDM,112,易,727,20240117,1": {"gt_sentence_months": "7"},  # #56
    "KLDM,113,易,257,20240605,1": {"gt_art59": "0"},  # #59
    "KLDM,112,訴,344,20240613,2": {  # #53
        "gt_art59": "0",  # judgment rejects §59
        "gt_sentence_months": "88",  # first defendant 周家丞 single-罪 7年4月
    },
    "KLDM,111,重訴,6,20230111,2": {  # #60
        "gt_sentence_months": "82",  # first defendant 羅中彥 應執行 6年10月
    },
    # --- 61-100 ---
    "KLDM,112,基簡,122,20230204,1": {"gt_sentence_months": "6"},  # #70
    "KLDM,112,基簡,806,20230831,1": {"gt_sentence_months": "5"},  # #71
    "KLDM,112,基簡,1006,20231003,1": {"gt_behaviors": "施用,持有"},  # #79
    "KLDM,113,基簡,99,20240131,1": {"gt_art59": "0"},  # #83
    "KLDM,112,基簡,964,20230926,1": {"gt_behaviors": "持有"},  # #86
    "KLDM,114,基簡,775,20250917,1": {  # #90
        "gt_behaviors": "持有",
        "gt_drug_levels": "2",
    },
    "KLDM,112,基簡,127,20230209,1": {"gt_sentence_months": "7"},  # #92
    "KLDM,112,基簡,483,20230601,1": {"gt_behaviors": "持有"},  # #94
    "KLDM,112,基簡,19,20230209,1": {  # #97
        "gt_drug_levels": "3",
        "gt_behaviors": "持有",
    },
    "KLDM,113,基簡,387,20240409,1": {  # #100
        "gt_art59": "0",
        "gt_behaviors": "施用,持有",
    },
}

# Notes for cases (e.g. multi-defendant flagged, §62 自首 noted)
NOTES: dict[str, str] = {
    "KLDM,112,基簡,1225,20231211,1": "multi-罪 施用+持有",  # #6
    "KLDM,113,基原簡,42,20240614,1": "multi-罪 兩次施用, 應執行3月",  # #8
    "KLDM,114,基簡,828,20251020,1": "§62 自首",  # #10
    "KLDM,113,訴,210,20250711,1": "multi-defendant 甲乙丙",  # #14
    "KLDM,114,基簡,1129,20251217,1": "multi-罪 三次, 應執行9月; §62 自首",  # #16
    "KLDM,113,基簡,1034,20240918,1": "multi-罪 兩次, 應執行3月",  # #18
    "KLDM,113,訴,49,20240606,1": "§17Ⅱ FN by auto",  # #19
    "KLDM,113,基簡,917,20240822,1": "§62 自首",  # #20
    "KLDM,112,基簡,1088,20231031,1": "§62 自首",  # #25
    "KLDM,112,基簡,225,20230427,1": "§62 自首",  # #28
    "KLDM,112,原訴,13,20240131,1": "multi-defendant 5次販賣, 應執行42月; 檢察官未主張累犯",  # #31
    "KLDM,113,基簡,1384,20241217,1": "法院未認定累犯加重",  # #38
    "KLDM,113,基原簡,12,20240130,1": "multi-罪 兩次, 應執行4月",  # #39
    "KLDM,112,訴,353,20240220,1": "大量罪 11+2+1次, 應執行62月",  # #40
    "KLDM,112,基簡,995,20230928,1": "multi-罪 兩次, 應執行5月",  # #42
    "KLDM,113,訴,16,20240422,1": "behaviors 含轉讓禁藥",  # #45
    "KLDM,113,基簡,252,20240301,1": "multi-罪, 應執行7月",  # #50
    "KLDM,112,訴,344,20240613,2": "multi-defendant 大型製造案",  # #53
    "KLDM,112,易,577,20231221,1": "multi-罪, 應執行1年6月=18月",  # #54
    "KLDM,112,訴,70,20230606,1": "multi-罪, 應執行9月; §62 自首",  # #55
    "KLDM,112,易,727,20240117,1": "multi-罪, 應執行7月",  # #56
    "KLDM,111,重訴,6,20230111,2": "multi-defendant 大型運輸案",  # #60
    "KLDM,112,基簡,849,20230906,1": "§62 自首",  # #61
    "KLDM,114,基簡,462,20250618,1": "持有第三級因前案已執行",  # #62
    "KLDM,114,基簡,384,20250801,1": "§62 自首",  # #65
    "KLDM,114,基簡,559,20250708,1": "§62 自首",  # #66
    "KLDM,113,基簡,1136,20240927,1": "§62 自首",  # #73
    "KLDM,112,基簡,375,20230501,1": "§62 自首",  # #75
    "KLDM,113,基簡,928,20240819,1": "§62 自首",  # #76
    "KLDM,114,基簡,672,20250813,1": "§62 自首",  # #77
    "KLDM,112,基簡,122,20230204,1": "multi-罪, 應執行6月; §62 自首",  # #70
    "KLDM,112,基簡,806,20230831,1": "multi-罪 兩次, 應執行5月",  # #71
    "KLDM,113,易,342,20240711,1": "§62 自首",  # #81
    "KLDM,113,基簡,591,20240528,1": "§62 自首",  # #84
    "KLDM,114,基簡,775,20250917,1": "§62 自首",  # #90
    "KLDM,112,基簡,127,20230209,1": "multi-罪, 應執行7月",  # #92
    "KLDM,112,易,659,20231229,1": "§62 自首",  # #93
    "KLDM,112,基簡,355,20230509,1": "§62 自首",  # #95
    "KLDM,113,基簡,387,20240409,1": "§62 自首",  # #100
}


def main() -> int:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))
    fieldnames = rows[0].keys()

    n_overridden = 0
    n_noted = 0
    for r in rows:
        jid = r["jid"]
        if jid in OVERRIDES:
            for field, val in OVERRIDES[jid].items():
                r[field] = val
            n_overridden += 1
        if jid in NOTES:
            r["notes"] = NOTES[jid]
            n_noted += 1

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)

    print(f"Updated {CSV_PATH}")
    print(f"  rows with at least one gt_* override: {n_overridden}")
    print(f"  rows with notes: {n_noted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
