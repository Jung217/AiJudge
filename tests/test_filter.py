from filter import (
    NORTHERN_5_COURT_CODES,
    court_code_for,
    filter_drug_cases,
    is_drug_case,
    is_first_instance_guilty,
    is_keelung,
    keelung_drug_cases,
)
from records import Record


def _rec(jtitle: str = "", jfull: str = "", jcase: str = "訴",
         jid: str = "KLDM,112,訴,1,20230101,1") -> Record:
    return Record(
        jid=jid, jyear="112", jcase=jcase, jno="1",
        jdate="20230101", jtitle=jtitle, jfull=jfull,
    )


def test_keelung_accepts_official_form():
    r = _rec(jfull="臺灣基隆地方法院刑事判決\n主文\n...")
    assert is_keelung(r)


def test_keelung_accepts_variant_tai():
    r = _rec(jfull="台灣基隆地方法院刑事判決\n主文\n...")
    assert is_keelung(r)


def test_keelung_rejects_other_court():
    r = _rec(jid="TPDM,112,訴,1,20230101,1",
             jfull="臺灣臺北地方法院刑事判決\n主文\n...")
    assert not is_keelung(r)


def test_keelung_rejects_non_kl_jid_even_if_text_mentions_keelung():
    # Regression: Yunlin case merely citing a prior Keelung ruling
    r = _rec(jid="ULDM,114,虎簡,215,20260106,1",
             jfull="臺灣雲林地方法院刑事簡易判決\n前經臺灣基隆地方法院裁定...")
    assert not is_keelung(r)


def test_drug_case_by_title():
    assert is_drug_case(_rec(jtitle="違反毒品危害防制條例"))
    assert is_drug_case(_rec(jtitle="毒品"))


def test_drug_case_rejects_other():
    assert not is_drug_case(_rec(jtitle="詐欺"))
    assert not is_drug_case(_rec(jtitle="竊盜"))


def test_guilty_accept():
    r = _rec(jfull="主文\n甲○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n...")
    assert is_first_instance_guilty(r)


def test_not_guilty_rejected():
    r = _rec(jfull="主文\n甲○○無罪。\n犯罪事實\n...")
    assert not is_first_instance_guilty(r)


def test_dismiss_rejected():
    r = _rec(jfull="主文\n公訴不受理。\n犯罪事實\n...")
    assert not is_first_instance_guilty(r)


def test_sheng_case_rejected():
    r = _rec(jcase="聲",
             jfull="主文\n甲○○應送勒戒處所觀察勒戒。\n理由\n...")
    assert not is_first_instance_guilty(r)


def test_jianshang_rejected_second_instance():
    # 簡上 = 簡易判決上訴，地院第二審合議庭；不可與原 基簡 雙算
    r = _rec(jcase="簡上",
             jfull="臺灣基隆地方法院刑事判決\n主文\n原判決撤銷。\n甲○○施用第二級毒品，處有期徒刑3月。")
    assert not is_first_instance_guilty(r)


def test_pipeline_end_to_end():
    keelung_guilty = _rec(
        jid="KLDM,112,訴,1,20230101,1",
        jtitle="違反毒品危害防制條例",
        jfull="臺灣基隆地方法院刑事判決\n主文\n甲○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n...",
    )
    taipei_guilty = _rec(
        jid="TPDM,112,訴,2,20230101,1",
        jtitle="違反毒品危害防制條例",
        jfull="臺灣臺北地方法院刑事判決\n主文\n乙○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n...",
    )
    keelung_fraud = _rec(
        jid="KLDM,112,訴,3,20230101,1",
        jtitle="詐欺",
        jfull="臺灣基隆地方法院刑事判決\n主文\n丙○○犯詐欺罪，處有期徒刑壹年。\n犯罪事實\n...",
    )

    out = list(keelung_drug_cases([keelung_guilty, taipei_guilty, keelung_fraud]))
    assert len(out) == 1
    assert out[0].jid.startswith("KLDM")


def test_court_code_for_returns_2letter_prefix():
    keelung = _rec(jid="KLDM,112,訴,1,20230101,1",
                    jfull="臺灣基隆地方法院刑事判決\n主文\n...")
    taipei = _rec(jid="TPDM,112,訴,2,20230101,1",
                   jfull="臺灣臺北地方法院刑事判決\n主文\n...")
    new_taipei = _rec(jid="PCDM,112,訴,3,20230101,1",
                       jfull="臺灣新北地方法院刑事判決\n主文\n...")
    legacy_panqiao = _rec(jid="PCDM,101,訴,4,20120101,1",
                           jfull="臺灣板橋地方法院刑事判決\n主文\n...")
    yunlin = _rec(jid="ULDM,112,訴,5,20230101,1",
                   jfull="臺灣雲林地方法院刑事判決\n主文\n...")

    assert court_code_for(keelung, NORTHERN_5_COURT_CODES) == "KL"
    assert court_code_for(taipei, NORTHERN_5_COURT_CODES) == "TP"
    assert court_code_for(new_taipei, NORTHERN_5_COURT_CODES) == "PC"
    # 新北 was renamed from 板橋 in 2012, JID prefix retained — header name
    # still works since 板橋 is in the alias list.
    assert court_code_for(legacy_panqiao, NORTHERN_5_COURT_CODES) == "PC"
    # Non-northern court should not be picked up.
    assert court_code_for(yunlin, NORTHERN_5_COURT_CODES) is None


def test_filter_drug_cases_emits_court_code_for_each_record():
    cases = [
        _rec(jid="KLDM,112,訴,1,20230101,1", jtitle="毒品",
             jfull="臺灣基隆地方法院刑事判決\n主文\n甲○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n"),
        _rec(jid="TYDM,112,訴,2,20230101,1", jtitle="毒品",
             jfull="臺灣桃園地方法院刑事判決\n主文\n乙○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n"),
    ]
    out = list(filter_drug_cases(cases, NORTHERN_5_COURT_CODES))
    assert {c for _, c in out} == {"KL", "TY"}
