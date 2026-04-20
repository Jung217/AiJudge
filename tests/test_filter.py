from filter import (
    is_drug_case,
    is_first_instance_guilty,
    is_keelung,
    keelung_drug_cases,
)
from records import Record


def _rec(jtitle: str = "", jfull: str = "", jcase: str = "訴") -> Record:
    return Record(
        jid="TEST", jyear="112", jcase=jcase, jno="1",
        jdate="20230101", jtitle=jtitle, jfull=jfull,
    )


def test_keelung_accepts_official_form():
    r = _rec(jfull="臺灣基隆地方法院刑事判決\n主文\n...")
    assert is_keelung(r)


def test_keelung_accepts_variant_tai():
    r = _rec(jfull="台灣基隆地方法院刑事判決\n主文\n...")
    assert is_keelung(r)


def test_keelung_rejects_other_court():
    r = _rec(jfull="臺灣臺北地方法院刑事判決\n主文\n...")
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


def test_pipeline_end_to_end():
    keelung_guilty = _rec(
        jtitle="違反毒品危害防制條例",
        jfull="臺灣基隆地方法院刑事判決\n主文\n甲○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n...",
    )
    taipei_guilty = _rec(
        jtitle="違反毒品危害防制條例",
        jfull="臺灣臺北地方法院刑事判決\n主文\n乙○○販賣第二級毒品，處有期徒刑柒年陸月。\n犯罪事實\n...",
    )
    keelung_fraud = _rec(
        jtitle="詐欺",
        jfull="臺灣基隆地方法院刑事判決\n主文\n丙○○犯詐欺罪，處有期徒刑壹年。\n犯罪事實\n...",
    )

    out = list(keelung_drug_cases([keelung_guilty, taipei_guilty, keelung_fraud]))
    assert len(out) == 1
    assert out[0].jid == "TEST"
