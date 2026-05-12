"""Feature-extraction helpers (partial — focuses on recently added logic)."""
from features import _count_defendants, extract_features
from records import Record


def _rec(jfull: str, jid: str = "KLDM,113,基簡,1,20240101,1") -> Record:
    return Record(jid=jid, jyear="113", jcase="基簡", jno="1",
                  jdate="20240101", jtitle="毒品危害防制條例", jfull=jfull)


def test_count_defendants_single():
    main = "陳○○施用第二級毒品，處有期徒刑肆月，如易科罰金，以新臺幣壹仟元折算壹日。"
    assert _count_defendants(main) == 1


def test_count_defendants_single_with_continuation_counts():
    # one defendant, two counts joined by 又 — must NOT be read as two people
    main = ("藍佳鈞施用第一級毒品，處有期徒刑肆月；又施用第二級毒品，"
            "處有期徒刑壹月。應執行有期徒刑伍月。")
    assert _count_defendants(main) == 1


def test_count_defendants_multi():
    main = ("羅中彥共同運輸第二級毒品，處有期徒刑5年6月。"
            "陳德名共同運輸第二級毒品，處有期徒刑7年2月。"
            "王健明販賣第三級毒品，處有期徒刑1年10月。")
    assert _count_defendants(main) >= 2


def test_count_defendants_falls_back_to_one():
    assert _count_defendants("") == 1
    assert _count_defendants("扣案如附表所示之物均沒收。") == 1


def test_extract_features_sets_n_defendants():
    f = extract_features(_rec(
        "臺灣基隆地方法院刑事簡易判決\n主文\n"
        "甲○○施用第二級毒品，處有期徒刑3月，如易科罰金，以新臺幣1000元折算1日。\n"
        "犯罪事實\n..."))
    assert f.n_defendants == 1
    assert f.n_sentence_counts == 1
    assert f.is_aggregate_sentence is False
    assert f.is_attempt is False
    assert f.sentence_months == 3
    assert "施用" in f.convicted_behaviors
    assert f.convicted_drug_levels == [2]


def test_extract_features_aggregate_and_count():
    f = extract_features(_rec(
        "臺灣基隆地方法院刑事判決\n主文\n"
        "甲○○施用第一級毒品，處有期徒刑肆月；又施用第二級毒品，處有期徒刑壹月。"
        "應執行有期徒刑伍月，如易科罰金，以新臺幣壹仟元折算壹日。\n犯罪事實\n..."))
    assert f.n_sentence_counts == 2
    assert f.is_aggregate_sentence is True
    assert f.sentence_months == 5          # the 應執行 aggregate, not the first count


def test_extract_features_detects_attempt():
    f = extract_features(_rec(
        "臺灣基隆地方法院刑事判決\n主文\n"
        "甲○○販賣第二級毒品未遂，處有期徒刑貳年。\n犯罪事實\n..."))
    assert f.is_attempt is True
    assert f.sentence_months == 24
