"""Tests for scripts/07_llm_extract_factors.py.

Module name starts with a digit so we load it via importlib.util.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "scripts" / "07_llm_extract_factors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("art57_extractor", SPEC_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so @dataclass can resolve __module__ → sys.modules.
    sys.modules["art57_extractor"] = mod
    spec.loader.exec_module(mod)
    return mod


extractor = _load_module()


def _good_factors_json() -> str:
    obj = {}
    for name, _ in extractor.ART57_FACTORS:
        obj[name] = {"direction": "neutral", "evidence": "略"}
    return json.dumps(obj, ensure_ascii=False)


def _mock_client(text_response: str, input_tokens: int = 100,
                 output_tokens: int = 200, cache_read: int = 0,
                 cache_write: int = 0) -> MagicMock:
    client = MagicMock()
    resp = SimpleNamespace(
        content=[SimpleNamespace(text=text_response)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )
    client.messages.create.return_value = resp
    return client


def _row(jid: str, jfull: str = "判決全文 主文 ...") -> dict:
    return {"jid": jid, "jtitle": "違反毒品危害防制條例", "jfull": jfull}


def test_extract_reason_text_trims_at_marker():
    sample = "前文略\n臺灣基隆地方法院\n主文\n甲處有期徒刑3月\n理由\n如上\n中華民國 115 年 1 月 1 日"
    out = extractor._extract_reason_text(sample)
    assert out.startswith("主文")
    assert "中華民國" not in out


def test_writes_one_line_per_input_case(tmp_path: Path):
    out_path = tmp_path / "out.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    rows = [_row("A"), _row("B"), _row("C")]
    client = _mock_client(_good_factors_json())

    summary = extractor.run_extraction(
        client=client, rows=rows, out_path=out_path,
        failures_path=fail_path, model="m", max_concurrency=1,
    )
    assert summary["ok"] == 3
    assert summary["failed"] == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    jids = {json.loads(l)["jid"] for l in lines}
    assert jids == {"A", "B", "C"}
    parsed = json.loads(lines[0])
    assert set(parsed["factors"].keys()) == {n for n, _ in extractor.ART57_FACTORS}


def test_resume_skips_already_extracted(tmp_path: Path):
    out_path = tmp_path / "out.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    out_path.write_text(
        json.dumps({"jid": "A", "factors": {}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [_row("A"), _row("B")]
    client = _mock_client(_good_factors_json())

    summary = extractor.run_extraction(
        client=client, rows=rows, out_path=out_path,
        failures_path=fail_path, model="m", max_concurrency=1,
    )
    assert summary["ok"] == 1
    assert client.messages.create.call_count == 1
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["jid"] == "B"


def test_malformed_json_logged_to_failures(tmp_path: Path):
    out_path = tmp_path / "out.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    client = _mock_client("this is not json at all, sorry")

    summary = extractor.run_extraction(
        client=client, rows=[_row("X")], out_path=out_path,
        failures_path=fail_path, model="m", max_concurrency=1,
    )
    assert summary["ok"] == 0
    assert summary["failed"] == 1
    assert not out_path.exists() or out_path.read_text(encoding="utf-8").strip() == ""
    fail_lines = fail_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(fail_lines) == 1
    rec = json.loads(fail_lines[0])
    assert rec["jid"] == "X"
    assert "parse" in rec["error"]


def test_missing_factor_treated_as_failure(tmp_path: Path):
    out_path = tmp_path / "out.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    incomplete = json.dumps({"motive": {"direction": "neutral", "evidence": ""}})
    client = _mock_client(incomplete)
    summary = extractor.run_extraction(
        client=client, rows=[_row("Y")], out_path=out_path,
        failures_path=fail_path, model="m", max_concurrency=1,
    )
    assert summary["failed"] == 1


def test_cost_estimate_is_finite():
    u = extractor.Usage(input_tokens=1000, output_tokens=500,
                         cache_read_tokens=2000, cache_write_tokens=500)
    cost = extractor._usd_cost(u)
    assert cost > 0
    assert cost < 1.0
