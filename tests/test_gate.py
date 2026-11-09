import json
import subprocess
import sys

import pytest

from relevance_gate.gate import compare, evaluate, markdown, paired_p_value
from relevance_gate.data import load
from relevance_gate.engines import Fts5Engine


def report(values, metric="ndcg@10"):
    return {"version": 1, "per_query": {f"q{i}": {"query": f"query {i}", metric: v} for i, v in enumerate(values)}}


def test_identical_reports_pass():
    a = report([0.5] * 100)
    checks, info = compare(a, a, {"ndcg@10": 0.0})
    assert info["shared"] == 100 and not checks[0].failed and not checks[0].worse


def test_big_drop_on_many_queries_fails():
    checks, _ = compare(report([0.8] * 200), report([0.7] * 200), {"ndcg@10": 0.01})
    assert checks[0].failed and checks[0].p < 0.001


def test_drop_on_few_queries_is_not_significant():
    base = report([0.9, 0.1, 0.5, 0.5, 0.4])
    cand = report([0.1, 0.2, 0.6, 0.4, 0.4])
    checks, info = compare(base, cand, {"ndcg@10": 0.01})
    assert checks[0].worse and not checks[0].failed
    assert "cannot be told apart from noise" in markdown(checks, info, base, cand, 0.05)


def test_tolerance_absorbs_small_drops():
    checks, _ = compare(report([0.80] * 200), report([0.799] * 200), {"ndcg@10": 0.005})
    assert not checks[0].worse


def test_zero_results_is_lower_is_better():
    base = report([0.0] * 200, "zero_results")
    worse = report([1.0] * 20 + [0.0] * 180, "zero_results")
    better = report([0.0] * 200, "zero_results")
    assert compare(base, worse, {"zero_results": 0.01})[0][0].failed
    assert not compare(worse, better, {"zero_results": 0.01})[0][0].worse


def test_p_value_is_deterministic_and_sane():
    a, b = [0.5] * 50, [0.5] * 25 + [0.4] * 25
    assert paired_p_value(a, b) == paired_p_value(a, b)
    assert paired_p_value(a, a) == 1.0
    assert paired_p_value([0, 1], [1, 0]) > 0.5


def test_missing_metric_is_an_error():
    with pytest.raises(ValueError, match="not in both reports"):
        compare(report([0.5]), report([0.5]), {"recall@100": 0.0})


def test_evaluate_on_toy(toy):
    ds = load(toy)
    good = evaluate(Fts5Engine(ds.docs), ds)
    flat = evaluate(Fts5Engine(ds.docs, tokenizer="unicode61"), ds)
    assert good["queries"] == 24 and good["metrics"]["ndcg@10"] > flat["metrics"]["ndcg@10"]
    assert set(good["per_query"]["q-volcano-0"]) >= {"query", "ndcg@10", "p@10", "recall@100", "mrr@10", "zero_results"}
    assert good["latency_ms"]["p50"] >= 0
    bare = evaluate(Fts5Engine(ds.docs), ds, query_text=False)
    assert "query" not in bare["per_query"]["q-volcano-0"]


def cli(*args, env=None):
    return subprocess.run([sys.executable, "-m", "relevance_gate", *args], capture_output=True, text=True, env=env)


def test_cli_end_to_end(toy, tmp_path):
    base, good, bad = tmp_path / "base.json", tmp_path / "good.json", tmp_path / "bad.json"
    assert cli("eval", "--data", str(toy), "-o", str(base)).returncode == 0
    assert cli("eval", "--data", str(toy), "-o", str(good)).returncode == 0  # same config: must pass
    assert cli("eval", "--data", str(toy), "--opt", "mode=and", "--opt", "tokenizer=unicode61", "-o", str(bad)).returncode == 0
    summary = tmp_path / "summary.md"
    r = cli("gate", str(base), str(bad), "--summary", str(summary))
    assert r.returncode == 1 and "FAIL" in r.stdout and "FAIL" in summary.read_text()
    r = cli("gate", str(base), str(good))
    assert r.returncode == 0, r.stdout
    assert json.loads(base.read_text())["engine"]["name"] == "fts5"


def test_cli_errors_are_clear(toy, tmp_path):
    r = cli("eval", "--data", str(tmp_path / "nope"))
    assert r.returncode == 2 and "missing" in r.stderr
    r = cli("eval", "--data", str(toy), "--opt", "boost=3")
    assert r.returncode == 2 and "fts5 options" in r.stderr
    r = cli("eval", "--data", str(toy), "--metrics", "map@10")
    assert r.returncode == 2 and "unknown metric" in r.stderr
