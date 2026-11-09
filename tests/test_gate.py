import json
import subprocess
import sys

import pytest

import itertools
import random
import statistics

from relevance_gate.gate import compare, drop_p_value, evaluate, markdown
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
    assert drop_p_value(a, b) == drop_p_value(a, b)
    assert drop_p_value(a, a) == 1.0
    assert drop_p_value(a, b) < 0.001
    assert drop_p_value(b, a) > 0.99  # an improvement is never evidence of a drop


def exact_p(a, b, allowed, up=True):
    """Every sign pattern, no sampling: the number the randomization test estimates."""
    sign = 1 if up else -1
    d = [sign * (y - x) + allowed for x, y in zip(a, b)]
    total = sum(d)
    flips = list(itertools.product((1, -1), repeat=len(d)))
    return sum(sum(s * v for s, v in zip(f, d)) <= total + 1e-12 for f in flips) / len(flips)


@pytest.mark.parametrize("seed", range(4))
def test_p_value_matches_exact_enumeration(seed):
    rng = random.Random(seed)
    a = [rng.random() for _ in range(13)]
    b = [min(1, max(0, x + rng.gauss(-0.05, 0.1))) for x in a]
    for allowed, up in ((0.0, True), (0.02, True), (0.01, False)):
        assert abs(drop_p_value(a, b, allowed, up, rounds=40000) - exact_p(a, b, allowed, up)) < 0.01


def test_test_is_one_sided():
    rng = random.Random(3)
    a = [rng.random() for _ in range(300)]
    b = [x + rng.gauss(-0.004, 0.05) for x in a]
    one = drop_p_value(a, b)
    flipped = drop_p_value(b, a)  # the same changes seen as a gain
    assert abs(one + flipped - 1) < 0.02 and one < 0.5


def test_allowed_drop_is_part_of_the_test():
    # 300 queries that lose 0.006 on average with per-query noise, 0.005 allowed: the evidence that the true
    # drop is beyond the allowed 0.005 is weak, so the gate must not fail (a test of "any drop" would).
    rng = random.Random(7)
    d = [rng.gauss(-0.006, 0.03) for _ in range(300)]
    m = statistics.fmean(d)
    a = [0.5] * 300
    b = [0.5 + x - m - 0.006 for x in d]
    checks, _ = compare(report(a), report(b), {"ndcg@10": 0.005})
    assert checks[0].worse and not checks[0].failed and checks[0].p > 0.2
    assert drop_p_value(a, b, 0.0) < 0.001  # but a drop of some size is certain


def test_reports_must_share_queries():
    with pytest.raises(ValueError, match="share 0 of the baseline's 3"):
        compare(report([0.5] * 3), {"version": 1, "per_query": {"other": {"ndcg@10": 0.1}}}, {"ndcg@10": 0.0})
    half = {"version": 1, "per_query": dict(list(report([0.5] * 10)["per_query"].items())[:4])}
    with pytest.raises(ValueError, match="share 4 of the baseline's 10"):
        compare(report([0.5] * 10), half, {"ndcg@10": 0.0})
    grown = report([0.5] * 20)  # more judged queries in the candidate is fine
    assert compare(report([0.5] * 10), grown, {"ndcg@10": 0.0})[1]["only_candidate"] == 10


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
    base, other = tmp_path / "base.json", tmp_path / "other.json"
    assert cli("eval", "--data", str(toy), "-o", str(base)).returncode == 0
    other.write_text(json.dumps({"version": 1, "per_query": {"x": {"ndcg@10": 0.0}}}))
    r = cli("gate", str(base), str(other))
    assert r.returncode == 2 and "cannot be compared" in r.stderr
