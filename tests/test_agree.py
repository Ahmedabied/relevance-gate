import subprocess
import sys

import pytest

from relevance_gate.agree import band, compare, interval, kappa, markdown


def test_kappa_textbook_example():
    # 50 items: both yes 20, A yes B no 5, A no B yes 10, both no 15. Observed 0.70, chance 0.50, kappa 0.40.
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert kappa(a, b) == pytest.approx(0.40)
    assert band(0.40) == "fair" and band(0.85) == "almost perfect" and band(None) == "not defined"


def test_weighted_kappa_by_hand():
    # Worked by hand: observed weighted disagreement 0.0625, chance 0.3125, so 1 - 0.0625 / 0.3125 = 0.8.
    assert kappa([0, 0, 1, 2], [0, 1, 1, 2], "quadratic") == pytest.approx(0.8)
    # A one-step miss costs a quarter of a two-step miss under quadratic weights, half under linear.
    near = kappa([0, 1, 2, 2], [0, 1, 2, 1], "quadratic")
    far = kappa([0, 1, 2, 2], [0, 1, 2, 0], "quadratic")
    assert near > far
    assert kappa([0, 1, 2], [0, 1, 2], "quadratic") == 1.0 and kappa([0, 1, 2], [0, 1, 2], "linear") == 1.0


def test_kappa_is_not_defined_when_chance_explains_everything():
    assert kappa([0, 0, 0], [0, 0, 0]) is None
    assert kappa([], []) is None


def test_compare_counts_and_orders_disagreements():
    a = {"q1": {"d1": 2, "d2": 0, "d3": 1}, "q2": {"d9": 2}}
    b = {"q1": {"d1": 0, "d2": 0, "d3": 2, "d4": 1}}
    ag = compare(a, b)
    assert len(ag.pairs) == 3 and ag.only_a == 1 and ag.only_b == 1
    assert ag.exact() == pytest.approx(1 / 3)
    assert [(q, d) for q, d, _, _ in ag.disagreements()] == [("q1", "d1"), ("q1", "d3")]  # the 2-step gap first


def test_interval_brackets_the_estimate_and_is_deterministic():
    a = [0, 1, 2, 2, 1, 0, 0, 2, 1, 1] * 8
    b = [0, 1, 2, 1, 1, 0, 1, 2, 0, 1] * 8
    k = kappa(a, b, "quadratic")
    lo, hi = interval(a, b, "quadratic")
    assert lo <= k <= hi and (lo, hi) == interval(a, b, "quadratic")


def test_markdown_shows_the_table_and_the_small_sample_warning():
    ag = compare({"q": {"a": 2, "b": 1, "c": 0}}, {"q": {"a": 2, "b": 0, "c": 0}})
    md = markdown(ag, "sam", "alex")
    assert "3 documents graded by both" in md and "| 2 | 0 | 0 | 1 |" in md
    assert "`q` `b`: sam 1, alex 0" in md and "interval is wide" in md


def cli(*args):
    return subprocess.run([sys.executable, "-m", "relevance_gate", *args], capture_output=True, text=True)


def test_cli_agree(tmp_path):
    a, b, out = tmp_path / "sam.tsv", tmp_path / "alex.tsv", tmp_path / "settle.tsv"
    a.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td3\t1\nq2\td4\t2\n")
    b.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq2\td3\t1\nq2\td4\t0\n")
    r = cli("agree", str(a), str(b), "--disagreements", str(out))
    assert r.returncode == 0, r.stderr
    assert "Judge agreement: sam and alex" in r.stdout and "weighted kappa" in r.stdout
    assert out.read_text().splitlines()[1] == "q2\td4\t2\t0"  # the biggest gap first
    assert cli("agree", str(a), str(b), "--min-kappa", "0.9").returncode == 1
    other = tmp_path / "other.tsv"
    other.write_text("q9\td9\t1\n")
    r = cli("agree", str(a), str(other))
    assert r.returncode == 2 and "share no graded documents" in r.stderr
