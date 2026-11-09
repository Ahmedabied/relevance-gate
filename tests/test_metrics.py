import math

import pytest

from relevance_gate import metrics as M

J = {"a": 1, "c": 2, "z": 0}


def test_ndcg_by_hand():
    ideal = 2 + 1 / math.log2(3)
    assert M.ndcg(["a", "b", "c"], J, 3) == pytest.approx((1 + 2 / math.log2(4)) / ideal)
    assert M.ndcg(["c", "a"], J, 10) == pytest.approx(1.0)
    assert M.ndcg(["b", "z"], J, 10) == 0.0
    assert M.ndcg(["a"], {"a": 0}, 10) == 0.0  # nothing relevant to find


def test_duplicates_count_once():
    once = M.score(["a", "c"], J, ["ndcg@3"])["ndcg@3"]
    assert M.score(["a", "a", "c"], J, ["ndcg@3"])["ndcg@3"] == pytest.approx(once)
    assert M.score(["a", "a", "a"], J, ["p@3"])["p@3"] == pytest.approx(1 / 3)


def test_other_metrics():
    assert M.precision(["a", "b"], J, 2) == 0.5
    assert M.recall(["a", "b", "c"], J, 3) == 1.0
    assert M.recall(["a"], J, 3) == 0.5
    assert M.mrr(["b", "c"], J, 10) == 0.5
    assert M.mrr(["b"], J, 10) == 0.0
    assert M.hit(["b", "a"], J, 1) == 0.0 and M.hit(["b", "a"], J, 2) == 1.0
    assert M.zero_results([], J, 0) == 1.0 and M.zero_results(["b"], J, 0) == 0.0


def test_parse_and_direction():
    assert M.parse("ndcg@10")[1] == 10
    assert not M.higher_is_better("zero_results") and M.higher_is_better("recall@100")
    assert M.depth_needed(["ndcg@10", "recall@100"]) == 100
    for bad in ["ndcg", "ndcg@0", "map@10", "zero"]:
        with pytest.raises(ValueError):
            M.parse(bad)
