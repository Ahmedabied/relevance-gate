import gzip
import json

import pytest

from relevance_gate.cli import main
from relevance_gate.data import DataError, load
from relevance_gate.engines import EngineError, FusedEngine, Fts5Engine, RunEngine
from relevance_gate.gate import evaluate
from relevance_gate.runs import load_run, rrf, write_run


def trec(rows):
    return "".join(" ".join(str(x) for x in r) + "\n" for r in rows)


def test_run_is_ordered_by_score_not_by_line(tmp_path):
    p = tmp_path / "a.run"
    p.write_text(trec([("q1", "Q0", "d3", 3, 0.1, "x"), ("q1", "Q0", "d1", 1, 0.9, "x"),
                       ("q1", "Q0", "d2", 2, 0.5, "x"), ("q2", "Q0", "d9", 1, 2.0, "x")]))
    run, sha = load_run(p)
    assert run == {"q1": ["d1", "d2", "d3"], "q2": ["d9"]}
    assert len(sha) == 64


def test_equal_scores_keep_the_files_rank(tmp_path):
    p = tmp_path / "a.run"
    p.write_text(trec([("q1", "Q0", "b", 2, 1.0, "x"), ("q1", "Q0", "a", 1, 1.0, "x")]))
    assert load_run(p)[0]["q1"] == ["a", "b"]


def test_a_document_listed_twice_keeps_its_best_place(tmp_path):
    p = tmp_path / "a.run"
    p.write_text(trec([("q1", "Q0", "a", 1, 0.9, "x"), ("q1", "Q0", "b", 2, 0.5, "x"), ("q1", "Q0", "b", 3, 0.95, "x")]))
    assert load_run(p)[0]["q1"] == ["b", "a"]


def test_gzip_is_read_by_its_bytes_not_its_name(tmp_path):
    p = tmp_path / "plain-name.run"
    p.write_bytes(gzip.compress(trec([("q1", "Q0", "a", 1, 1, "x")]).encode()))
    assert load_run(p)[0] == {"q1": ["a"]}


@pytest.mark.parametrize("line, why", [
    ("q1 Q0 a 1 0.5", "6 fields"),
    ("q1 Q0 a one 0.5 x", "rank"),
    ("q1 Q0 a 1 high x", "score"),
    ("q1 Q0 a 1 nan x", "score"),
    ("q1 Q0 a 1 inf x", "score"),
])
def test_bad_lines_are_refused_with_their_line_number(tmp_path, line, why):
    p = tmp_path / "bad.run"
    p.write_text("q1 Q0 ok 1 1.0 x\n" + line + "\n")
    with pytest.raises(DataError, match=rf"bad.run:2: .*{why}"):
        load_run(p)


def test_an_empty_run_is_refused(tmp_path):
    p = tmp_path / "empty.run"
    p.write_text("\n")
    with pytest.raises(DataError, match="no results"):
        load_run(p)


def test_rrf_by_hand():
    # a: 1/61 + 1/62, c: 1/63 + 1/61, b: 1/62, d: 1/63
    assert rrf([["a", "b", "c"], ["c", "a", "d"]]) == ["a", "c", "b", "d"]


def test_rrf_weight_moves_the_order():
    assert rrf([["a", "b"], ["b", "a"]], [1.0, 2.0]) == ["b", "a"]
    assert rrf([["a", "b"], ["b", "a"]], [2.0, 1.0]) == ["a", "b"]


def test_rrf_weight_zero_is_the_first_list_unchanged():
    assert rrf([["a", "b", "c"], ["z", "c", "a"]], [1.0, 0.0]) == ["a", "b", "c"]


def test_rrf_ties_go_to_the_best_rank_then_the_id():
    assert rrf([["b", "x"], ["a", "y"]]) == ["a", "b", "x", "y"]


def test_rrf_equal_sums_go_to_the_better_best_rank_before_the_id():
    # 'z' gets 1/61 at rank 1; 'a' gets 2/122, the same number, at rank 62
    fillers = [f"f{i:02d}" for i in range(61)]
    fused = rrf([["z"], fillers + ["a"]], [1.0, 2.0])
    assert fused.index("z") < fused.index("a")


def test_rrf_counts_a_repeated_document_once_at_its_first_place():
    # without the dedupe 'a' would collect 1/61 + 1/62 and win, and 'b' would count from rank 3
    assert rrf([["a", "a", "b"], ["b"]]) == ["b", "a"]


def test_rrf_depth_cuts_each_list_before_fusing():
    assert rrf([["a", "b", "c"], ["c"]], depth=2) == ["a", "c", "b"]
    assert rrf([["a", "b", "c"]], depth=2) == ["a", "b"]


@pytest.mark.parametrize("kw", [{"k": 0}, {"k": float("nan")}, {"weights": [1.0, -1.0]}, {"weights": [1.0]}])
def test_rrf_refuses_bad_settings(kw):
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], **kw)


def test_run_engine_scores_the_file_and_counts_missing_queries(toy, tmp_path):
    ds = load(toy, with_corpus=False)
    qid = sorted(ds.qrels)[0]
    best = sorted(ds.qrels[qid], key=lambda d: -ds.qrels[qid][d])
    p = tmp_path / "one.run"
    p.write_text(trec([(qid, "Q0", d, i, 10 - i, "x") for i, d in enumerate(best, 1)]))
    eng = RunEngine(p)
    report = evaluate(eng, ds, ["ndcg@10", "zero_results"])
    assert report["per_query"][qid]["ndcg@10"] == 1.0
    assert eng.missing == len(ds.queries) - 1
    assert report["latency_ms"] == {}  # a file has no latency to report
    with pytest.raises(EngineError):
        eng.search("text", 10)


def test_a_run_with_none_of_the_judged_queries_is_refused(toy, tmp_path):
    p = tmp_path / "other.run"
    p.write_text(trec([("not-a-query", "Q0", "d", 1, 1.0, "x")]))
    with pytest.raises(EngineError, match="none of the"):
        evaluate(RunEngine(p), load(toy, with_corpus=False))


def test_fusing_with_weight_zero_changes_nothing(toy, tmp_path):
    ds = load(toy)
    p = tmp_path / "noise.run"
    p.write_text(trec([(q, "Q0", "sailing-0", 1, 1.0, "x") for q in ds.queries]))
    plain = evaluate(Fts5Engine(ds.docs), ds)
    fused = evaluate(FusedEngine(Fts5Engine(ds.docs), [RunEngine(p)], weight=0.0), ds)
    assert fused["metrics"] == plain["metrics"]


def test_fusion_lifts_a_document_both_lists_agree_on(toy, tmp_path):
    ds = load(toy)
    fts = Fts5Engine(ds.docs)
    qid = sorted(ds.queries)[0]
    base = fts.search_all({qid: ds.queries[qid]}, 100)[0][qid]
    pick = base[5]
    p = tmp_path / "agree.run"
    p.write_text(trec([(qid, "Q0", pick, 1, 1.0, "x")]))
    fused = FusedEngine(fts, [RunEngine(p)])
    ranked = fused.search_all({qid: ds.queries[qid]}, 10)[0][qid]
    assert ranked[0] == pick and sorted(ranked) == sorted(base[:10])
    d = fused.describe()
    assert d["name"] == "fts5" and d["fused"][0]["file"] == "agree.run" and d["rrf_k"] == 60.0


def test_cli_run_out_round_trips(toy, tmp_path, capsys):
    out = tmp_path / "fts5.run.gz"
    assert main(["eval", "--data", str(toy), "-o", str(tmp_path / "a.json"), "--run-out", str(out)]) == 0
    assert main(["eval", "--data", str(toy), "--engine", "run", "--run", str(out), "-o", str(tmp_path / "b.json")]) == 0
    a = json.loads((tmp_path / "a.json").read_text())
    b = json.loads((tmp_path / "b.json").read_text())
    assert a["metrics"] == b["metrics"] and a["per_query"] == b["per_query"]
    assert b["engine"]["name"] == "run" and b["engine"]["file"] == "fts5.run.gz"


def test_cli_fuse(toy, tmp_path, capsys):
    run = tmp_path / "fts5.run"
    assert main(["eval", "--data", str(toy), "--run-out", str(run)]) == 0
    assert main(["eval", "--data", str(toy), "--fuse", str(run), "--fuse-weight", "0.5", "--rrf-k", "10",
                 "-o", str(tmp_path / "f.json")]) == 0
    rep = json.loads((tmp_path / "f.json").read_text())
    assert rep["engine"]["fused"][0]["weight"] == 0.5 and rep["engine"]["rrf_k"] == 10.0
    assert "fused with fts5.run" in capsys.readouterr().err


def test_cli_run_engine_needs_a_file(toy, capsys):
    assert main(["eval", "--data", str(toy), "--engine", "run"]) == 2
    assert "--run" in capsys.readouterr().err


def test_cli_warns_when_a_run_misses_queries(toy, tmp_path, capsys):
    ds = load(toy, with_corpus=False)
    qid = sorted(ds.queries)[0]
    p = tmp_path / "part.run"
    p.write_text(trec([(qid, "Q0", "x", 1, 1.0, "x")]))
    assert main(["eval", "--data", str(toy), "--fuse", str(p)]) == 0
    assert f"part.run has no results for {len(ds.queries) - 1} of {len(ds.queries)} judged queries" in capsys.readouterr().err


def test_write_run_scores_fall_with_rank(tmp_path):
    p = tmp_path / "w.run"
    write_run({"q2": ["b"], "q1": ["a", "c"]}, p, "t")
    assert p.read_text() == "q1 Q0 a 1 2 t\nq1 Q0 c 2 1 t\nq2 Q0 b 1 1 t\n"


def test_a_fused_run_out_says_it_was_fused(toy, tmp_path):
    run = tmp_path / "fts5.run"
    assert main(["eval", "--data", str(toy), "--run-out", str(run)]) == 0
    fused = tmp_path / "fused.run"
    assert main(["eval", "--data", str(toy), "--fuse", str(run), "--run-out", str(fused)]) == 0
    assert run.read_text().split("\n", 1)[0].endswith(" fts5")
    assert fused.read_text().split("\n", 1)[0].endswith(" fts5+rrf")
