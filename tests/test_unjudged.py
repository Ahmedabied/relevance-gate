import io
import json

from relevance_gate import metrics as M
from relevance_gate.cli import main
from relevance_gate.data import load_qrels
from relevance_gate.gate import compare, markdown
from relevance_gate.judging import write_pool


def test_unjudged_counts_documents_nobody_graded():
    judged = {"a": 2, "b": 0}
    assert M.score(["a", "x", "b", "y"], judged, ["unjudged@4"])["unjudged@4"] == 0.5
    assert M.score(["a", "b"], judged, ["unjudged@10"])["unjudged@10"] == 0.0  # a grade of 0 is judged
    assert M.score(["x"], judged, ["unjudged@10"])["unjudged@10"] == 0.1  # empty places are not unjudged documents
    assert M.score([], judged, ["unjudged@10"])["unjudged@10"] == 0.0
    assert M.score(["x", "x"], judged, ["unjudged@2"])["unjudged@2"] == 0.5  # a repeat counts once
    assert M.score(["a", "x", "y"], judged, ["unjudged@1"])["unjudged@1"] == 0.0  # only the top K count


def test_unjudged_is_lower_is_better_and_in_the_default_report():
    assert not M.higher_is_better("unjudged@10") and not M.higher_is_better("zero_results")
    assert M.higher_is_better("ndcg@10")
    assert "unjudged@10" in M.DEFAULT


def report(ndcg, unjudged):
    return {"version": 1, "per_query": {f"q{i}": {"ndcg@10": n, "unjudged@10": u}
                                        for i, (n, u) in enumerate(zip(ndcg, unjudged))}}


def test_gate_notes_a_candidate_that_shows_more_unjudged_documents():
    base, cand = report([0.5] * 60, [0.1] * 60), report([0.5] * 60, [0.4] * 60)
    checks, info = compare(base, cand, {"ndcg@10": 0.005})
    md = markdown(checks, info, base, cand, 0.05)
    assert "40% of the candidate's top 10 places hold documents nobody graded, against 10% for the baseline" in md
    assert "lower bound" in md and "--pool-run" in md


def test_gate_says_nothing_when_unjudged_is_level_or_missing():
    base, cand = report([0.5] * 60, [0.1] * 60), report([0.5] * 60, [0.13] * 60)
    checks, info = compare(base, cand, {"ndcg@10": 0.005})
    assert "nobody graded" not in markdown(checks, info, base, cand, 0.05)
    old = {"version": 1, "per_query": {f"q{i}": {"ndcg@10": 0.5} for i in range(60)}}  # a report from 0.2
    checks, info = compare(old, cand, {"ndcg@10": 0.005})
    assert "nobody graded" not in markdown(checks, info, old, cand, 0.05)


def _data(tmp_path):
    root = tmp_path / "shop"
    (root / "qrels").mkdir(parents=True)
    (root / "queries.jsonl").write_text('{"_id": "q1", "text": "shoes"}\n{"_id": "q2", "text": "hats"}\n')
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "title": t, "text": ""}) + "\n"
                                               for d, t in [("1", "shoes"), ("2", "hats"), ("3", "shoe rack"),
                                                            ("9", "boots")]))
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\t1\t2\n")
    return root


def test_judge_pools_a_run_file_by_query_id(tmp_path, capsys):
    root = _data(tmp_path)
    run = tmp_path / "vectors.run"
    run.write_text("q1 Q0 9 1 0.9 v\nq1 Q0 1 2 0.8 v\nq2 Q0 2 1 0.7 v\n")
    pool = tmp_path / "pool.jsonl"
    assert main(["judge", "--data", str(root), "--pool-run", str(run), "--pool-out", str(pool)]) == 0
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    got = {(r["query_id"], r["doc_id"]) for r in rows}
    assert ("q1", "9") in got and ("q2", "2") in got  # the run's documents are in the pool
    assert ("q1", "1") not in got  # already graded
    assert ("q1", "3") in got  # the fts5 pool is still there, so one engine does not shape the set


def test_judge_pool_run_alone_skips_the_fts5_pool(tmp_path):
    root = _data(tmp_path)
    run = tmp_path / "vectors.run"
    run.write_text("q1 Q0 9 1 0.9 v\n")
    pool = tmp_path / "pool.jsonl"
    assert main(["judge", "--data", str(root), "--pool-run", str(run), "--pool", "none", "--pool-out", str(pool)]) == 0
    assert [(r["query_id"], r["doc_id"]) for r in map(json.loads, pool.read_text().splitlines())] == [("q1", "9")]


def test_judge_asks_for_run_documents_in_the_terminal(tmp_path):
    root = _data(tmp_path)
    run = tmp_path / "vectors.run"
    run.write_text("q2 Q0 9 1 0.9 v\n")
    out = root / "qrels" / "test.tsv"
    from relevance_gate.cli import _run_searcher
    from relevance_gate.judging import judge

    stats = judge(root, [_run_searcher(run, root)], 10, out, io.StringIO("1\n"), io.StringIO())
    assert stats["judged"] == 1 and load_qrels(out)["q2"] == {"9": 1}
    stats = write_pool(root, [_run_searcher(run, root)], 10, out, tmp_path / "left.jsonl")
    assert stats["documents"] == 0


def test_pool_none_with_other_pools_is_refused_plainly(tmp_path, capsys):
    root = _data(tmp_path)
    assert main(["judge", "--data", str(root), "--pool", "none", "--pool", "mode=and",
                 "--pool-out", str(tmp_path / "p.jsonl")]) == 2
    assert "--pool none cannot be combined" in capsys.readouterr().err


def test_no_corpus_points_at_pool_run(tmp_path, capsys):
    root = _data(tmp_path)
    (root / "corpus.jsonl").unlink()
    run = tmp_path / "vectors.run"
    run.write_text("q1 Q0 9 1 0.9 v\n")
    assert main(["judge", "--data", str(root), "--pool-run", str(run), "--pool-out", str(tmp_path / "p.jsonl")]) == 2
    assert "--pool none" in capsys.readouterr().err
    assert main(["judge", "--data", str(root), "--pool-run", str(run), "--pool", "none",
                 "--pool-out", str(tmp_path / "p.jsonl")]) == 0
