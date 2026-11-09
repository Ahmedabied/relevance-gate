import io
import json
import subprocess
import sys
from collections import Counter

from relevance_gate.data import load, load_qrels
from relevance_gate.judging import judge, normalise, query_id, read_log, sample, strata


def zipf_log():
    # 200 distinct queries; query i is searched about 1000 / (i + 1) times.
    return Counter({f"query number {i}": max(1, 1000 // (i + 1)) for i in range(200)})


def test_strata_split_traffic_in_thirds():
    s = strata(zipf_log())
    assert len(s["head"]) < len(s["torso"]) < len(s["tail"])  # few head queries carry a third of traffic
    assert s["head"][0] == "query number 0"


def test_sample_is_even_and_deterministic():
    picked, info = sample(zipf_log(), 30, seed=7)
    assert len(picked) == 30 and info["picked"] == 30
    by = Counter(q["stratum"] for q in picked)
    head = len(strata(zipf_log())["head"])
    assert by["head"] == min(10, head) and by["tail"] == 10  # a short head hands its spare share to the torso
    assert by["torso"] == 30 - by["head"] - by["tail"]
    assert picked == sample(zipf_log(), 30, seed=7)[0]
    assert len({q["_id"] for q in picked}) == 30


def test_short_stratum_hands_over_its_share():
    log = Counter({"a": 1000, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1})
    picked, _ = sample(log, 6)
    assert len(picked) == 6  # head has only "a", the rest comes from the other strata


def test_pii_is_dropped_by_default():
    log = Counter({"blue shoes": 5, "ahmed@example.com order": 2, "call +44 20 7946 0000": 1, "order 12345678": 1})
    picked, info = sample(log, 10)
    assert [q["text"] for q in picked] == ["blue shoes"] and info["dropped_pii"] == 3
    assert len(sample(log, 10, keep_pii=True)[0]) == 4


def test_read_log_formats(tmp_path):
    (tmp_path / "a.txt").write_text("Blue  Shoes\nblue shoes\n\nred hat\n")
    (tmp_path / "b.csv").write_text("query,count\nblue shoes,3\nred hat,1\n")
    (tmp_path / "c.jsonl").write_text('{"query": "Blue shoes", "count": 2}\n{"query": "red hat"}\n')
    assert read_log(tmp_path / "a.txt") == {"blue shoes": 2, "red hat": 1}
    assert read_log(tmp_path / "b.csv") == {"blue shoes": 3, "red hat": 1}
    assert read_log(tmp_path / "c.jsonl") == {"blue shoes": 2, "red hat": 1}
    assert normalise("  A  b ") == "a b" and query_id("a b") == query_id("a b")


def _setup(tmp_path):
    root = tmp_path / "mine"
    root.mkdir()
    (root / "queries.jsonl").write_text('{"_id": "q1", "text": "shoes"}\n{"_id": "q2", "text": "hats"}\n')
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "title": t, "text": ""}) + "\n"
                                               for d, t in [("1", "shoes"), ("2", "hats"), ("3", "shoe rack")]))
    return root


def test_judge_writes_and_resumes(tmp_path):
    root = _setup(tmp_path)
    out = root / "qrels" / "test.tsv"
    pools = [lambda q, k: ["1", "3"] if q == "shoes" else ["2"], lambda q, k: ["3", "2"] if q == "shoes" else []]
    shown = io.StringIO()
    stats = judge(root, pools, 10, out, io.StringIO("2\nbanana\n1\nq\n"), shown)
    assert stats["judged"] == 2 and "please type" in shown.getvalue()
    first = load_qrels(out)["q1"]
    assert set(first) <= {"1", "2", "3"} and len(first) == 2
    # Rerun: only what is left is asked.
    stats = judge(root, pools, 10, out, io.StringIO("0\n2\n"), io.StringIO())
    q = load_qrels(out)
    assert stats["judged"] == 2 and len(q["q1"]) == 3 and q["q2"] == {"2": 2}
    assert out.read_text().count("query-id") == 1
    ds = load(root)  # the result is a dataset eval can run on
    assert set(ds.queries) == {"q1", "q2"}


def test_cli_sample_and_judge(tmp_path):
    root = _setup(tmp_path)
    log = tmp_path / "log.txt"
    log.write_text("shoes\nshoes\nhats\nme@example.com\n")
    r = subprocess.run([sys.executable, "-m", "relevance_gate", "sample", str(log), "-n", "5", "-o",
                        str(tmp_path / "s.jsonl")], capture_output=True, text=True)
    assert r.returncode == 0 and "dropped 1" in r.stdout
    r = subprocess.run([sys.executable, "-m", "relevance_gate", "judge", "--data", str(root), "--pool", "",
                        "--pool", "tokenizer=unicode61"], input="2\n2\n2\n2\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "grades saved" in r.stdout and load_qrels(root / "qrels" / "test.tsv")["q1"]["1"] == 2


def test_terminal_output_is_sanitised(tmp_path):
    from relevance_gate.judging import safe

    assert safe("ok\x1b[2J\u202etxt") == "ok\\u001b[2J\\u202etxt"
    assert safe("a\nb", keep_newlines=True) == "a\nb" and safe("a\nb") == "a\\u000ab"
    root = tmp_path / "evil"
    root.mkdir()
    (root / "queries.jsonl").write_text(json.dumps({"_id": "q", "text": "x\x1b]0;pwned\x07"}) + "\n")
    (root / "corpus.jsonl").write_text(json.dumps({"_id": "d", "title": "\x1b[31mred", "text": "b"}) + "\n")
    shown = io.StringIO()
    judge(root, [lambda q, k: ["d"]], 10, root / "qrels" / "test.tsv", io.StringIO("1\n"), shown)
    assert "\x1b" not in shown.getvalue() and "\x07" not in shown.getvalue()


def test_pool_out_writes_what_is_left_for_judge_ui(tmp_path):
    root = _setup(tmp_path)
    out = root / "qrels" / "test.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("query-id\tcorpus-id\tscore\nq1\t1\t2\n")
    pool = tmp_path / "pool.jsonl"
    r = subprocess.run([sys.executable, "-m", "relevance_gate", "judge", "--data", str(root), "--pool", "",
                        "--pool-out", str(pool)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in pool.read_text().splitlines()]
    assert rows and all(set(x) == {"query_id", "query", "doc_id", "title", "text"} for x in rows)
    assert ("q1", "1") not in {(x["query_id"], x["doc_id"]) for x in rows}  # already judged
    assert any(x["doc_id"] == "3" and x["title"] == "shoe rack" for x in rows)
    assert "judge-ui" in r.stdout
