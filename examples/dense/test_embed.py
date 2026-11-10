"""embed.py with a fake encoder: the run file is right without downloading a model."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

spec = importlib.util.spec_from_file_location("embed", Path(__file__).with_name("embed.py"))
embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed)

WORDS = ["kidney", "stone", "vitamin", "sleep", "heart", "cancer", "bone", "zinc"]


def fake_encode(texts):
    """Bag of known words, normalized. Deterministic."""
    vecs = np.array([[t.lower().count(w) for w in WORDS] for t in texts], dtype=np.float32) + 1e-3
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "toy"
    (root / "qrels").mkdir(parents=True)
    docs = [("d1", "Kidney", "kidney stone kidney"), ("d2", "Sleep", "sleep and the heart"),
            ("d3", "", "zinc and bone"), ("d4", "Vitamin D", "vitamin and bone")]
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": i, "title": t, "text": x}) + "\n" for i, t, x in docs))
    qs = [("q1", "kidney stone"), ("q2", "bone vitamin"), ("q9", "never judged")]
    (root / "queries.jsonl").write_text("".join(json.dumps({"_id": i, "text": t}) + "\n" for i, t in qs))
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td4\t2\n")
    return root


def read(path):
    return [line.split() for line in gzip.open(path, "rt", encoding="utf-8")]


def test_run_file_ranks_judged_queries_best_first(data, tmp_path):
    out = tmp_path / "toy.run.gz"
    assert embed.main(["--data", str(data), "-o", str(out), "-k", "3", "--meta", str(tmp_path / "m.json")],
                      encoder=fake_encode) == 0
    rows = read(out)
    assert {r[0] for r in rows} == {"q1", "q2"}  # only judged queries
    for qid in ("q1", "q2"):
        mine = [r for r in rows if r[0] == qid]
        assert [r[3] for r in mine] == ["1", "2", "3"]
        scores = [float(r[4]) for r in mine]
        assert scores == sorted(scores, reverse=True)
        assert all(r[1] == "Q0" and r[5] == embed.TAG for r in mine)
    assert [r[2] for r in rows if r[0] == "q1"][0] == "d1"
    assert [r[2] for r in rows if r[0] == "q2"][0] == "d4"
    meta = json.loads((tmp_path / "m.json").read_text())
    assert meta["queries"] == 2 and meta["docs"] == 4 and meta["model"] == embed.MODEL
    assert meta["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_same_run_gives_the_same_bytes(data, tmp_path):
    a, b = tmp_path / "a.run.gz", tmp_path / "b.run.gz"
    embed.main(["--data", str(data), "-o", str(a)], encoder=fake_encode)
    embed.main(["--data", str(data), "-o", str(b)], encoder=fake_encode)
    assert a.read_bytes() == b.read_bytes()


def test_k_larger_than_the_corpus_keeps_every_document_once(data, tmp_path):
    out = tmp_path / "all.run"
    embed.main(["--data", str(data), "-o", str(out), "-k", "100"], encoder=fake_encode)
    rows = [line.split() for line in out.read_text().splitlines()]
    for qid in ("q1", "q2"):
        ids = [r[2] for r in rows if r[0] == qid]
        assert sorted(ids) == ["d1", "d2", "d3", "d4"]


def test_ties_go_to_the_lower_document_index():
    docs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ranked = embed.top_k(docs, np.array([[1.0, 0.0]], dtype=np.float32), 2)
    assert [d for d, _ in ranked[0]] == [0, 1]


def test_doc_cache_embeds_the_documents_once(data, tmp_path):
    seen = []

    def counting(texts):
        seen.append(len(texts))
        return fake_encode(texts)

    cache = tmp_path / "docs.npy"
    embed.main(["--data", str(data), "-o", str(tmp_path / "a.run"), "--doc-cache", str(cache)], encoder=counting)
    assert seen == [4, 2] and cache.exists()  # documents, then queries
    seen.clear()
    embed.main(["--data", str(data), "-o", str(tmp_path / "b.run"), "--doc-cache", str(cache)], encoder=counting)
    assert seen == [2]  # queries only
    assert (tmp_path / "a.run").read_bytes() == (tmp_path / "b.run").read_bytes()


def test_doc_cache_from_another_corpus_is_refused(data, tmp_path):
    cache = tmp_path / "docs.npy"
    embed.main(["--data", str(data), "-o", str(tmp_path / "a.run"), "--doc-cache", str(cache)], encoder=fake_encode)
    with open(data / "corpus.jsonl", "a") as f:
        f.write(json.dumps({"_id": "d5", "title": "", "text": "new"}) + "\n")
    with pytest.raises(SystemExit, match="another corpus"):
        embed.main(["--data", str(data), "-o", str(tmp_path / "b.run"), "--doc-cache", str(cache)], encoder=fake_encode)


def test_train_split_ranks_its_own_queries(data, tmp_path):
    (data / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq9\td2\t1\n")
    out = tmp_path / "train.run"
    embed.main(["--data", str(data), "--split", "train", "-o", str(out)], encoder=fake_encode)
    assert {line.split()[0] for line in out.read_text().splitlines()} == {"q9"}
