import hashlib
import io
import zipfile

import pytest

from relevance_gate import data
from relevance_gate.data import DataError, load


def test_load_beir_layout(toy):
    ds = load(toy)
    assert len(ds.queries) == 24 and "unjudged" not in ds.queries  # only judged queries count
    assert ds.qrels["q-volcano-0"]["volcano-0"] == 2
    assert len(ds.docs) == 48
    assert not load(toy, with_corpus=False).docs


def test_missing_files_and_text(toy, tmp_path):
    with pytest.raises(DataError, match="missing"):
        load(tmp_path / "nope")
    (toy / "queries.jsonl").write_text('{"_id": "q-volcano-0", "text": "x"}\n')
    with pytest.raises(DataError, match="no text"):
        load(toy)


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in entries.items():
            z.writestr(name, text)
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_fetch_checks_hash_and_unpacks(tmp_path, monkeypatch):
    blob = _zip({"demo/queries.jsonl": '{"_id": "1", "text": "q"}\n', "demo/qrels/test.tsv": "1\td\t1\n",
                 "demo/corpus.jsonl": '{"_id": "d", "title": "", "text": "t"}\n'})
    monkeypatch.setattr(data, "KNOWN", {"demo": hashlib.sha256(blob).hexdigest()})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: _Resp(blob))
    path = data.fetch("demo", tmp_path)
    assert load(path).queries == {"1": "q"}
    assert data.fetch("demo", tmp_path) == path  # cached

    monkeypatch.setattr(data, "KNOWN", {"demo2": "0" * 64})
    with pytest.raises(DataError, match="does not match"):
        data.fetch("demo2", tmp_path)
    with pytest.raises(DataError, match="unknown dataset"):
        data.fetch("nope", tmp_path)


def test_fetch_refuses_paths_outside_the_folder(tmp_path, monkeypatch):
    blob = _zip({"evil/queries.jsonl": "", "../escape.txt": "x"})
    monkeypatch.setattr(data, "KNOWN", {"evil": hashlib.sha256(blob).hexdigest()})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: _Resp(blob))
    with pytest.raises(DataError, match="unsafe path"):
        data.fetch("evil", tmp_path / "d")
    assert not (tmp_path / "escape.txt").exists()
