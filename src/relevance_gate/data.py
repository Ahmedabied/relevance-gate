"""Judged queries and documents, in the BEIR layout.

A dataset folder holds:
    corpus.jsonl      {"_id": "d1", "title": "...", "text": "..."}   (only the built-in engine needs it)
    queries.jsonl     {"_id": "q1", "text": "..."}
    qrels/test.tsv    query-id <tab> corpus-id <tab> score            (score 0 means judged not relevant)

The same layout is used by the BEIR benchmark, so its public datasets load as they are.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


class DataError(ValueError):
    pass


@dataclass
class Dataset:
    name: str
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    docs: dict[str, tuple[str, str]] = field(default_factory=dict)


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as e:
                    raise DataError(f"{path}:{n}: not JSON: {e}") from None


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or (n == 1 and not parts[2].lstrip("-").isdigit()):
                continue  # header or blank
            qid, doc, score = parts[0], parts[1], parts[-1]
            try:
                qrels.setdefault(qid, {})[doc] = int(score)
            except ValueError:
                raise DataError(f"{path}:{n}: score must be an integer, got {score!r}") from None
    return qrels


def load(path: str | Path, split: str = "test", with_corpus: bool = True) -> Dataset:
    root = Path(path)
    qrels_path = root / "qrels" / f"{split}.tsv"
    for p in (root / "queries.jsonl", qrels_path):
        if not p.exists():
            raise DataError(f"{root}: missing {p.relative_to(root)} (expected the BEIR layout, see the README)")
    qrels = load_qrels(qrels_path)
    queries = {}
    for q in _jsonl(root / "queries.jsonl"):
        qid = str(q.get("_id"))
        if qid in qrels:
            queries[qid] = str(q.get("text", ""))
    missing = set(qrels) - set(queries)
    if missing:
        raise DataError(f"{root}: {len(missing)} judged queries have no text in queries.jsonl, e.g. {sorted(missing)[:3]}")
    docs = {}
    if with_corpus:
        corpus = root / "corpus.jsonl"
        if not corpus.exists():
            raise DataError(f"{root}: missing corpus.jsonl (the built-in fts5 engine indexes it)")
        for d in _jsonl(corpus):
            docs[str(d.get("_id"))] = (str(d.get("title") or ""), str(d.get("text") or ""))
    return Dataset(root.name, queries, {q: qrels[q] for q in queries}, docs)


# Public BEIR datasets, pinned by hash. The data is downloaded, never bundled: check each
# dataset's own licence before you use it (SciFact, for example, is CC BY-NC 2.0).
KNOWN = {
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
}
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "relevance-gate"


def fetch(name: str, dest: str | Path | None = None) -> Path:
    """Download a known BEIR dataset, check its hash, unpack it. Returns the dataset folder."""
    if name not in KNOWN:
        raise DataError(f"unknown dataset {name!r}; known: {', '.join(sorted(KNOWN))}")
    dest = Path(dest) if dest else cache_dir()
    target = dest / name
    if (target / "queries.jsonl").exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        zpath = Path(tmp) / f"{name}.zip"
        with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if digest != KNOWN[name]:
            raise DataError(f"{name}: download hash {digest} does not match the pinned {KNOWN[name]}")
        out = Path(tmp) / "out"
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():  # refuse paths that would land outside the folder
                p = (out / info.filename).resolve()
                if out.resolve() not in p.parents and p != out.resolve():
                    raise DataError(f"{name}: unsafe path in archive: {info.filename}")
            z.extractall(out)
        shutil.move(str(out / name), str(target))
    return target
