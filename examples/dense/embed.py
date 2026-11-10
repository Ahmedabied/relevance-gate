"""Make a dense retrieval run for a BEIR dataset with an open embedding model.

It downloads the model (about 130 MB) and embeds every document, so it runs in GitHub Actions
(.github/workflows/dense.yml). The output is a TREC run file, ids and scores only, which
`relevance-gate eval --engine run` scores and `--fuse` mixes with keyword search.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers numpy
    python examples/dense/embed.py --data ~/.cache/relevance-gate/scifact -o scifact.run.gz --meta scifact.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

MODEL = "BAAI/bge-small-en-v1.5"  # MIT licence, 33M parameters, 384 dimensions
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned so the run can be made again
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # the model card's retrieval prompt
TAG = "bge-small-en-v1.5"


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load(data: Path, split: str = "test"):
    """Judged queries and every document, the way relevance-gate reads them."""
    judged = set()
    with open(data / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and not (n == 1 and not parts[2].lstrip("-").isdigit()):
                judged.add(parts[0])
    queries = {str(q["_id"]): str(q.get("text") or "") for q in _jsonl(data / "queries.jsonl") if str(q["_id"]) in judged}
    doc_ids, doc_texts = [], []
    for d in _jsonl(data / "corpus.jsonl"):
        doc_ids.append(str(d["_id"]))
        doc_texts.append((str(d.get("title") or "") + " " + str(d.get("text") or "")).strip())
    qids = sorted(queries)
    return doc_ids, doc_texts, qids, [queries[q] for q in qids]


def top_k(doc_vecs, query_vecs, k: int):
    """Best k documents per query by dot product (cosine on normalized vectors). Ties go to the lower index."""
    import numpy as np

    scores = query_vecs @ doc_vecs.T
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = []
    for row in range(scores.shape[0]):
        cand = part[row]
        s = scores[row, cand]
        order = np.lexsort((cand, -s))
        out.append([(int(cand[i]), float(s[i])) for i in order])
    return out


def run_text(qids, ranked, doc_ids, tag: str = TAG) -> str:
    lines = []
    for qid, hits in zip(qids, ranked):
        for rank, (d, score) in enumerate(hits, 1):
            lines.append(f"{qid} Q0 {doc_ids[d]} {rank} {score:.6f} {tag}\n")
    return "".join(lines)


def write(path: Path, text: str) -> str:
    """Write the run (gzip with no timestamp, so the same run gives the same bytes). Returns its sha256."""
    data = text.encode("utf-8")
    if path.suffix == ".gz":
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
            g.write(data)
        data = buf.getvalue()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def cached_docs(path: str, corpus_sha: str, n: int, model: str = MODEL, revision: str = REVISION):
    """The cached document vectors, or None when there is no cache yet. A cache made from another corpus
    (or another model) is refused, never silently used."""
    import numpy as np

    vecs, meta = Path(path), Path(path + ".json")
    if not vecs.exists():
        return None
    info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    if info.get("corpus_sha256") != corpus_sha or info.get("model") != model or info.get("revision") != revision:
        raise SystemExit(f"{path} was made from another corpus or model; delete it or pass another --doc-cache")
    out = np.load(vecs, allow_pickle=False)
    if out.shape[0] != n:
        raise SystemExit(f"{path} holds {out.shape[0]} vectors for {n} documents")
    return out


def save_docs(path: str, corpus_sha: str, vecs, model: str = MODEL, revision: str = REVISION) -> None:
    import numpy as np

    with open(path, "wb") as f:
        np.save(f, vecs, allow_pickle=False)
    Path(path + ".json").write_text(json.dumps({"corpus_sha256": corpus_sha, "model": model, "revision": revision}),
                                    encoding="utf-8")


def model_encoder(batch_size: int = 32, name: str = MODEL, revision: str = REVISION):
    import numpy
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    model = SentenceTransformer(name, revision=revision, device="cpu")

    def encode(texts):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)

    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__, "numpy": numpy.__version__,
                "max_seq_length": model.max_seq_length, "cpu": platform.processor() or platform.machine()}
    return encode, versions


def main(argv=None, encoder=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="BEIR dataset folder (relevance-gate fetch prints it)")
    p.add_argument("--split", default="test")
    p.add_argument("-k", type=int, default=100, help="results kept per query (default 100)")
    p.add_argument("-o", "--output", required=True, help="run file to write (.gz to compress)")
    p.add_argument("--meta", help="also write what made the run (model, versions, timings) as JSON here")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model", default=MODEL, help=f"a sentence-transformers model (default {MODEL})")
    p.add_argument("--revision", help="its commit on the Hugging Face hub; required with --model, so the run can be "
                                      "made again")
    p.add_argument("--tag", help="the run's tag (default: the model's name)")
    p.add_argument("--doc-cache", help="keep the document vectors in this .npy file and reuse them when it exists, "
                                       "so another split's queries do not embed the corpus again")
    a = p.parse_args(argv)
    if a.model != MODEL and not a.revision:
        p.error("--model needs --revision (the model's commit), so the run can be made again")
    revision = a.revision or REVISION
    tag = a.tag or a.model.rsplit("/", 1)[-1]

    doc_ids, doc_texts, qids, qtexts = load(Path(a.data).expanduser(), a.split)
    if encoder is None:
        encode, versions = model_encoder(a.batch_size, a.model, revision)
    else:
        encode, versions = encoder, {"encoder": "test"}
    corpus_sha = hashlib.sha256("\0".join(doc_ids + doc_texts).encode("utf-8")).hexdigest()
    t0 = time.perf_counter()
    doc_vecs = cached_docs(a.doc_cache, corpus_sha, len(doc_ids), a.model, revision) if a.doc_cache else None
    cached = doc_vecs is not None
    if not cached:
        doc_vecs = encode(doc_texts)
        if a.doc_cache:
            save_docs(a.doc_cache, corpus_sha, doc_vecs, a.model, revision)
    t1 = time.perf_counter()
    query_vecs = encode([QUERY_PREFIX + q for q in qtexts])
    t2 = time.perf_counter()
    ranked = top_k(doc_vecs, query_vecs, a.k)
    t3 = time.perf_counter()
    sha = write(Path(a.output), run_text(qids, ranked, doc_ids, tag))
    meta = {"model": a.model, "revision": revision, "query_prefix": QUERY_PREFIX, "dataset": Path(a.data).name,
            "split": a.split, "docs": len(doc_ids), "queries": len(qids), "k": a.k, "sha256": sha,
            "seconds": {"docs": round(t1 - t0, 2), "queries": round(t2 - t1, 2), "search": round(t3 - t2, 3)},
            "docs_per_second": None if cached else round(len(doc_ids) / max(t1 - t0, 1e-9), 1),
            "docs_from_cache": cached, "versions": versions}
    if a.meta:
        Path(a.meta).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
