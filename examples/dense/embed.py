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


def model_encoder(batch_size: int = 32):
    import numpy
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")

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
    a = p.parse_args(argv)

    doc_ids, doc_texts, qids, qtexts = load(Path(a.data).expanduser(), a.split)
    if encoder is None:
        encode, versions = model_encoder(a.batch_size)
    else:
        encode, versions = encoder, {"encoder": "test"}
    t0 = time.perf_counter()
    doc_vecs = encode(doc_texts)
    t1 = time.perf_counter()
    query_vecs = encode([QUERY_PREFIX + q for q in qtexts])
    t2 = time.perf_counter()
    ranked = top_k(doc_vecs, query_vecs, a.k)
    t3 = time.perf_counter()
    sha = write(Path(a.output), run_text(qids, ranked, doc_ids))
    meta = {"model": MODEL, "revision": REVISION, "query_prefix": QUERY_PREFIX, "dataset": Path(a.data).name,
            "split": a.split, "docs": len(doc_ids), "queries": len(qids), "k": a.k, "sha256": sha,
            "seconds": {"docs": round(t1 - t0, 2), "queries": round(t2 - t1, 2), "search": round(t3 - t2, 3)},
            "docs_per_second": round(len(doc_ids) / max(t1 - t0, 1e-9), 1), "versions": versions}
    if a.meta:
        Path(a.meta).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
