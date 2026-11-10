"""Run files and rank fusion.

A run file is the TREC format every search evaluation tool reads and writes, one result per line:

    query-id  Q0  doc-id  rank  score  tag

It lets you score results made anywhere (a vector database, a batch job, another team's engine) and mix
them with a live engine by reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009):
each list gives a document weight / (k + rank), and the sums decide the order.
"""

from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path

from .data import DataError


def load_run(path: str | Path) -> tuple[dict[str, list[str]], str]:
    """Ranked ids per query, best first, and the file's sha256. Gzip is read by its bytes, not its name.

    The order is by score, highest first, as trec_eval reads it; equal scores keep the file's rank. A
    document listed twice for one query keeps its best place.
    """
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    name = Path(path).name
    best: dict[str, dict[str, tuple[float, int, int]]] = {}
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 6:
            raise DataError(f"{name}:{n}: a run line has 6 fields (query-id Q0 doc-id rank score tag), got {len(parts)}")
        qid, _, doc, rank, score, _ = parts
        try:
            r = int(rank)
        except ValueError:
            raise DataError(f"{name}:{n}: rank must be an integer, got {rank!r}") from None
        try:
            s = float(score)
        except ValueError:
            raise DataError(f"{name}:{n}: score must be a number, got {score!r}") from None
        if not math.isfinite(s):
            raise DataError(f"{name}:{n}: score must be a finite number, got {score!r}")
        key = (-s, r, n)
        q = best.setdefault(qid, {})
        if doc not in q or key < q[doc]:
            q[doc] = key
    if not best:
        raise DataError(f"{name}: no results in the run file")
    return {qid: sorted(docs, key=docs.__getitem__) for qid, docs in best.items()}, sha


def write_run(run: dict[str, list[str]], path: str | Path, tag: str) -> None:
    """Write ranked ids as a TREC run, queries in id order. The score is the place from the bottom (last = 1).

    A path ending in .gz is compressed with no timestamp, so the same run gives the same bytes.
    """
    tag = "".join(tag.split()) or "run"
    lines = []
    for qid in sorted(run):
        ids = list(dict.fromkeys(run[qid]))
        for rank, doc in enumerate(ids, 1):
            lines.append(f"{qid} Q0 {doc} {rank} {len(ids) - rank + 1} {tag}\n")
    data = "".join(lines).encode("utf-8")
    if str(path).endswith(".gz"):
        data = gzip.compress(data, mtime=0)
    Path(path).write_bytes(data)


def rrf(lists: list[list[str]], weights: list[float] | None = None, k: float = 60.0,
        depth: int | None = None) -> list[str]:
    """Reciprocal rank fusion. Each list adds weight / (k + rank) to its documents; highest sum first.

    A document repeated in one list counts once, at its first place. Each list is cut to `depth` first.
    A list with weight 0 adds nothing (so fusing at weight 0 leaves the first list as it was). Equal
    sums go to the document with the better best rank, then to the smaller id, so the order never
    depends on dict order.
    """
    weights = [1.0] * len(lists) if weights is None else [float(w) for w in weights]
    if len(weights) != len(lists):
        raise ValueError(f"rrf needs one weight per list, got {len(weights)} for {len(lists)}")
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"rrf k must be a positive number, got {k}")
    if any(not (math.isfinite(w) and w >= 0) for w in weights):
        raise ValueError(f"rrf weights must be zero or more, got {weights}")
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for w, ids in zip(weights, lists):
        if w == 0:
            continue
        ranked = list(dict.fromkeys(ids))
        if depth is not None:
            ranked = ranked[:depth]
        for r, doc in enumerate(ranked, 1):
            score[doc] = score.get(doc, 0.0) + w / (k + r)
            if r < best.get(doc, r + 1):
                best[doc] = r
    return sorted(score, key=lambda d: (-score[d], best[d], d))
