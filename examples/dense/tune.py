"""Pick the fusion settings on held-out queries, never on the ones you report.

    python examples/dense/tune.py --data ~/.cache/relevance-gate/scifact --split train \
        --dense examples/dense/scifact-train.run.gz

Fuses the built-in BM25 with the dense run for every k and weight in the grid, scores each on the held-out
split, and prints the table and the best setting. Then evaluate that one setting on test, once:

    relevance-gate eval --data ~/.cache/relevance-gate/scifact --fuse examples/dense/scifact.run.gz \
        --rrf-k K --fuse-weight W -o tuned.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from relevance_gate.data import load
from relevance_gate.engines import Fts5Engine, RunEngine
from relevance_gate.gate import evaluate
from relevance_gate.runs import rrf

KS = [5, 10, 20, 40, 60, 100, 200]
WEIGHTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


class Fixed(RunEngine):
    """Ranked lists already in memory."""

    def __init__(self, run, name):
        self.run, self.missing, self.label, self.path = run, 0, name, Path(name)

    def describe(self):
        return {"name": self.label}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True, help="the held-out split (train or dev), never test")
    p.add_argument("--dense", required=True, help="the dense run for that split")
    p.add_argument("--metric", default="ndcg@10")
    p.add_argument("--depth", type=int, default=100)
    a = p.parse_args(argv)
    if a.split == "test":
        sys.exit("tuning on test is what this script exists to avoid; pick train or dev")
    ds = load(a.data, a.split)
    t0 = time.perf_counter()
    bm25, _ = Fts5Engine(ds.docs).search_all(ds.queries, a.depth)
    dense = RunEngine(a.dense)
    dense_run, _ = dense.search_all(ds.queries, a.depth)
    t1 = time.perf_counter()
    score = {}
    for k in KS:
        for w in WEIGHTS:
            fused = {q: rrf([bm25[q], dense_run[q]], [1.0, w], k, a.depth) for q in ds.queries}
            score[k, w] = evaluate(Fixed(fused, "fused"), ds, [a.metric], query_text=False)["metrics"][a.metric]
    alone = {name: evaluate(Fixed(run, name), ds, [a.metric], query_text=False)["metrics"][a.metric]
             for name, run in (("bm25", bm25), ("dense", dense_run))}
    t2 = time.perf_counter()
    print(f"{ds.name} {a.split}: {len(ds.queries)} queries, {a.metric}; BM25 alone {alone['bm25']:.4f}, "
          f"dense alone {alone['dense']:.4f} ({t1 - t0:.1f} s to search, {t2 - t1:.1f} s for {len(score)} settings)")
    print("\n| k \\ weight | " + " | ".join(f"{w:g}" for w in WEIGHTS) + " |")
    print("|---|" + "---|" * len(WEIGHTS))
    for k in KS:
        print(f"| {k} | " + " | ".join(f"{score[k, w]:.4f}" for w in WEIGHTS) + " |")
    best = max(score, key=lambda kw: (score[kw], kw == (60, 1.0)))
    print(f"\nbest: k {best[0]}, weight {best[1]:g}: {score[best]:.4f} "
          f"(default k 60, weight 1: {score[60, 1.0]:.4f}, gain {score[best] - score[60, 1.0]:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
