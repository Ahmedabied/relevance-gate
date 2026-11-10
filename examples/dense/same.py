"""Do two dense runs give the same numbers? Used by the dense workflow to check the committed runs.

    python examples/dense/same.py --data ~/.cache/relevance-gate/scifact examples/dense/scifact.run.gz out/scifact.run.gz

Exits 1 if any metric differs by more than --tolerance. Embeddings made on another CPU can differ in the
last digits and swap two near-equal documents, so the lists are compared too, but only the metrics fail it.
"""

from __future__ import annotations

import argparse
import sys

from relevance_gate.data import load
from relevance_gate.engines import RunEngine
from relevance_gate.gate import evaluate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("committed")
    p.add_argument("fresh")
    p.add_argument("--tolerance", type=float, default=0.002)
    a = p.parse_args(argv)
    ds = load(a.data, a.split, with_corpus=False)
    runs = [RunEngine(a.committed), RunEngine(a.fresh)]
    reports = [evaluate(r, ds, query_text=False) for r in runs]
    same10 = sum(runs[0].run.get(q, [])[:10] == runs[1].run.get(q, [])[:10] for q in ds.queries)
    print(f"{ds.name} {a.split}: top 10 identical for {same10} of {len(ds.queries)} queries")
    bad = 0
    for name, x in reports[0]["metrics"].items():
        y = reports[1]["metrics"][name]
        flag = "" if abs(x - y) <= a.tolerance else "  DIFFERS"
        bad += bool(flag)
        print(f"  {name:<14}{x:.4f}  {y:.4f}{flag}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
