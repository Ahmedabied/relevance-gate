"""How often does the gate fail? Simulated pairs of reports with a known true change.

Each simulated query has a true nDCG and recall drawn at random; both reports see it with independent
per-query noise (sd 0.1, clipped to 0..1), and the candidate's nDCG is lowered by the true drop. The
gate runs with its defaults (three metrics, 0.005 allowed each, alpha 0.05).

    python examples/power.py            # about 10 s
"""

from __future__ import annotations

import random
import sys

from relevance_gate.gate import compare

QUERIES, SIMS, NOISE = 300, 1000, 0.1
TOLERANCES = {"ndcg@10": 0.005, "recall@100": 0.005, "zero_results": 0.005}


def clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def fail_rate(true_drop: float, rng: random.Random) -> float:
    fails = 0
    for _ in range(SIMS):
        base, cand = {}, {}
        for i in range(QUERIES):
            nd, rc = rng.random(), rng.random()
            base[str(i)] = {"ndcg@10": clip(nd + rng.gauss(0, NOISE)), "recall@100": clip(rc + rng.gauss(0, NOISE)),
                            "zero_results": 0.0}
            cand[str(i)] = {"ndcg@10": clip(nd - true_drop + rng.gauss(0, NOISE)),
                            "recall@100": clip(rc + rng.gauss(0, NOISE)), "zero_results": 0.0}
        checks, _ = compare({"per_query": base}, {"per_query": cand}, TOLERANCES, rounds=2000)
        fails += any(c.failed for c in checks)
    return fails / SIMS


def main() -> int:
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
    print(f"{QUERIES} queries, per-query noise sd {NOISE}, {SIMS} simulated gate runs per row\n")
    print("| true nDCG@10 drop | gate fails |")
    print("|---|---|")
    for drop in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        print(f"| {drop:.3f} | {fail_rate(drop, rng):.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
