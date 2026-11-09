"""Ranking metrics. Same definitions as trec_eval and the BEIR benchmark, so numbers compare.

nDCG uses linear gain (the grade itself) and a log2 rank discount. A document returned twice
counts once, at its first position.
"""

from __future__ import annotations

import math
import re
from typing import Callable

Ranked = list
Judged = dict


def _dedupe(ranked: Ranked) -> list:
    seen: set = set()
    out = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(ranked: Ranked, judged: Judged, k: int) -> float:
    ideal = dcg(sorted((g for g in judged.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg(max(judged.get(d, 0), 0) for d in ranked[:k]) / ideal


def precision(ranked: Ranked, judged: Judged, k: int) -> float:
    return sum(1 for d in ranked[:k] if judged.get(d, 0) > 0) / k


def recall(ranked: Ranked, judged: Judged, k: int) -> float:
    relevant = sum(1 for g in judged.values() if g > 0)
    if relevant == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if judged.get(d, 0) > 0) / relevant


def mrr(ranked: Ranked, judged: Judged, k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if judged.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def hit(ranked: Ranked, judged: Judged, k: int) -> float:
    return 1.0 if any(judged.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def zero_results(ranked: Ranked, judged: Judged, k: int) -> float:
    return 0.0 if ranked else 1.0


FUNCS: dict[str, Callable] = {"ndcg": ndcg, "p": precision, "recall": recall, "mrr": mrr, "hit": hit}
LOWER_IS_BETTER = {"zero_results"}
DEFAULT = ["ndcg@10", "p@10", "recall@100", "mrr@10", "zero_results"]


def parse(name: str) -> tuple[Callable, int]:
    if name == "zero_results":
        return zero_results, 0
    m = re.fullmatch(r"(ndcg|p|recall|mrr|hit)@(\d+)", name)
    if not m or int(m.group(2)) < 1:
        raise ValueError(f"unknown metric {name!r}; use ndcg@K, p@K, recall@K, mrr@K, hit@K or zero_results")
    return FUNCS[m.group(1)], int(m.group(2))


def higher_is_better(name: str) -> bool:
    return name not in LOWER_IS_BETTER


def score(ranked: Ranked, judged: Judged, names: list[str]) -> dict[str, float]:
    ranked = _dedupe(ranked)
    out = {}
    for n in names:
        fn, k = parse(n)
        out[n] = fn(ranked, judged, k)
    return out


def depth_needed(names: list[str]) -> int:
    return max([parse(n)[1] for n in names] + [10])
