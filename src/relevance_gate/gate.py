"""Evaluate an engine into a report, and compare two reports with a gate.

The gate fails only when a metric got worse by more than its allowed drop AND that is unlikely to
be noise: a one-sided paired randomization test over the queries both reports share, with the
allowed drop built into the null. A small set of judged queries cannot prove much; the gate says
so instead of failing on luck.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import metrics as M
from .data import Dataset
from .engines import Engine


def evaluate(engine: Engine, ds: Dataset, names: list[str] | None = None, query_text: bool = True) -> dict:
    names = names or M.DEFAULT
    for n in names:
        M.parse(n)
    depth = M.depth_needed(names)
    run, ms = engine.search_all(ds.queries, depth)
    per_query = {}
    for qid, judged in ds.qrels.items():
        s = M.score(run.get(qid, []), judged, names)
        per_query[qid] = ({"query": ds.queries[qid]} if query_text else {}) | {k: round(v, 6) for k, v in s.items()}
    means = {n: round(statistics.fmean(pq[n] for pq in per_query.values()), 6) if per_query else 0.0 for n in names}
    lat = sorted(ms.values())
    return {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": ds.name,
        "engine": engine.describe(),
        "queries": len(per_query),
        "metrics": means,
        "latency_ms": {"p50": round(lat[len(lat) // 2], 3), "p95": round(lat[int(len(lat) * 0.95)], 3)} if lat else {},
        "per_query": per_query,
    }


def load_report(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or "per_query" not in data:
        raise ValueError(f"{path}: not a relevance-gate report")
    return data


def drop_p_value(a: list[float], b: list[float], allowed: float = 0.0, higher_is_better: bool = True,
                 rounds: int = 10000, seed: int = 0) -> float:
    """One-sided paired randomization test: is the candidate worse than the baseline by more than `allowed`?

    The null is "the true drop is at most `allowed`". Each query's change is moved by the allowed drop,
    so a change of exactly -allowed counts as zero, then signs are flipped at random. p is the share of
    flips at least as bad as what was observed. Deterministic for a seed.
    """
    sign = 1.0 if higher_is_better else -1.0
    shifted = [sign * (y - x) + allowed for x, y in zip(a, b)]
    shifted = [v for v in shifted if v != 0.0]  # flipping a zero changes nothing
    if not shifted:
        return 1.0
    # A flip keeps the sign of the queries whose bit is set: flipped sum = 2 * kept - total. It is at
    # least as bad as observed (the total) when kept <= total. Subset sums are read from one 256-entry
    # table per 8 queries, so a round costs n / 8 lookups instead of n random draws.
    tables = []
    for i in range(0, len(shifted), 8):
        chunk = shifted[i:i + 8]
        table = [0.0] * (1 << len(chunk))
        for mask in range(1, len(table)):
            low = mask & -mask
            table[mask] = table[mask ^ low] + chunk[low.bit_length() - 1]
        tables.append(table)
    total = sum(shifted)
    limit = total + 1e-9 * (1.0 + sum(abs(v) for v in shifted))
    nbytes, nbits = len(tables), len(shifted)
    bits = random.Random(seed).getrandbits
    hits = 0
    for _ in range(rounds):
        if sum(map(list.__getitem__, tables, bits(nbits).to_bytes(nbytes, "little"))) <= limit:
            hits += 1
    return (hits + 1) / (rounds + 1)


@dataclass
class Check:
    metric: str
    baseline: float
    candidate: float
    delta: float
    tolerance: float
    p: float
    worse: bool
    failed: bool


MIN_SHARED = 0.5  # share of the baseline's judged queries the candidate must also have


def compare(base: dict, cand: dict, tolerances: dict[str, float], alpha: float = 0.05,
            rounds: int = 10000) -> tuple[list[Check], dict]:
    shared = sorted(set(base["per_query"]) & set(cand["per_query"]))
    info = {"shared": len(shared), "only_baseline": len(set(base["per_query"]) - set(cand["per_query"])),
            "only_candidate": len(set(cand["per_query"]) - set(base["per_query"]))}
    n_base = len(base["per_query"])
    if not shared or len(shared) < MIN_SHARED * n_base:
        raise ValueError(f"the reports share {len(shared)} of the baseline's {n_base} judged queries, so they cannot be "
                         "compared. Were both made from the same judged set? Re-run the baseline if the set changed.")
    checks = []
    for metric, tol in tolerances.items():
        missing = [r for r in (base, cand) if shared and metric not in r["per_query"][shared[0]]]
        if missing:
            raise ValueError(f"metric {metric} is not in both reports; evaluate with --metrics including it")
        a = [base["per_query"][q][metric] for q in shared]
        b = [cand["per_query"][q][metric] for q in shared]
        ma = statistics.fmean(a) if a else 0.0
        mb = statistics.fmean(b) if b else 0.0
        delta = mb - ma
        up = M.higher_is_better(metric)
        worse = delta < -tol if up else delta > tol
        p = drop_p_value(a, b, tol, up, rounds) if worse else 1.0
        checks.append(Check(metric, ma, mb, delta, tol, p, worse, worse and p < alpha))
    return checks, info


def movers(base: dict, cand: dict, metric: str, n: int = 5) -> tuple[list, list]:
    shared = set(base["per_query"]) & set(cand["per_query"])
    sign = 1 if M.higher_is_better(metric) else -1
    moves = sorted(((cand["per_query"][q][metric] - base["per_query"][q][metric]) * sign, q) for q in shared)
    worst = [(q, d * sign, cand["per_query"][q].get("query", "")) for d, q in moves[:n] if d < 0]
    best = [(q, d * sign, cand["per_query"][q].get("query", "")) for d, q in reversed(moves[-n:]) if d > 0]
    return worst, best


def markdown(checks: list[Check], info: dict, base: dict, cand: dict, alpha: float) -> str:
    failed = any(c.failed for c in checks)
    lines = [f"### relevance-gate: {'FAIL, search got worse' if failed else 'pass'}", "",
             f"{info['shared']} judged queries compared"
             + (f" ({info['only_baseline']} only in baseline, {info['only_candidate']} only in candidate)"
                if info["only_baseline"] or info["only_candidate"] else "") + ".", "",
             "| metric | baseline | candidate | change | allowed drop | p | result |",
             "|---|---|---|---|---|---|---|"]
    for c in checks:
        if c.failed:
            result = "FAIL"
        elif c.worse:
            result = f"worse, but not significant at {alpha:g}"
        else:
            result = "ok"
        p = f"{c.p:.4f}" if c.worse else ""
        lines.append(f"| {c.metric} | {c.baseline:.4f} | {c.candidate:.4f} | {c.delta:+.4f} | {c.tolerance:g} | {p} | {result} |")
    worst, best = movers(base, cand, checks[0].metric) if checks else ([], [])
    if worst:
        lines += ["", f"Queries that lost the most {checks[0].metric}:", ""]
        lines += [f"- `{q}` {d:+.3f}: {text[:90]}" for q, d, text in worst]
    if best:
        lines += ["", f"Queries that gained the most {checks[0].metric}:", ""]
        lines += [f"- `{q}` {d:+.3f}: {text[:90]}" for q, d, text in best]
    if info["shared"] < 50:
        lines += ["", f"Only {info['shared']} queries: small drops cannot be told apart from noise. Judge more queries."]
    return "\n".join(lines) + "\n"
