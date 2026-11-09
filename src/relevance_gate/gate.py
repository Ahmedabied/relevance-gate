"""Evaluate an engine into a report, and compare two reports with a gate.

The gate fails only when a metric got worse by more than its tolerance AND the drop is unlikely
to be noise (a paired randomization test over the queries both reports share). A small set of
judged queries cannot prove much; the gate says so instead of failing on luck.
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


def evaluate(engine: Engine, ds: Dataset, names: list[str] | None = None) -> dict:
    names = names or M.DEFAULT
    for n in names:
        M.parse(n)
    depth = M.depth_needed(names)
    run, ms = engine.search_all(ds.queries, depth)
    per_query = {}
    for qid, judged in ds.qrels.items():
        s = M.score(run.get(qid, []), judged, names)
        per_query[qid] = {"query": ds.queries[qid], **{k: round(v, 6) for k, v in s.items()}}
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


def paired_p_value(a: list[float], b: list[float], rounds: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomization test on the mean difference. Deterministic for a seed."""
    diffs = [y - x for x, y in zip(a, b) if y != x]
    if not diffs:
        return 1.0
    n = len(a)
    observed = abs(sum(diffs)) / n
    rng = random.Random(seed)
    hits = 0
    for _ in range(rounds):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s) / n >= observed - 1e-12:
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


def compare(base: dict, cand: dict, tolerances: dict[str, float], alpha: float = 0.05,
            rounds: int = 10000) -> tuple[list[Check], dict]:
    shared = sorted(set(base["per_query"]) & set(cand["per_query"]))
    info = {"shared": len(shared), "only_baseline": len(set(base["per_query"]) - set(cand["per_query"])),
            "only_candidate": len(set(cand["per_query"]) - set(base["per_query"]))}
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
        worse = delta < -tol if M.higher_is_better(metric) else delta > tol
        p = paired_p_value(a, b, rounds) if worse else 1.0
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
