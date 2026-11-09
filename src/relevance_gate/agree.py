"""How far do two judges agree? Cohen's kappa on the documents both graded.

A judged set is only as good as its judgments. Have a second person grade a sample of the same pool, then:
raw agreement says how often they gave the same grade; kappa says how much of that is beyond chance (two judges
who mark nearly everything "not relevant" agree a lot by luck). For graded relevance the weighted kappa counts a
2 against a 1 as a smaller miss than a 2 against a 0. The documents they disagree on most are the ones to talk
through and settle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Landis and Koch (1977), the usual words for kappa. A rough guide, not a law.
BANDS = [(0.0, "poor"), (0.21, "fair"), (0.41, "moderate"), (0.61, "substantial"), (0.81, "almost perfect")]


def band(k: float | None) -> str:
    if k is None:
        return "not defined"
    word = "poor"
    for lo, w in BANDS:
        if k >= lo:
            word = w
    return word


def kappa(x: list[int], y: list[int], weights: str = "none") -> float | None:
    """Cohen's kappa for paired grades. weights: "none", "linear" or "quadratic" (by grade distance).

    None when it is not defined: every grade from both judges is the same value, so chance explains it all."""
    if not x or len(x) != len(y):
        return None
    cats = sorted(set(x) | set(y))
    if len(cats) < 2:
        return None
    pos = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(x)
    obs = [[0.0] * k for _ in range(k)]
    for a, b in zip(x, y):
        obs[pos[a]][pos[b]] += 1
    rows = [sum(r) for r in obs]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        d = abs(cats[i] - cats[j]) / (cats[-1] - cats[0])
        return d * d if weights == "quadratic" else d

    seen = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    chance = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    if chance == 0:
        return None
    return 1.0 - seen / chance


def interval(x: list[int], y: list[int], weights: str, rounds: int = 2000, seed: int = 0,
             groups: list[str] | None = None) -> tuple[float, float] | None:
    """95% bootstrap interval. Deterministic for a seed.

    groups (the query of each document): resample queries and keep each one's documents together. Judges disagree
    by query (an ambiguous query splits them on many of its documents), so documents are not independent, and
    resampling them one by one gives an interval that is too narrow."""
    rng = random.Random(seed)
    clusters: dict[str, list[int]] = {}
    for i, g in enumerate(groups if groups is not None else range(len(x))):
        clusters.setdefault(str(g), []).append(i)
    units = list(clusters.values())
    vals = []
    for _ in range(rounds):
        idx = [i for _ in units for i in units[rng.randrange(len(units))]]
        v = kappa([x[i] for i in idx], [y[i] for i in idx], weights)
        if v is not None:
            vals.append(v)
    if len(vals) < rounds // 2:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


@dataclass
class Agreement:
    pairs: list[tuple[str, str, int, int]]  # query, document, grade from A, grade from B
    only_a: int
    only_b: int

    @property
    def a(self) -> list[int]:
        return [p[2] for p in self.pairs]

    @property
    def b(self) -> list[int]:
        return [p[3] for p in self.pairs]

    def interval(self, weights: str) -> tuple[float, float] | None:
        """Bootstrap over queries: a query's documents stay together."""
        return interval(self.a, self.b, weights, groups=[p[0] for p in self.pairs])

    def exact(self) -> float:
        return sum(p[2] == p[3] for p in self.pairs) / len(self.pairs) if self.pairs else 0.0

    def disagreements(self, min_diff: int = 1) -> list[tuple[str, str, int, int]]:
        """Biggest gaps first, then by query and document so the list is stable."""
        out = [p for p in self.pairs if abs(p[2] - p[3]) >= min_diff]
        return sorted(out, key=lambda p: (-abs(p[2] - p[3]), p[0], p[1]))


def compare(a: dict[str, dict[str, int]], b: dict[str, dict[str, int]]) -> Agreement:
    keys_a = {(q, d) for q, docs in a.items() for d in docs}
    keys_b = {(q, d) for q, docs in b.items() for d in docs}
    shared = sorted(keys_a & keys_b)
    return Agreement([(q, d, a[q][d], b[q][d]) for q, d in shared], len(keys_a - keys_b), len(keys_b - keys_a))


def markdown(ag: Agreement, name_a: str = "A", name_b: str = "B", show: int = 10) -> str:
    x, y = ag.a, ag.b
    grades = sorted(set(x) | set(y))
    lines = [f"### Judge agreement: {name_a} and {name_b}", "",
             f"{len(ag.pairs)} documents graded by both"
             + (f" ({ag.only_a} only by {name_a}, {ag.only_b} only by {name_b})" if ag.only_a or ag.only_b else "") + ".", ""]
    rows = [("same grade", f"{ag.exact():.1%}", "")]
    for label, w in (("kappa", "none"), ("weighted kappa (quadratic)", "quadratic")):
        k = kappa(x, y, w)
        ci = ag.interval(w) if k is not None else None
        rows.append((label, "not defined" if k is None else f"{k:.3f}",
                     (f"95% {ci[0]:.3f} to {ci[1]:.3f} over queries, " if ci else "") + band(k)))
    if len(grades) > 2:
        kb = kappa([int(v > 0) for v in x], [int(v > 0) for v in y])
        rows.append(("kappa, relevant or not", "not defined" if kb is None else f"{kb:.3f}", band(kb)))
    lines += ["| | value | |", "|---|---|---|"] + [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
    if ag.pairs:
        lines += ["", f"Grades ({name_a} down, {name_b} across):", "",
                  "| | " + " | ".join(str(g) for g in grades) + " |", "|---" * (len(grades) + 1) + "|"]
        for g in grades:
            lines.append(f"| {g} | " + " | ".join(str(sum(1 for p in ag.pairs if p[2] == g and p[3] == h)) for h in grades) + " |")
    worst = ag.disagreements()[:show]
    if worst:
        lines += ["", "Biggest disagreements, to talk through first:", ""]
        lines += [f"- `{q}` `{d}`: {name_a} {ga}, {name_b} {gb}" for q, d, ga, gb in worst]
    queries = len({p[0] for p in ag.pairs})
    if 0 < len(ag.pairs) < 50 or 0 < queries < 10:
        lines += ["", f"Only {len(ag.pairs)} documents over {queries} quer{'y' if queries == 1 else 'ies'}: the interval is "
                      "wide. Have both judges grade more of the same pool."]
    return "\n".join(lines) + "\n"
