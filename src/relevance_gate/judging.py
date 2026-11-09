"""From a query log to a judged test set.

sample: pick which logged queries to judge. Uniform sampling of distinct queries is mostly rare
        tail queries; sampling by traffic is mostly the same few head queries. This splits the
        log into head, torso and tail by traffic (a third of all searches each) and samples
        evenly from each, so the test set covers both. Queries that look like personal data
        (emails, phone numbers, long digit runs) are dropped unless you ask to keep them.

judge:  for each sampled query, pool the top results of one or more search configs, shuffle
        them so the judge cannot tell which config found what, and ask for a grade. Grades are
        appended to qrels/test.tsv as they are given, so quitting keeps the work and a rerun
        continues where it stopped.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, TextIO

from .data import DataError, _jsonl, load_qrels

PII = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"            # email
    r"|\+?\d[\d\s().-]{7,}\d"             # phone-like
    r"|\b\d{6,}\b"                         # long digit runs: ids, card or account numbers
)


def normalise(q: str) -> str:
    return " ".join(q.lower().split())


def read_log(path: str | Path) -> Counter:
    """Plain text (one search per line), CSV with a 'query' column and optional 'count', or
    JSON lines with "query" and optional "count"."""
    path = Path(path)
    counts: Counter = Counter()
    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.lstrip()[:1]
    if path.suffix == ".jsonl" or first == "{":
        for obj in _jsonl(path):
            q = normalise(str(obj.get("query", "")))
            if q:
                counts[q] += int(obj.get("count", 1))
    elif path.suffix == ".csv":
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or "query" not in reader.fieldnames:
            raise DataError(f"{path}: a CSV log needs a 'query' column")
        for row in reader:
            q = normalise(row.get("query") or "")
            if q:
                counts[q] += int(row.get("count") or 1)
    else:
        for line in text.splitlines():
            q = normalise(line)
            if q:
                counts[q] += 1
    return counts


def query_id(q: str) -> str:
    return "q" + hashlib.sha256(q.encode("utf-8")).hexdigest()[:10]


def strata(counts: Counter) -> dict[str, list[str]]:
    total = sum(counts.values())
    out: dict[str, list[str]] = {"head": [], "torso": [], "tail": []}
    running = 0
    for q, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        share = running / total if total else 0
        out["head" if share < 1 / 3 else "torso" if share < 2 / 3 else "tail"].append(q)
        running += c
    return out


def sample(counts: Counter, n: int, seed: int = 0, keep_pii: bool = False) -> tuple[list[dict], dict]:
    dropped = 0
    if not keep_pii:
        clean = Counter({q: c for q, c in counts.items() if not PII.search(q)})
        dropped = len(counts) - len(clean)
        counts = clean
    groups = strata(counts)
    rng = random.Random(seed)
    picked: list[dict] = []
    share = {k: n // 3 + (1 if i < n % 3 else 0) for i, k in enumerate(groups)}
    spare = 0
    for name in ("head", "torso", "tail"):  # a short stratum hands its share to the next one
        want = share[name] + spare
        pool = groups[name]
        take = rng.sample(pool, min(want, len(pool)))
        spare = want - len(take)
        picked += [{"_id": query_id(q), "text": q, "stratum": name, "count": counts[q]} for q in sorted(take)]
    info = {"distinct": len(counts), "searches": sum(counts.values()), "dropped_pii": dropped,
            "strata": {k: len(v) for k, v in groups.items()}, "picked": len(picked)}
    return picked, info


def judge(data: str | Path, searchers: list[Callable[[str, int], list[str]]], depth: int, out: str | Path,
          ask_in: TextIO, say: TextIO, seed: int = 0) -> dict:
    root = Path(data)
    queries = {str(q["_id"]): str(q.get("text", "")) for q in _jsonl(root / "queries.jsonl")}
    docs = {}
    if (root / "corpus.jsonl").exists():
        docs = {str(d["_id"]): (str(d.get("title") or ""), str(d.get("text") or "")) for d in _jsonl(root / "corpus.jsonl")}
    out = Path(out)
    done = load_qrels(out) if out.exists() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists() or out.stat().st_size == 0
    stats = {"judged": 0, "skipped": 0, "queries": 0}
    with open(out, "a", encoding="utf-8") as f:
        if new_file:
            f.write("query-id\tcorpus-id\tscore\n")
        for qid, text in queries.items():
            pool: list[str] = []
            for s in searchers:
                pool += [d for d in s(text, depth) if d not in pool]
            todo = [d for d in pool if d not in done.get(qid, {})]
            if not todo:
                continue
            random.Random(f"{seed}:{qid}").shuffle(todo)  # blind: no hint of which config found it
            stats["queries"] += 1
            say.write(f"\n=== {text}   ({len(todo)} to judge)\n")
            for doc in todo:
                title, body = docs.get(doc, ("", ""))
                say.write(f"\n[{doc}] {title}\n{body[:400]}\n  grade: 2 relevant, 1 partly, 0 not, s skip, q quit > ")
                say.flush()
                while True:
                    answer = ask_in.readline()
                    if answer == "":
                        return stats
                    answer = answer.strip().lower()
                    if answer in ("0", "1", "2", "s", "q"):
                        break
                    say.write("  please type 0, 1, 2, s or q > ")
                    say.flush()
                if answer == "q":
                    return stats
                if answer == "s":
                    stats["skipped"] += 1
                    continue
                f.write(f"{qid}\t{doc}\t{answer}\n")
                f.flush()
                done.setdefault(qid, {})[doc] = int(answer)
                stats["judged"] += 1
    return stats


def write_queries(path: str | Path, picked: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in picked), encoding="utf-8")
