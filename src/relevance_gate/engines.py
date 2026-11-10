"""Search engines to evaluate.

- fts5: a built-in BM25 engine on SQLite FTS5 over the dataset's corpus. A baseline, and the demo.
- http: your own search API. One GET per query; the result ids are read from the JSON response.
- command: your own program. Gets every query as JSON lines on stdin, answers with JSON lines.
- run: results made elsewhere, read from a TREC run file (a vector search, a batch job).

Any of them can be fused with run files by reciprocal rank fusion (FusedEngine), e.g. keyword plus dense.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .runs import load_run, rrf

_WORD = re.compile(r"\w+", re.UNICODE)


class EngineError(RuntimeError):
    pass


class Engine:
    name = "engine"

    def describe(self) -> dict:
        return {"name": self.name}

    def search(self, query: str, k: int) -> list[str]:
        raise NotImplementedError

    def search_all(self, queries: dict[str, str], k: int) -> tuple[dict[str, list[str]], dict[str, float]]:
        """Returns ranked ids and the time per query in milliseconds."""
        run, ms = {}, {}
        for qid, text in queries.items():
            t0 = time.perf_counter()
            run[qid] = self.search(text, k)
            ms[qid] = (time.perf_counter() - t0) * 1000
        return run, ms


class Fts5Engine(Engine):
    name = "fts5"

    def __init__(self, docs: dict[str, tuple[str, str]], tokenizer: str = "porter unicode61 remove_diacritics 2",
                 mode: str = "or", title_weight: float = 1.0):
        if mode not in ("or", "and"):
            raise EngineError("fts5 mode must be 'or' or 'and'")
        if not re.fullmatch(r"[a-z0-9_ ]+", tokenizer):
            raise EngineError(f"fts5 tokenizer must be plain words, got {tokenizer!r}")
        self.tokenizer, self.mode, self.title_weight = tokenizer, mode, float(title_weight)
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute(f"CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, title, body, tokenize='{tokenizer}')")
        t0 = time.perf_counter()
        self.db.executemany("INSERT INTO docs VALUES (?, ?, ?)", ((d, t, b) for d, (t, b) in docs.items()))
        self.db.commit()
        self.index_ms = (time.perf_counter() - t0) * 1000
        self.size = len(docs)

    def describe(self) -> dict:
        return {"name": self.name, "tokenizer": self.tokenizer, "mode": self.mode,
                "title_weight": self.title_weight, "docs": self.size}

    def match_expr(self, query: str) -> str:
        # Every word becomes a quoted phrase, so FTS5 operators in user input are just words.
        terms = list(dict.fromkeys(t.lower() for t in _WORD.findall(query)))
        joiner = " OR " if self.mode == "or" else " AND "
        return joiner.join('"' + t.replace('"', '""') + '"' for t in terms)

    def search(self, query: str, k: int) -> list[str]:
        expr = self.match_expr(query)
        if not expr:
            return []
        rows = self.db.execute(
            "SELECT doc_id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs, 0.0, ?, 1.0) LIMIT ?",
            (expr, self.title_weight, k),
        )
        return [r[0] for r in rows]


def _pick(obj: Any, path: str) -> list:
    """Read ids from a JSON response. 'hits.*._id' walks into hits, maps over the list, takes _id."""
    cur = [obj]
    for part in path.split(".") if path else []:
        nxt = []
        for c in cur:
            if part == "*":
                if isinstance(c, list):
                    nxt.extend(c)
            elif isinstance(c, dict) and part in c:
                nxt.append(c[part])
            elif isinstance(c, list) and part.isdigit() and int(part) < len(c):
                nxt.append(c[int(part)])
        cur = nxt
    if len(cur) == 1 and isinstance(cur[0], list):
        cur = cur[0]
    return [str(x) for x in cur if isinstance(x, (str, int))]


class HttpEngine(Engine):
    name = "http"

    def __init__(self, url: str, ids: str, workers: int = 8, timeout: float = 30):
        if "{query}" not in url:
            raise EngineError("the http engine needs {query} in the URL, e.g. http://localhost:8000/search?q={query}")
        if not url.startswith(("http://", "https://")):
            raise EngineError("the http engine only fetches http:// or https:// URLs")
        self.url, self.ids, self.workers, self.timeout = url, ids, workers, timeout

    def describe(self) -> dict:
        return {"name": self.name, "url": self.url, "ids": self.ids}

    def search(self, query: str, k: int) -> list[str]:
        url = self.url.replace("{query}", urllib.parse.quote(query)).replace("{k}", str(k))
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "relevance-gate"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return _pick(json.loads(r.read()), self.ids)[:k]
        except (OSError, ValueError) as e:
            raise EngineError(f"search request failed for {query!r}: {e}") from None

    def search_all(self, queries, k):
        def one(item):
            qid, text = item
            t0 = time.perf_counter()
            ids = self.search(text, k)
            return qid, ids, (time.perf_counter() - t0) * 1000

        run, ms = {}, {}
        with ThreadPoolExecutor(self.workers) as pool:
            for qid, ids, t in pool.map(one, queries.items()):
                run[qid], ms[qid] = ids, t
        return run, ms


class CommandEngine(Engine):
    name = "command"

    def __init__(self, command: list[str], timeout: float = 600):
        if not command:
            raise EngineError("the command engine needs a command")
        self.command, self.timeout = command, timeout

    def describe(self) -> dict:
        return {"name": self.name, "command": self.command}

    def search_all(self, queries, k):
        payload = "".join(json.dumps({"qid": q, "query": t, "k": k}) + "\n" for q, t in queries.items())
        t0 = time.perf_counter()
        try:
            r = subprocess.run(self.command, input=payload, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise EngineError(f"search command failed: {e}") from None
        if r.returncode != 0:
            raise EngineError(f"search command exited {r.returncode}: {r.stderr.strip()[:300]}")
        total = (time.perf_counter() - t0) * 1000
        run = {}
        for line in r.stdout.splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                except ValueError:
                    raise EngineError(f"search command printed a line that is not JSON: {line[:120]!r}") from None
                run[str(obj.get("qid"))] = [str(x) for x in obj.get("ids", [])][:k]
        each = total / max(len(queries), 1)
        return {q: run.get(q, []) for q in queries}, {q: each for q in queries}


class RunEngine(Engine):
    """Results from a TREC run file, looked up by query id. A file has no latency, so none is reported."""

    name = "run"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.run, self.sha256 = load_run(self.path)
        self.missing = 0

    def describe(self) -> dict:
        return {"name": self.name, "file": self.path.name, "sha256": self.sha256[:16]}

    def search(self, query: str, k: int) -> list[str]:
        raise EngineError("a run file answers by query id, not by text")

    def search_all(self, queries, k):
        hits = [q for q in queries if q in self.run]
        if queries and not hits:
            raise EngineError(f"{self.path.name} has none of the {len(queries)} judged queries "
                              f"(e.g. {sorted(queries)[0]!r}); is it from another dataset or split?")
        self.missing = len(queries) - len(hits)
        return {q: self.run.get(q, [])[:k] for q in queries}, {}


class FusedEngine(Engine):
    """An engine's results mixed with run files by reciprocal rank fusion.

    The engine's list has weight 1 and each run `weight`. Every list is cut to `depth` (at least k)
    before fusing. The time per query is the engine's plus the fusion's.
    """

    def __init__(self, base: Engine, runs: list[RunEngine], weight: float = 1.0, rrf_k: float = 60.0,
                 depth: int = 100):
        if not runs:
            raise EngineError("fusion needs at least one run file")
        rrf([[] for _ in range(len(runs) + 1)], [1.0] + [weight] * len(runs), rrf_k)  # checks the settings
        if depth < 1:
            raise EngineError(f"fusion depth must be 1 or more, got {depth}")
        self.base, self.runs, self.weight, self.rrf_k, self.depth = base, runs, float(weight), float(rrf_k), depth
        self.name = base.name

    def describe(self) -> dict:
        fused = [{"file": r.path.name, "sha256": r.sha256[:16], "weight": self.weight} for r in self.runs]
        return self.base.describe() | {"fused": fused, "rrf_k": self.rrf_k, "fuse_depth": self.depth}

    def search(self, query: str, k: int) -> list[str]:
        raise EngineError("fusion with a run file answers by query id; use search_all")

    def search_all(self, queries, k):
        depth = max(k, self.depth)
        base, ms = self.base.search_all(queries, depth)
        others = [r.search_all(queries, depth)[0] for r in self.runs]
        weights = [1.0] + [self.weight] * len(self.runs)
        out = {}
        for q in queries:
            t0 = time.perf_counter()
            out[q] = rrf([base.get(q, [])] + [o[q] for o in others], weights, self.rrf_k, depth)[:k]
            if q in ms:
                ms[q] += (time.perf_counter() - t0) * 1000
        return out, ms
