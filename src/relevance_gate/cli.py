"""Command line: relevance-gate fetch | eval | gate | show"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from . import __version__
from . import metrics as M
from .data import DataError, fetch, load
from .engines import CommandEngine, EngineError, Fts5Engine, HttpEngine
from .gate import compare, evaluate, load_report, markdown


def _opts(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--opt takes key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_fetch(a) -> int:
    path = fetch(a.name, a.dir)
    print(path)
    return 0


def cmd_eval(a) -> int:
    names = [n.strip() for n in a.metrics.split(",") if n.strip()]
    ds = load(a.data, a.split, with_corpus=a.engine == "fts5")
    if a.limit:
        keep = sorted(ds.queries)[: a.limit]
        ds.queries = {q: ds.queries[q] for q in keep}
        ds.qrels = {q: ds.qrels[q] for q in keep}
    o = _opts(a.opt)
    if a.engine == "fts5":
        unknown = set(o) - {"tokenizer", "mode", "title_weight"}
        if unknown:
            raise ValueError(f"fts5 options are tokenizer, mode and title_weight; got {sorted(unknown)}")
        engine = Fts5Engine(ds.docs, o.get("tokenizer", "porter unicode61 remove_diacritics 2"),
                            o.get("mode", "or"), float(o.get("title_weight", 1.0)))
    elif a.engine == "http":
        if not a.url or not a.ids:
            raise ValueError("the http engine needs --url (with {query}) and --ids (where the ids are in the JSON)")
        engine = HttpEngine(a.url, a.ids, workers=a.workers)
    else:
        if not a.cmd:
            raise ValueError("the command engine needs --cmd")
        engine = CommandEngine(shlex.split(a.cmd))
    report = evaluate(engine, ds, names)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.output:
        Path(a.output).write_text(text + "\n", encoding="utf-8")
    _summary(report)
    return 0


def _summary(report: dict) -> None:
    eng = report["engine"]
    desc = ", ".join(f"{k}={v}" for k, v in eng.items() if k != "name")
    print(f"{report['dataset']}: {report['queries']} queries, engine {eng['name']} ({desc})", file=sys.stderr)
    for k, v in report["metrics"].items():
        print(f"  {k:<14}{v:.4f}", file=sys.stderr)
    lat = report.get("latency_ms") or {}
    if lat:
        print(f"  {'latency':<14}p50 {lat['p50']:.2f} ms, p95 {lat['p95']:.2f} ms", file=sys.stderr)


def _tolerances(spec: str) -> dict[str, float]:
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, tol = part.partition("=")
        M.parse(name.strip())
        out[name.strip()] = float(tol) if tol else 0.0
    if not out:
        raise ValueError("--fail-on needs at least one metric, e.g. ndcg@10=0.01")
    return out


def cmd_gate(a) -> int:
    base, cand = load_report(a.baseline), load_report(a.candidate)
    checks, info = compare(base, cand, _tolerances(a.fail_on), a.alpha, a.rounds)
    md = markdown(checks, info, base, cand, a.alpha)
    print(md)
    summary = a.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return 1 if any(c.failed for c in checks) else 0


def cmd_show(a) -> int:
    _summary(load_report(a.report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relevance-gate", description="Search evals that block regressions in CI.")
    p.add_argument("--version", action="version", version=f"relevance-gate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download a public BEIR dataset (scifact, nfcorpus), hash-checked")
    f.add_argument("name")
    f.add_argument("--dir", help="where to put it (default ~/.cache/relevance-gate)")
    f.set_defaults(fn=cmd_fetch)

    e = sub.add_parser("eval", help="run the judged queries through a search engine and write a report")
    e.add_argument("--data", required=True, help="dataset folder (BEIR layout)")
    e.add_argument("--split", default="test")
    e.add_argument("--engine", choices=["fts5", "http", "command"], default="fts5")
    e.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="fts5 options: tokenizer='porter unicode61', mode=or|and, title_weight=2")
    e.add_argument("--url", help="http engine: URL with {query} (and optional {k})")
    e.add_argument("--ids", help="http engine: where the ids are in the JSON, e.g. hits.*.id")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--cmd", help="command engine: reads JSON lines {qid, query, k}, prints {qid, ids}")
    e.add_argument("--metrics", default=",".join(M.DEFAULT))
    e.add_argument("--limit", type=int, help="only the first N queries (by id)")
    e.add_argument("-o", "--output", help="write the report JSON here")
    e.set_defaults(fn=cmd_eval)

    g = sub.add_parser("gate", help="compare a candidate report to a baseline; exit 1 if search got worse")
    g.add_argument("baseline")
    g.add_argument("candidate")
    g.add_argument("--fail-on", default="ndcg@10=0.005,recall@100=0.005,zero_results=0.005",
                   help="metric=allowed drop, comma separated (default ndcg@10=0.005,recall@100=0.005,zero_results=0.005)")
    g.add_argument("--alpha", type=float, default=0.05, help="fail only if the drop has p below this")
    g.add_argument("--rounds", type=int, default=10000, help="randomization test rounds")
    g.add_argument("--summary", help="also append the markdown here (GITHUB_STEP_SUMMARY is used if set)")
    g.set_defaults(fn=cmd_gate)

    s = sub.add_parser("show", help="print a report's numbers")
    s.add_argument("report")
    s.set_defaults(fn=cmd_show)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (DataError, EngineError, ValueError, OSError) as e:
        print(f"relevance-gate: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
