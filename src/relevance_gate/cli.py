"""Command line: relevance-gate fetch | sample | judge | eval | gate | agree | show"""

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
from .engines import CommandEngine, EngineError, Fts5Engine, FusedEngine, HttpEngine, RunEngine
from .gate import compare, evaluate, load_report, markdown
from .runs import write_run


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
    elif a.engine == "run":
        if not a.run:
            raise ValueError("the run engine needs --run FILE (a TREC run: query-id Q0 doc-id rank score tag)")
        engine = RunEngine(a.run)
    elif a.engine == "http":
        if not a.url or not a.ids:
            raise ValueError("the http engine needs --url (with {query}) and --ids (where the ids are in the JSON)")
        engine = HttpEngine(a.url, a.ids, workers=a.workers)
    else:
        if not a.cmd:
            raise ValueError("the command engine needs --cmd")
        engine = CommandEngine(shlex.split(a.cmd))
    fused = [RunEngine(f) for f in a.fuse or []]
    if fused:
        engine = FusedEngine(engine, fused, a.fuse_weight, a.rrf_k, a.fuse_depth)
    ranked: dict[str, list[str]] = {}
    report = evaluate(engine, ds, names, query_text=not a.no_query_text, ranked=ranked)
    for r in fused + ([engine] if isinstance(engine, RunEngine) else []):
        if r.missing:
            print(f"relevance-gate: warning: {r.path.name} has no results for {r.missing} of {len(ds.queries)} "
                  "judged queries", file=sys.stderr)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.output:
        Path(a.output).write_text(text + "\n", encoding="utf-8")
    if a.run_out:
        write_run(ranked, a.run_out, engine.name)
    _summary(report)
    return 0


def _summary(report: dict) -> None:
    eng = report["engine"]
    skip = {"name", "fused", "rrf_k", "fuse_depth"}
    desc = ", ".join(f"{k}={v}" for k, v in eng.items() if k not in skip)
    print(f"{report['dataset']}: {report['queries']} queries, engine {eng['name']} ({desc})", file=sys.stderr)
    for f in eng.get("fused", []):
        print(f"  fused with {f['file']} (weight {f['weight']:g}, rrf k {eng['rrf_k']:g}, depth {eng['fuse_depth']})",
              file=sys.stderr)
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


def cmd_sample(a) -> int:
    from .judging import read_log, sample, write_queries

    picked, info = sample(read_log(a.log), a.n, a.seed, a.keep_pii)
    out = a.output or "queries.jsonl"
    write_queries(out, picked)
    st = info["strata"]
    print(f"{info['searches']} searches, {info['distinct']} distinct queries "
          f"(head {st['head']}, torso {st['torso']}, tail {st['tail']}); "
          f"dropped {info['dropped_pii']} that look like personal data")
    by = {k: sum(1 for q in picked if q["stratum"] == k) for k in ("head", "torso", "tail")}
    print(f"wrote {info['picked']} queries to {out}: head {by['head']}, torso {by['torso']}, tail {by['tail']}")
    return 0


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


def cmd_judge(a) -> int:
    from .data import _jsonl
    from .judging import judge

    root = Path(a.data)
    out = a.output or str(root / "qrels" / "test.tsv")
    if a.take:
        from .judging import safe, take

        s = take(a.take, out)
        print(f"{_count(s['added'], 'grade')} added to {out}, {len(s['changed'])} changed, {s['kept']} kept "
              f"({s['total']} in all)")
        for qid, doc, old, new in s["changed"][:20]:
            print(f"  {safe(qid)} {safe(doc)}: {old} -> {new}")
        if len(s["changed"]) > 20:
            print(f"  and {len(s['changed']) - 20} more")
        return 0
    searchers = []
    if a.url:
        if not a.ids:
            raise ValueError("--url needs --ids")
        searchers.append(HttpEngine(a.url, a.ids).search)
    if a.pool or not a.url:
        if not (root / "corpus.jsonl").exists():
            raise ValueError(f"{root}: the fts5 pool needs corpus.jsonl; or pool from your engine with --url")
        docs = {str(d["_id"]): (str(d.get("title") or ""), str(d.get("text") or "")) for d in _jsonl(root / "corpus.jsonl")}
        for spec in a.pool or [""]:
            o = _opts([x for x in spec.split(",") if x.strip()])
            searchers.append(Fts5Engine(docs, o.get("tokenizer", "porter unicode61 remove_diacritics 2"),
                                        o.get("mode", "or"), float(o.get("title_weight", 1.0))).search)
    if a.pool_out:
        from .judging import write_pool

        stats = write_pool(root, searchers, a.depth, out, a.pool_out)
        back = ["relevance-gate", "judge", "--data", a.data, *(["-o", a.output] if a.output else []), "--take"]
        print(f"{_count(stats['documents'], 'document')} for {_count(stats['queries'], 'query', 'queries')} "
              f"written to {a.pool_out}. Grade them "
              f"at https://judge-ui.ahmed-abied.workers.dev, then add the export to {out}:\n"
              f"  {' '.join(shlex.quote(x) for x in back)} grades.tsv")
        return 0
    stats = judge(root, searchers, a.depth, out, sys.stdin, sys.stdout)
    print(f"\n{_count(stats['judged'], 'grade')} saved to {out} ({stats['skipped']} skipped, "
          f"{_count(stats['queries'], 'query', 'queries')})")
    return 0


def cmd_agree(a) -> int:
    from .agree import compare, kappa, markdown
    from .data import load_qrels

    for f in (a.a, a.b):
        if not Path(f).is_file():
            raise ValueError(f"{f}: no such qrels file")
    ag = compare(load_qrels(a.a), load_qrels(a.b))
    if not ag.pairs:
        raise ValueError("the two qrels files share no graded documents; grade the same pool twice")
    md = markdown(ag, Path(a.a).stem if not a.names else a.names.split(",")[0],
                  Path(a.b).stem if not a.names else a.names.split(",")[-1])
    print(md)
    summary = a.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if a.disagreements:
        with open(a.disagreements, "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tgrade-a\tgrade-b\n")
            for q, d, ga, gb in ag.disagreements():
                f.write(f"{q}\t{d}\t{ga}\t{gb}\n")
        print(f"{len(ag.disagreements())} disagreements written to {a.disagreements}")
    if a.min_kappa is not None:
        k = kappa(ag.a, ag.b, "quadratic")
        if k is None or k < a.min_kappa:
            print(f"weighted kappa {'not defined' if k is None else f'{k:.3f}'} is below {a.min_kappa}", file=sys.stderr)
            return 1
    return 0


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
    e.add_argument("--engine", choices=["fts5", "http", "command", "run"], default="fts5")
    e.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="fts5 options: tokenizer='porter unicode61', mode=or|and, title_weight=2")
    e.add_argument("--url", help="http engine: URL with {query} (and optional {k})")
    e.add_argument("--ids", help="http engine: where the ids are in the JSON, e.g. hits.*.id")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--cmd", help="command engine: reads JSON lines {qid, query, k}, prints {qid, ids}")
    e.add_argument("--run", help="run engine: a TREC run file (query-id Q0 doc-id rank score tag), .gz is fine")
    e.add_argument("--fuse", action="append", metavar="RUN",
                   help="mix the engine's results with this run file by reciprocal rank fusion (repeatable)")
    e.add_argument("--fuse-weight", type=float, default=1.0, help="weight of each --fuse run; the engine's is 1")
    e.add_argument("--rrf-k", type=float, default=60.0, help="fusion constant k in weight / (k + rank) (default 60)")
    e.add_argument("--fuse-depth", type=int, default=100, help="results taken from each list before fusing")
    e.add_argument("--run-out", metavar="FILE", help="also write the ranked results as a TREC run (.gz to compress)")
    e.add_argument("--metrics", default=",".join(M.DEFAULT))
    e.add_argument("--limit", type=int, help="only the first N queries (by id)")
    e.add_argument("-o", "--output", help="write the report JSON here")
    e.add_argument("--no-query-text", action="store_true",
                   help="leave query text out of the report (for committing reports on datasets you may not share)")
    e.set_defaults(fn=cmd_eval)

    g = sub.add_parser("gate", help="compare a candidate report to a baseline; exit 1 if search got worse")
    g.add_argument("baseline")
    g.add_argument("candidate")
    g.add_argument("--fail-on", default="ndcg@10=0.005,recall@100=0.005,zero_results=0.005",
                   help="metric=allowed drop, comma separated (default ndcg@10=0.005,recall@100=0.005,zero_results=0.005)")
    g.add_argument("--alpha", type=float, default=0.05, help="fail only if p (chance of a drop this big if the true drop were within the allowed one) is below this")
    g.add_argument("--rounds", type=int, default=10000, help="randomization test rounds")
    g.add_argument("--summary", help="also append the markdown here (GITHUB_STEP_SUMMARY is used if set)")
    g.set_defaults(fn=cmd_gate)

    sm = sub.add_parser("sample", help="pick queries to judge from a search log (head, torso and tail)")
    sm.add_argument("log", help="text (one search per line), CSV with a query column, or JSON lines")
    sm.add_argument("-n", type=int, default=150, help="how many queries (default 150)")
    sm.add_argument("-o", "--output", help="queries.jsonl to write (default ./queries.jsonl)")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--keep-pii", action="store_true", help="keep queries that look like emails or phone numbers")
    sm.set_defaults(fn=cmd_sample)

    j = sub.add_parser("judge", help="grade pooled results for each query in the terminal; writes qrels")
    j.add_argument("--data", required=True, help="folder with queries.jsonl (and corpus.jsonl for the fts5 pool)")
    j.add_argument("--pool", action="append", metavar="OPTS",
                   help="an fts5 config to pool from, e.g. 'tokenizer=unicode61,mode=and'; repeat for more")
    j.add_argument("--url", help="also pool from your engine (URL with {query})")
    j.add_argument("--ids", help="where the ids are in your engine's JSON")
    j.add_argument("--depth", type=int, default=10, help="top results per config to pool (default 10)")
    j.add_argument("-o", "--output", help="qrels file (default DATA/qrels/test.tsv)")
    browser = j.add_mutually_exclusive_group()
    browser.add_argument("--pool-out", metavar="FILE", help="write the documents to judge to FILE for judge-ui in "
                                                          "the browser instead of asking here")
    browser.add_argument("--take", metavar="FILE", help="add a judge-ui export to the qrels file: new grades go in, "
                                                      "a newer grade wins, every earlier grade stays")
    j.set_defaults(fn=cmd_judge)

    ag = sub.add_parser("agree", help="how far two judges agree on the documents both graded (Cohen's kappa)")
    ag.add_argument("a", help="qrels TSV from the first judge")
    ag.add_argument("b", help="qrels TSV from the second judge")
    ag.add_argument("--names", help="what to call them, e.g. 'sam,alex' (default: the file names)")
    ag.add_argument("--disagreements", metavar="FILE", help="write every disagreement here, biggest first, to settle")
    ag.add_argument("--min-kappa", type=float, help="exit 1 if the weighted kappa is below this")
    ag.add_argument("--summary", help="also append the markdown here (GITHUB_STEP_SUMMARY is used if set)")
    ag.set_defaults(fn=cmd_agree)

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
