# relevance-gate

Search gets worse quietly. A tokenizer change, a new boost, a reindex, and nobody notices until people stop
finding things. relevance-gate scores your search on judged queries and fails the pull request when it gets worse.

```
search log --sample--> queries --judge--> judged set --eval--> report --gate--> pass / FAIL in CI
```

- **`sample`** picks which logged searches to judge: a third from the head, torso and tail of your traffic, with
  searches that look like personal data dropped.
- **`judge`** pools the top results of one or more search configs, shuffles them so you cannot tell which config
  found what, and asks for a grade in the terminal. Quit any time; a rerun continues where you stopped.
- **`eval`** runs the judged queries through a search engine and writes a report: nDCG, precision, recall, MRR,
  zero-result rate and latency, per query and overall. The engine is the built-in SQLite FTS5 (BM25), your own
  HTTP search API, or any program that reads queries on stdin.
- **`gate`** compares a candidate report to a baseline and exits 1 if a metric dropped by more than you allow
  **and** the drop is unlikely to be noise (a paired randomization test over the queries). A GitHub Action
  wraps it.

Stdlib Python only. No dependencies.

## See it block a change

[Pull request #1](https://github.com/Ahmedabied/relevance-gate/pull/1) changes one setting: the tokenizer goes
from `porter unicode61` to plain `unicode61`, so "shoes" stops matching "shoe". It looks harmless in review. The
`search-quality` check runs the 300 judged SciFact queries and fails it:

| metric | baseline | candidate | change | allowed drop | p | result |
|---|---|---|---|---|---|---|
| ndcg@10 | 0.6874 | 0.6695 | -0.0180 | 0.005 | 0.0701 | worse, but not significant at 0.05 |
| recall@100 | 0.9253 | 0.8859 | -0.0394 | 0.005 | 0.0028 | FAIL |
| zero_results | 0.0000 | 0.0000 | +0.0000 | 0.005 | | ok |

Note the first row. nDCG@10 dropped by 0.018, but on 300 queries a drop that size happens by chance 7% of the
time, so the gate reports it instead of failing on it. A gate that fails on noise gets switched off.

## Try it in under a minute

```sh
git clone https://github.com/Ahmedabied/relevance-gate && cd relevance-gate
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
relevance-gate fetch scifact                      # 2.8 MB, hash-checked, from the BEIR benchmark
relevance-gate eval --data ~/.cache/relevance-gate/scifact -o base.json
relevance-gate eval --data ~/.cache/relevance-gate/scifact --opt mode=and -o and.json
relevance-gate gate base.json and.json            # exit 1: zero results went from 0% to 95%
```

The datasets are downloaded from BEIR and never bundled here. Check each one's licence before you use it
(SciFact is CC BY-NC 2.0). The baseline committed in `examples/scifact` holds ids and numbers only.

## Numbers from real runs

The built-in engine is SQLite FTS5 with BM25, porter stemming and OR matching. On two public BEIR datasets it
lands next to the BM25 results published with the benchmark, which is a check that the metrics are computed the
standard way (linear-gain nDCG, as in trec_eval and BEIR):

| dataset | queries | docs | nDCG@10 here | BM25 nDCG@10 in the BEIR paper | recall@100 | full eval |
|---|---|---|---|---|---|---|
| SciFact | 300 | 5,183 | 0.687 | 0.665 | 0.925 | 2.0 s |
| NFCorpus | 323 | 3,633 | 0.322 | 0.325 | 0.249 | 0.6 s |

"Full eval" is the whole command on an Intel i5-14400F: load, index, run every query, score, write the report.
The gate's randomization test (10,000 rounds, 300 queries) takes under 0.1 s. Search changes tried on SciFact:

| change | nDCG@10 | recall@100 | zero results | gate |
|---|---|---|---|---|
| baseline (porter, OR, title weight 1) | 0.6874 | 0.9253 | 0.0% | |
| no stemming | 0.6695 | 0.8859 | 0.0% | FAIL (recall, p 0.003) |
| AND instead of OR | 0.0467 | 0.0458 | 94.7% | FAIL (p 0.0001) |
| title weight 3 | 0.6930 | 0.9253 | 0.0% | pass |

## Use it on your own search

**1. Build the judged set** from your search log (a text file with one search per line, a CSV with a `query`
column and optional `count`, or JSON lines):

```sh
relevance-gate sample search-log.txt -n 150 -o mysearch/queries.jsonl
relevance-gate judge --data mysearch --url 'http://localhost:8000/search?q={query}' --ids 'results.*.id'
```

`judge` writes `mysearch/qrels/test.tsv`. Give it a `corpus.jsonl` too (`{"_id", "title", "text"}`) and it shows
each document's text, and can pool extra FTS5 configs with `--pool 'tokenizer=unicode61'` so the judged set is
not shaped by one engine alone. 150 queries with 10 results each is a few hours of judging, once.

**2. Evaluate** your engine. Over HTTP (one GET per query, ids read from the JSON with a path like
`hits.hits.*._id`):

```sh
relevance-gate eval --data mysearch --engine http \
    --url 'http://localhost:8000/search?q={query}&size={k}' --ids 'hits.hits.*._id' -o baseline.json
```

Or with a program: it gets `{"qid", "query", "k"}` lines on stdin and prints `{"qid", "ids": [...]}` lines.

```sh
relevance-gate eval --data mysearch --engine command --cmd 'python my_search.py' -o baseline.json
```

**3. Gate in CI.** Commit `baseline.json` from main. In the pull request, evaluate again and gate:

```yaml
- run: relevance-gate eval --data mysearch --engine command --cmd 'python my_search.py' -o candidate.json
- uses: Ahmedabied/relevance-gate@v0.1.0
  with:
    baseline: baseline.json
    candidate: candidate.json
    fail-on: ndcg@10=0.005,recall@100=0.005,zero_results=0.005
```

The action needs Python 3.10 or newer on the runner (`actions/setup-python`). The result table goes to the job
summary. When a change is meant to move the numbers, update `baseline.json` in
the same pull request, so the new baseline is reviewed like code. A full working example is
[.github/workflows/search-quality.yml](.github/workflows/search-quality.yml).

## Metrics

| name | meaning |
|---|---|
| `ndcg@K` | graded ranking quality in the top K, linear gain, log2 discount |
| `p@K` | share of the top K that is relevant |
| `recall@K` | share of all relevant documents found in the top K |
| `mrr@K` | 1 / rank of the first relevant result |
| `hit@K` | 1 if anything relevant is in the top K |
| `zero_results` | share of queries with no results at all (lower is better) |

A document returned twice counts once. Pick metrics with `--metrics ndcg@10,recall@100,zero_results`.

## Limits

- **Judgments are the expensive part, and they age.** The gate is only as good as the judged set. Unjudged
  documents count as not relevant, so a new engine that finds good documents nobody judged scores too low. Judge
  again with `judge --pool` including the new engine when that happens.
- **Small sets cannot see small changes.** On the 300 SciFact queries the gate caught a 0.039 recall drop
  (p 0.003) but could not tell a 0.018 nDCG drop from noise (p 0.07). Under about 50 queries it says so in its
  output. More judged queries make smaller drops visible.
- **One judge.** There is no inter-judge agreement or adjudication yet.
- **Latency numbers are indicative**, measured in the same process as the evaluation, not under load.
- **The built-in engine is a baseline**, not a product: one SQLite FTS5 table, BM25, no synonyms. Hybrid search
  with local embeddings is the next step.

## Development

```sh
pip install -e '.[test]'
pytest -q               # 45 tests, about 1.5 s
```

Tests use a generated corpus, so CI needs no downloads. The `search-quality` workflow runs the real SciFact gate.

Built with AI pair programming.

## Licence

MIT. Datasets you fetch keep their own licences.
