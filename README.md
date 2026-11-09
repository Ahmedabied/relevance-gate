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
  **and** that is unlikely to be noise: a one-sided paired randomization test over the queries, asking whether
  the true drop is bigger than the allowed one. A GitHub Action wraps it.
- **`agree`** checks that two judges grade alike before you trust the set: Cohen's kappa (plain, weighted for
  graded relevance, and relevant-or-not) with a bootstrap interval over queries (a query's documents are not
  independent), the grade table, and the documents they disagree on most, to settle first.

Stdlib Python only. No dependencies.

## See it block a change

[Pull request #1](https://github.com/Ahmedabied/relevance-gate/pull/1) changes one setting: the tokenizer goes
from `porter unicode61` to plain `unicode61`, so "shoes" stops matching "shoe". It looks harmless in review. The
`search-quality` check runs the 300 judged SciFact queries and fails it:

| metric | baseline | candidate | change | allowed drop | p | result |
|---|---|---|---|---|---|---|
| ndcg@10 | 0.6874 | 0.6695 | -0.0180 | 0.005 | 0.0983 | worse, but not significant at 0.05 |
| recall@100 | 0.9253 | 0.8859 | -0.0394 | 0.005 | 0.0028 | FAIL |
| zero_results | 0.0000 | 0.0000 | +0.0000 | 0.005 | | ok |

Note the first row. nDCG@10 dropped by 0.018, and you allow 0.005. The gate asks: if the true drop were only the
0.005 you allow, how often would 300 queries show a drop this big? About 10% of the time (p 0.098), so it reports
the drop instead of failing on it. That some drop happened is likely (p 0.032 against no change at all), but not
one beyond what you allow. Recall fails because its drop clears the allowed 0.005 with p 0.003. A gate that fails
on noise gets switched off.

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
The gate's randomization test (10,000 rounds) takes 9 ms for 300 queries and 88 ms for 3,000; the whole `gate`
command takes 0.07 s. Search changes tried on SciFact:

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

Rather grade in the browser? Add `--pool-out pool.jsonl` and `judge` writes the documents still to grade instead
of asking. Open that file in [judge-ui](https://github.com/Ahmedabied/judge-ui) (keys 2, 1, 0; progress kept in
the browser) and save its export as `mysearch/qrels/test.tsv`.

**Check the judges agree.** Have a second person grade a sample of the same pool (say 20 queries) into their own
file, then:

```sh
relevance-gate agree sam.tsv alex.tsv --disagreements settle.tsv
```

```
| | value | |
|---|---|---|
| same grade | 70.0% |  |
| kappa | 0.545 | 95% 0.091 to 1.000, moderate |
| weighted kappa (quadratic) | 0.565 | 95% -0.064 to 1.000, moderate |
| kappa, relevant or not | 0.583 | moderate |
```

That is a 10-document example, so the intervals are wide. Raw agreement flatters: two judges who mark nearly
everything "not relevant" agree most of the time by luck, and kappa takes that out. Talk through `settle.tsv`
(biggest gaps first), write down the rule you agreed on, and fix the grades. `--min-kappa 0.6` makes it a CI check.

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
  (p 0.003) but could not show that a 0.018 nDCG drop was beyond the allowed 0.005 (p 0.10). Under about 50
  queries it says so in its output. More judged queries make smaller drops visible. How often it fails, in
  simulation (`python examples/power.py`: 300 queries, per-query noise sd 0.1, 1,000 runs per row, default
  settings):

  | true nDCG@10 drop | 0 | 0.005 (the allowed drop) | 0.01 | 0.02 | 0.03 | 0.05 |
  |---|---|---|---|---|---|---|
  | gate fails | 1.5% | 6.7% | 13.0% | 54.4% | 89.3% | 100% |

  Each metric is tested on its own at alpha, so with three metrics the chance of a fail at exactly the allowed
  drop is a little above alpha (6.7% here at 0.05). With no real change it fails 1.5% of the time.
- **Both reports must come from the same judged set.** The gate exits 2 when the candidate has fewer than half of
  the baseline's judged queries; re-run the baseline when the judged set changes. Extra queries in the candidate
  are fine (they are left out of the comparison and counted in the output).
- **Agreement is measured, not settled.** `agree` finds where two judges differ; settling it is a conversation, and
  the settled grades go back into one qrels file by hand.
- **Latency numbers are indicative**, measured in the same process as the evaluation, not under load.
- **The built-in engine is a baseline**, not a product: one SQLite FTS5 table, BM25, no synonyms. Hybrid search
  with local embeddings is the next step.

## Development

```sh
pip install -e '.[test]'
pytest -q               # 60 tests, about 2 s
```

Tests use a generated corpus, so CI needs no downloads. The `search-quality` workflow runs the real SciFact gate.

Built with AI pair programming.

## Licence

MIT. Datasets you fetch keep their own licences.
