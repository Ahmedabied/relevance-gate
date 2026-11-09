import json
import random

import pytest

TOPICS = {
    "volcano": ["eruption", "lava", "magma", "ash"],
    "garden": ["tomato", "compost", "seedling", "watering"],
    "railway": ["locomotive", "signal", "timetable", "carriage"],
    "bakery": ["sourdough", "oven", "flour", "yeast"],
    "astronomy": ["telescope", "galaxy", "orbit", "comet"],
    "sailing": ["harbour", "mast", "anchor", "tide"],
}
FILLER = ["the", "a", "report", "notes", "about", "with", "local", "daily", "small", "guide"]


def make_dataset(root, n_docs_per_topic=8, seed=1):
    """A tiny judged collection where stemming and OR matching matter."""
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "qrels").mkdir(exist_ok=True)
    docs, queries, qrels = [], [], []
    for t, words in TOPICS.items():
        for i in range(n_docs_per_topic):
            body = " ".join(rng.choice(words + FILLER) for _ in range(30))
            docs.append({"_id": f"{t}-{i}", "title": f"{t} {rng.choice(words)}", "text": body})
        for j, w in enumerate(words):
            qid = f"q-{t}-{j}"
            queries.append({"_id": qid, "text": f"{t}s {w}s"})  # plural forms: needs stemming
            for i in range(n_docs_per_topic):
                qrels.append((qid, f"{t}-{i}", 2 if i < 2 else 1))
    (root / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (root / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries + [{"_id": "unjudged", "text": "x"}]))
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t{s}\n" for q, d, s in qrels))
    return root


@pytest.fixture
def toy(tmp_path):
    return make_dataset(tmp_path / "toy")
