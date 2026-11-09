import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from relevance_gate.data import load
from relevance_gate.engines import CommandEngine, EngineError, Fts5Engine, HttpEngine, _pick

DOCS = {
    "1": ("Running shoes", "light shoes for long runs"),
    "2": ("Trail guide", "maps of mountain trails"),
    "3": ("Shoe repair", "how to fix a sole"),
}


def test_fts5_or_and_stemming():
    e = Fts5Engine(DOCS)
    assert e.search("running shoes", 10)[0] == "1"
    assert set(e.search("shoe", 10)) == {"1", "3"}  # porter: shoe == shoes
    assert e.search("shoes mountain", 10)  # OR finds something
    assert Fts5Engine(DOCS, mode="and").search("shoes mountain", 10) == []
    assert Fts5Engine(DOCS, tokenizer="unicode61").search("shoe", 10) == ["3"]  # no stemming


def test_fts5_title_weight_changes_order():
    docs = {"t": ("comet", "notes"), "b": ("notes", "comet comet comet comet")}
    assert Fts5Engine(docs, title_weight=1.0).search("comet", 2)[0] == "b"
    assert Fts5Engine(docs, title_weight=20.0).search("comet", 2)[0] == "t"


@pytest.mark.parametrize("q", ['"', 'NEAR(shoes', "shoes OR", "a*b", "col:shoes", "-shoes", "", "   ", "))((", "shoes\x00"])
def test_fts5_query_syntax_is_never_interpreted(q):
    assert isinstance(Fts5Engine(DOCS).search(q, 10), list)


def test_fts5_rejects_odd_tokenizer():
    with pytest.raises(EngineError):
        Fts5Engine(DOCS, tokenizer="porter'); DROP TABLE docs; --")


def test_pick_paths():
    body = {"hits": {"hits": [{"_id": "a"}, {"_id": "b"}]}, "ids": ["x", 2], "results": [{"doc": {"id": "k"}}]}
    assert _pick(body, "hits.hits.*._id") == ["a", "b"]
    assert _pick(body, "ids") == ["x", "2"]
    assert _pick(body, "results.*.doc.id") == ["k"]
    assert _pick(body, "nothing.here") == []


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)["q"][0]
        ids = [d for d, (t, b) in DOCS.items() if any(w in (t + " " + b).lower() for w in q.lower().split())]
        body = json.dumps({"results": [{"id": i} for i in ids]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_http_engine():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        e = HttpEngine(f"http://127.0.0.1:{srv.server_port}/search?q={{query}}&n={{k}}", "results.*.id")
        run, ms = e.search_all({"a": "shoes", "b": "mountain trails", "c": "nothing"}, 10)
        assert run == {"a": ["1"], "b": ["2"], "c": []} and all(v >= 0 for v in ms.values())
    finally:
        srv.shutdown()
    with pytest.raises(EngineError):
        HttpEngine("file:///etc/passwd?{query}", "x")
    with pytest.raises(EngineError):
        HttpEngine("http://x/search", "x")


def test_command_engine(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import sys, json\nfor l in sys.stdin:\n    q = json.loads(l)\n"
                      "    print(json.dumps({'qid': q['qid'], 'ids': [q['query'][::-1]]}))\n")
    run, _ = CommandEngine([sys.executable, str(script)]).search_all({"1": "ab", "2": "cd"}, 5)
    assert run == {"1": ["ba"], "2": ["dc"]}
    with pytest.raises(EngineError, match="exited 3"):
        CommandEngine([sys.executable, "-c", "import sys; sys.exit(3)"]).search_all({"1": "x"}, 5)


def test_fts5_on_toy(toy):
    ds = load(toy)
    e = Fts5Engine(ds.docs)
    assert e.search(ds.queries["q-volcano-0"], 10)[0].startswith("volcano")
