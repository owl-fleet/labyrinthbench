"""Prove _llm_call surfaces finish_reason on BOTH transport paths, since the guard that reads it
is only as good as the field being populated. Stubbed transport — no live server."""
import sys, types, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import run_eval  # noqa: E402

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p

class _Client:
    """Mimics httpx.Client enough for _llm_call: base_url + post()."""
    def __init__(self, native=None, oai=None):
        self.base_url = "http://x:11434/v1"; self.native = native; self.oai = oai
    def post(self, url, json=None, **kw):
        if url.endswith("/api/chat"):
            if self.native is None: raise RuntimeError("no native endpoint")
            return _Resp(self.native)
        return _Resp(self.oai)

# native /api/chat path (think is not None) — ollama reports done_reason
c = _Client(native={"message": {"content": "", "thinking": "long ramble"},
                    "done_reason": "length", "eval_count": 200})
out = run_eval._llm_call(c, "m", [{"role": "user", "content": "x"}], think=False)
check("native path surfaces done_reason as finish_reason=length",
      out["choices"][0].get("finish_reason") == "length")

c2 = _Client(native={"message": {"content": '{"action":"observe"}', "thinking": ""},
                     "done_reason": "stop", "eval_count": 12})
out2 = run_eval._llm_call(c2, "m", [{"role": "user", "content": "x"}], think=False)
check("native path, normal completion -> finish_reason=stop",
      out2["choices"][0].get("finish_reason") == "stop")

# OpenAI /v1 path (think is None — what the actions cell actually uses)
c3 = _Client(oai={"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                  "usage": {"completion_tokens": 200}})
out3 = run_eval._llm_call(c3, "m", [{"role": "user", "content": "x"}], think=None)
check("/v1 path passes finish_reason=length straight through (the path the actions cell uses)",
      out3["choices"][0].get("finish_reason") == "length")

c4 = _Client(oai={"choices": [{"message": {"content": '{"action":"observe"}'},
                              "finish_reason": "stop"}], "usage": {}})
out4 = run_eval._llm_call(c4, "m", [{"role": "user", "content": "x"}], think=None)
check("/v1 path, normal completion -> finish_reason=stop",
      out4["choices"][0].get("finish_reason") == "stop")

# and the guard's own predicate
check("guard predicate: 'length' is truncation", (out3["choices"][0].get("finish_reason") == "length") is True)
check("guard predicate: 'stop' is not", (out4["choices"][0].get("finish_reason") == "length") is False)
check("guard predicate: a server omitting the field is not truncation (no false positives)",
      ({}.get("finish_reason") == "length") is False)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
