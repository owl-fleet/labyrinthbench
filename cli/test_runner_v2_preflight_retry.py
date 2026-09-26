"""Runner v2 (MCV chunk 04 owed item, read out 2026-09-09; point-and-click chunk 01 depends on
the same runner): prove (1) _preflight_upstream FAILS closed against a dead/unreachable model
upstream instead of only being discovered per-row deep inside a campaign, and (2) _llm_call
retries a single HTTP 500 EXACTLY ONCE before letting the row become an error, distinct from the
existing connection-error backoff loop (which still gets its own regression check here so the new
except clause can't have shadowed it). Stubbed clients throughout — no live server, no network."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import httpx
import run_eval  # noqa: E402

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

# _llm_call sleeps for real between retries (settle time, connection backoff) — patch it out so
# this script doesn't cost the campaign's actual wall-clock delays just to prove the branching.
_slept = []
run_eval.time.sleep = lambda s: _slept.append(s)


class _FakeResp:
    """Mimics an httpx.Response enough for _llm_call: .status_code, .raise_for_status(), .json()."""
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"{self.status_code} error", request=None, response=self)
    def json(self):
        return self._payload


def _ok_payload():
    return {"choices": [{"message": {"content": '{"action":"observe"}'},
                         "finish_reason": "stop"}], "usage": {}}


class _SeqClient:
    """Mimics httpx.Client enough for _llm_call's /chat/completions path (think=None, so the
    native /api/chat branch is skipped entirely): base_url + post() returning queued responses."""
    def __init__(self, responses):
        self.base_url = "http://x:11434/v1"
        self._responses = list(responses)
        self.calls = 0
    def post(self, url, json=None, **kw):
        assert url == "/chat/completions", url
        self.calls += 1
        return self._responses.pop(0)


class _RaisingThenOKClient:
    """First .post() raises a connection-level error (unrelated to the new 500 path); second
    succeeds — regression guard that the existing multi-attempt backoff loop still works."""
    def __init__(self):
        self.base_url = "http://x:11434/v1"
        self.calls = 0
    def post(self, url, json=None, **kw):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("connection refused")
        return _FakeResp(200, _ok_payload())


# --- (2a) a single HTTP 500 is retried once, after a settle, and the row succeeds ---
c1 = _SeqClient([_FakeResp(500), _FakeResp(200, _ok_payload())])
out1 = run_eval._llm_call(c1, "m", [{"role": "user", "content": "x"}], think=None)
check("500 then 200: the row succeeds (does not become an error)",
      out1["choices"][0]["message"]["content"] == '{"action":"observe"}')
check("500 then 200: exactly 2 POSTs made (the one allowed retry, not the general retries=3 loop)",
      c1.calls == 2)
check("500 then 200: the retry slept for the settle window before trying again",
      run_eval._HTTP_500_SETTLE_SECONDS in _slept)

# --- (2b) TWO consecutive 500s: the one retry is spent, the second is NOT retried again ---
_slept.clear()
c2 = _SeqClient([_FakeResp(500), _FakeResp(500)])
raised = None
try:
    run_eval._llm_call(c2, "m", [{"role": "user", "content": "x"}], think=None)
except httpx.HTTPStatusError as e:
    raised = e
check("two consecutive 500s: the second IS raised (recorded as an error row by main(), not lost "
      "silently and not retried a second time)", raised is not None)
if raised is not None:
    check("two consecutive 500s: the raised error is the 500", raised.response.status_code == 500)
check("two consecutive 500s: exactly 2 POSTs made — the retry-once rule, never a third attempt",
      c2.calls == 2)

# --- (2c) a non-500 status (e.g. 503) is OUT of scope for the retry-once rule: raises immediately
#     — documents the intentional narrow scope ("an HTTP 500", not any 5xx) ---
c3 = _SeqClient([_FakeResp(503)])
raised3 = None
try:
    run_eval._llm_call(c3, "m", [{"role": "user", "content": "x"}], think=None)
except httpx.HTTPStatusError as e:
    raised3 = e
check("a 503 (not 500) is NOT covered by the retry-once rule: raises on the first attempt",
      raised3 is not None and raised3.response.status_code == 503)
check("a 503: only 1 POST attempted — no retry for non-500 statuses", c3.calls == 1)

# --- (2d) regression: the pre-existing connection-error backoff loop is untouched by the new
#     except clause (a ConnectError must still be retried and can still succeed) ---
c4 = _RaisingThenOKClient()
out4 = run_eval._llm_call(c4, "m", [{"role": "user", "content": "x"}], think=None)
check("regression: a connection error (not a 500) still retries via the existing loop and succeeds",
      out4["choices"][0]["message"]["content"] == '{"action":"observe"}')
check("regression: 2 POSTs made (1 failed, 1 succeeded)", c4.calls == 2)


# --- (1) _preflight_upstream ---
class _OKGetClient:
    def get(self, url, **kw):
        return _FakeResp(200, {"data": [{"id": "qwen3:14b"}]})
    def close(self):
        pass


class _DeadGetClient:
    def get(self, url, **kw):
        raise httpx.ConnectError("connection refused")
    def close(self):
        pass


class _ClosedClient:
    """Server answers but with a non-2xx (e.g. a gateway returning 502) — also a FAIL."""
    def get(self, url, **kw):
        return _FakeResp(502)
    def close(self):
        pass


check("preflight: a reachable upstream returns True",
      run_eval._preflight_upstream("http://x:11434/v1", client=_OKGetClient()) is True)
check("preflight: an unreachable upstream returns False (not an exception)",
      run_eval._preflight_upstream("http://dead:11434/v1", client=_DeadGetClient()) is False)
check("preflight: a non-2xx from the endpoint (e.g. 502 through a dead gateway) also returns False",
      run_eval._preflight_upstream("http://gw:11434/v1", client=_ClosedClient()) is False)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
