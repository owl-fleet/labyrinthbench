"""point-and-click chunk 01: prove the forced-choice arm's addition to cli/run_eval.py leaves
GENERATIVE mode's behavior byte-for-byte unchanged (the chunk's own Build-step requirement).
Three concrete, mechanically-checkable claims, no network, no GPU:

  1. _llm_call's /chat/completions JSON payload, called exactly as every EXISTING call site calls
     it (no new kwargs), is IDENTICAL to what it sent before the new max_tokens/logprobs/
     top_logprobs/grammar params existed — those are purely additive and only appear when truthy.
  2. build_system_prompt (generative's system-prompt builder) is untouched: it still demands a
     JSON object and says nothing about a menu/label. build_forced_choice_system_prompt is a
     genuinely SEPARATE function — it demands a label, never JSON.
  3. run_session's new parameters (response_mode, fc_*) all default to generative-mode-preserving
     values, so a caller that passes none of them (every pre-existing call site) gets exactly
     today's behavior.
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import httpx
import run_eval  # noqa: E402

FAILS = 0


def check(d, ok):
    global FAILS
    FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")


# --- (1) _llm_call payload parity ---
class _CapturingClient:
    """Mimics httpx.Client enough for _llm_call's /chat/completions path (think=None, so the
    native /api/chat branch is skipped entirely): records every payload it was asked to POST."""
    def __init__(self):
        self.base_url = "http://x:11434/v1"
        self.posts: list = []

    def post(self, url, json=None, **kw):
        self.posts.append((url, json))
        return _FakeResp(200, {"choices": [{"message": {"content": '{"action":"observe"}'},
                                             "finish_reason": "stop"}], "usage": {}})


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"{self.status_code}", request=None, response=self)

    def json(self):
        return self._payload


msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "obs"}]

c1 = _CapturingClient()
run_eval._llm_call(c1, "qwen3:14b", msgs, think=None)
check("no-options call: payload has EXACTLY {model, messages, stream} — no new keys leaked",
      c1.posts[0][1] == {"model": "qwen3:14b", "messages": msgs, "stream": False})

c2 = _CapturingClient()
run_eval._llm_call(c2, "qwen3:14b", msgs, options={"num_ctx": 16384}, think=None)
check("with-options call: payload has EXACTLY {model, messages, stream, options} — no new keys",
      c2.posts[0][1] == {"model": "qwen3:14b", "messages": msgs, "stream": False,
                          "options": {"num_ctx": 16384}})

# --- new params ARE additive when actually passed (proves they're real, not dead code) ---
c3 = _CapturingClient()
run_eval._llm_call(c3, "qwen3:14b", msgs, think=None, max_tokens=4, logprobs=True, top_logprobs=20,
                    grammar='root ::= "A" | "B"\n')
check("forced-choice-style call: max_tokens/logprobs/top_logprobs/grammar all present",
      c3.posts[0][1] == {"model": "qwen3:14b", "messages": msgs, "stream": False,
                          "max_tokens": 4, "logprobs": True, "top_logprobs": 20,
                          "grammar": 'root ::= "A" | "B"\n'})

# --- (2) system-prompt builders are genuinely separate ---
gen_prompt = run_eval.build_system_prompt("", pull_state=False, state_label="", recommend_observe=False)
check("generative system prompt still demands a JSON object",
      "Respond with ONLY a valid JSON object" in gen_prompt)
check("generative system prompt says nothing about a menu/label",
      "label" not in gen_prompt.lower() and "menu" not in gen_prompt.lower())

fc_prompt = run_eval.build_forced_choice_system_prompt("")
check("forced-choice system prompt demands a single label character",
      "single label character" in fc_prompt)
check("forced-choice system prompt never asks for a JSON object",
      "JSON object" not in fc_prompt)

# --- (3) run_session's new parameters default to generative-preserving values ---
sig = inspect.signature(run_eval.run_session)
check("run_session.response_mode defaults to 'generative'",
      sig.parameters["response_mode"].default == "generative")
check("run_session.fc_selection defaults to 'logprobs'",
      sig.parameters["fc_selection"].default == "logprobs")
check("run_session.fc_max_distractors defaults to 3",
      sig.parameters["fc_max_distractors"].default == 3)
check("run_session.fc_top_logprobs defaults to 20",
      sig.parameters["fc_top_logprobs"].default == 20)
check("run_session.fc_degs_dir defaults to None",
      sig.parameters["fc_degs_dir"].default is None)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
