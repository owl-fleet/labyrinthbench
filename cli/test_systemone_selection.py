"""point-and-click chunk 01b: unit tests for the forced-choice arm's systemone selection — the
request built from a real nav-3 menu, the per-label probability extraction, the scoring path's
choice/argmax/fallback order, and the HTTP call against a stubbed client. No network, no GPU."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402

FAILS = 0


def check(d, ok):
    global FAILS
    FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")


NAV3 = run_eval.load_deg(Path(__file__).resolve().parent.parent / "degs" / "nav-3.yaml")
LABELS = run_eval.FC_LABEL_SETS[1]
state = run_eval._FCState(deg=NAV3, current_node_id="start")
menu = run_eval._fc_build_menu(state, random.Random(42), LABELS, max_distractors=3)
sys_prompt = run_eval.build_forced_choice_system_prompt("")
msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "You are at START."}]

# --- _fc_systemone_request: shape, shuffle preserved, menu not in the state ---
req = run_eval._fc_systemone_request("decider", msgs, menu)
q = req["questions"][run_eval.FC_SYSTEMONE_QUESTION]
check("request carries model, state, questions", set(req) == {"model", "state", "questions"})
check("one choice question", q["type"] == "choice")
check("criteria keys are the menu labels in the menu's shuffled order", list(q["criteria"]) == menu.labels)
check("criteria text is the rendered menu's wording",
      all(q["criteria"][l] == run_eval._fc_option_desc(menu.options[l].action) for l in menu.labels))
check("state holds the user turn, not the system prompt",
      "You are at START." in req["state"]["context"] and sys_prompt not in req["state"]["context"])
check("state does not carry the rendered menu", "CHOOSE ONE" not in req["state"]["context"])
check("instructions drop the reply-with-one-character tail", run_eval._FC_JSON_TAIL not in q["instructions"])
check("instructions keep the objective", "reach EXIT" in q["instructions"])

# --- _fc_extract_systemone_probs ---
n = len(menu.labels)
as_dict = {l: (i + 1) for i, l in enumerate(menu.labels)}
pv = run_eval._fc_extract_systemone_probs({"probabilities": as_dict}, menu.labels)
check("dict probabilities -> normalized vector over every label",
      pv is not None and set(pv) == set(menu.labels) and abs(sum(pv.values()) - 1) < 1e-9)
pv_list = run_eval._fc_extract_systemone_probs({"probabilities": list(range(1, n + 1))}, menu.labels)
check("list probabilities in criteria order -> same vector as the dict form",
      pv_list is not None and all(abs(pv_list[l] - pv[l]) < 1e-9 for l in menu.labels))
check("a dict missing a label -> None",
      run_eval._fc_extract_systemone_probs({"probabilities": {menu.labels[0]: 1.0}}, menu.labels) is None)
check("a list of the wrong length -> None",
      run_eval._fc_extract_systemone_probs({"probabilities": [1.0]}, menu.labels) is None)
check("non-numeric values -> None",
      run_eval._fc_extract_systemone_probs({"probabilities": {l: "x" for l in menu.labels}}, menu.labels) is None)
check("no probabilities -> None", run_eval._fc_extract_systemone_probs({}, menu.labels) is None)


def resp(answer, latency=12.5):
    return {"choices": [{"message": {"content": str(answer.get("choice") or "")}, "finish_reason": "stop"}],
            "systemone": {"answer": answer, "latency_ms": latency}}


# --- _fc_score_turn(selection="systemone") ---
correct_label = next(l for l in menu.labels if menu.options[l].is_correct)
probs = {l: (0.9 if l == correct_label else 0.1 / (n - 1)) for l in menu.labels}
s = run_eval._fc_score_turn(resp({"choice": correct_label, "probabilities": probs, "confidence": 0.9}),
                            menu, "systemone")
check("the model's choice is the pick", s["label"] == correct_label and s["option"].is_correct)
check("method is 'systemone'", s["selection_method"] == "systemone")
check("confidence and latency are carried", s["decision_confidence"] == 0.9 and s["decision_latency_ms"] == 12.5)
check("margin is computed from the vector", s["top1_margin"] is not None and s["top1_margin"] > 0.8)

other = next(l for l in menu.labels if l != correct_label)
s = run_eval._fc_score_turn(resp({"choice": other, "probabilities": probs}), menu, "systemone")
check("choice wins over the argmax when both are present", s["label"] == other)

s = run_eval._fc_score_turn(resp({"choice": "zz", "probabilities": probs}), menu, "systemone")
check("a non-label choice falls back to the argmax", s["label"] == correct_label and s["selection_method"] == "systemone")

s = run_eval._fc_score_turn(resp({"choice": correct_label.lower()}), menu, "systemone")
check("a lower-case label choice is accepted, no vector needed",
      s["label"] == correct_label and s["probability_vector"] is None)

s = run_eval._fc_score_turn(resp({}), menu, "systemone")
check("an empty answer falls to the visible undecodable default",
      s["label"] == menu.labels[0] and s["selection_method"] == "systemone-unparsed-text-fallback-undecodable-default")

s = run_eval._fc_score_turn(resp({"choice": correct_label, "confidence": "high"}), menu, "systemone")
check("a non-numeric confidence is recorded as None, not a crash", s["decision_confidence"] is None)

# --- logprobs/grammar paths still return the new keys as None (row schema stays uniform) ---
s = run_eval._fc_score_turn({"choices": [{"message": {"content": menu.labels[2]}}]}, menu, "grammar")
check("grammar path: label read from text, decision fields None",
      s["label"] == menu.labels[2] and s["decision_confidence"] is None and s["decision_latency_ms"] is None)


# --- _fc_systemone_call against a stubbed client ---
class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise run_eval.httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self.body


class _Client:
    def __init__(self, body, status=200):
        self.base_url = "http://decider.test/v1"
        self.calls = []
        self._body, self._status = body, status

    def post(self, url, json=None, **kw):
        self.calls.append((url, json))
        return _Resp(self._body, self._status)


body = {"model": "decider", "answers": {run_eval.FC_SYSTEMONE_QUESTION: {"choice": correct_label, "probabilities": probs,
                                                                          "confidence": 0.9}},
        "usage": {"prompt_tokens": 100}}
c = _Client(body)
out = run_eval._fc_systemone_call(c, "decider", msgs, menu)
check("posts to <base_url>/systemone", c.calls[0][0] == "http://decider.test/v1/systemone")
check("posts the request the builder makes", c.calls[0][1] == run_eval._fc_systemone_request("decider", msgs, menu))
check("returns an OpenAI-shaped dict the turn loop can read",
      out["choices"][0]["message"]["content"] == correct_label and out["choices"][0]["finish_reason"] == "stop")
check("usage passes through", out["usage"] == {"prompt_tokens": 100})
check("latency is measured", isinstance(out["systemone"]["latency_ms"], float))
check("end to end: the scored pick is the correct option",
      run_eval._fc_score_turn(out, menu, "systemone")["option"].is_correct is True)

try:
    run_eval._fc_systemone_call(_Client(body, status=500), "decider", msgs, menu)
    check("an HTTP error raises (recorded as an error row)", False)
except run_eval.httpx.HTTPStatusError:
    check("an HTTP error raises (recorded as an error row)", True)

out = run_eval._fc_systemone_call(_Client({"answers": {}}), "decider", msgs, menu)
check("a response without our question decodes to an empty choice, not a crash",
      out["choices"][0]["message"]["content"] == "")

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILURE(S)'}")
sys.exit(1 if FAILS else 0)
