"""point-and-click chunk 01: unit tests for the forced-choice arm's menu building, logprobs
extraction/argmax scoring, grammar-fallback text parsing, and local-DEG-mirror dispatch update
(cli/run_eval.py's _fc_* helpers). No network, no GPU — all httpx/engine objects are stubbed or
loaded straight from the real degs/nav-3.yaml (read-only, no API server involved)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.gate_bank import score_gate  # noqa: E402

FAILS = 0


def check(d, ok):
    global FAILS
    FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")


NAV3 = run_eval.load_deg(Path(__file__).resolve().parent.parent / "degs" / "nav-3.yaml")
LABELS = run_eval.FC_LABEL_SETS[0]

# --- _fc_build_menu: structure, determinism, back-option gating ---
state = run_eval._FCState(deg=NAV3, current_node_id="start")
menu1 = run_eval._fc_build_menu(state, random.Random(42), LABELS, max_distractors=3)
menu2 = run_eval._fc_build_menu(state, random.Random(42), LABELS, max_distractors=3)
check("same seed -> identical menu labels (deterministic shuffle)", menu1.labels == menu2.labels)
check("same seed -> identical menu options",
      [menu1.options[l].action for l in menu1.labels] == [menu2.options[l].action for l in menu2.labels])
check("start node menu has 6 options (observe + [correct,8,6,9] + none-of-these; no back at turn 0)",
      len(menu1.labels) == 6)
commit_answers = sorted(o.action["answer"] for o in menu1.options.values() if o.action["action"] == "commit")
check("commit answers are exactly {6, 7, 8, 9, NONE_OF_THESE}",
      set(commit_answers) == {"6", "7", "8", "9", run_eval.NONE_OF_THESE})
correct = [o for o in menu1.options.values() if o.is_correct]
check("exactly one option is marked correct, and it is answer=7", len(correct) == 1 and correct[0].action["answer"] == "7")
observe_opts = [o for o in menu1.options.values() if o.action["action"] == "observe"]
check("exactly one observe option", len(observe_opts) == 1)
none_opts = [o for o in menu1.options.values() if o.action.get("answer") == run_eval.NONE_OF_THESE]
check("exactly one none-of-these option, and it is marked incorrect", len(none_opts) == 1 and none_opts[0].is_correct is False)

# n1's gate (c1b = int(c1a) + 5) depends on c1a, so gate_results must carry it forward for
# resolved_answer to resolve at all — mirrors an actual mid-run mirror state, not a fresh one.
state_with_back = run_eval._FCState(deg=NAV3, current_node_id="n1", traversal_stack=["start"],
                                     gate_results={"c1a": "7"})
menu3 = run_eval._fc_build_menu(state_with_back, random.Random(7), LABELS, max_distractors=3)
check("back option appears once traversal_stack is non-empty",
      any(o.action.get("path_id") == "back" for o in menu3.options.values()))
check("menu grows to 7 options with back added (observe + [12,13,11,14] + none + back)",
      len(menu3.labels) == 7)

# --- _fc_build_menu: an UNRESOLVABLE dependency degrades to a single placeholder, never crashes
#     (should not arise on nav-3's strictly-ordered chain, but the guard must hold) ---
state_unresolvable = run_eval._FCState(deg=NAV3, current_node_id="n1")  # gate_results empty: c1a missing
menu_unresolvable = run_eval._fc_build_menu(state_unresolvable, random.Random(3), LABELS, max_distractors=3)
check("unresolvable dependency: degrades to observe + one placeholder (no back, no candidates)",
      len(menu_unresolvable.labels) == 2)
placeholder = [o for o in menu_unresolvable.options.values() if o.action["action"] == "commit"][0]
check("unresolvable dependency: the placeholder is NONE_OF_THESE and marked incorrect",
      placeholder.action["answer"] == run_eval.NONE_OF_THESE and placeholder.is_correct is False)

# --- _fc_render_menu: sanity on the rendered text ---
text = run_eval._fc_render_menu(menu1)
check("render includes the CHOOSE ONE header", "--- CHOOSE ONE ---" in text)
check("render shows the correct candidate's value", "answer = 7" in text)
check("render names the none-of-these option in plain words", "none of the candidate answers is correct" in text)
check("render names the observe option in plain words", "observe (look around" in text)

# --- _fc_append_menu: never mutates the caller's list, appends to the last user turn ---
call_messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "OBS"}]
appended = run_eval._fc_append_menu(call_messages, "MENU-TEXT")
check("append_menu: caller's original list is untouched", call_messages[1]["content"] == "OBS")
check("append_menu: new list's last user turn is OBS + menu", appended[-1]["content"] == "OBS\nMENU-TEXT")
check("append_menu: system message carried over unchanged", appended[0] == {"role": "system", "content": "SYS"})

# --- _fc_build_grammar ---
g = run_eval._fc_build_grammar(["A", "B", "C"])
check("grammar: a single root rule", g.strip().startswith("root ::= ("))
check("grammar: every label appears as a quoted alternative", all(f'"{l}"' in g for l in ("A", "B", "C")))

# --- _fc_extract_label_probs ---
llm_json_lp = {"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{
    "token": "B", "logprob": -0.1,
    "top_logprobs": [{"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.0}, {"token": "C", "logprob": -3.0}],
}]}}]}
probs = run_eval._fc_extract_label_probs(llm_json_lp, ["A", "B", "C"])
check("extract: all 3 labels present", probs is not None and set(probs) == {"A", "B", "C"})
check("extract: probabilities sum to ~1.0", abs(sum(probs.values()) - 1.0) < 1e-9)
check("extract: the highest-logprob label (B) has the highest probability", probs["B"] == max(probs.values()))

llm_json_partial = {"choices": [{"message": {"content": "A"}, "logprobs": {"content": [{
    "token": "A", "logprob": -0.05, "top_logprobs": [{"token": "A", "logprob": -0.05}],
}]}}]}
probs2 = run_eval._fc_extract_label_probs(llm_json_partial, ["A", "B", "C", "D"])
check("extract: labels missing from top_logprobs still occupy the vector at a nonzero floor",
      probs2 is not None and set(probs2) == {"A", "B", "C", "D"} and all(p > 0 for p in probs2.values()))
check("extract: the one truly-observed label dominates", probs2["A"] == max(probs2.values()))

llm_json_none = {"choices": [{"message": {"content": "C"}}]}  # no logprobs block at all
check("extract: no logprobs block -> None (never raises)",
      run_eval._fc_extract_label_probs(llm_json_none, ["A", "B", "C"]) is None)

# --- _fc_score_turn ---
opt_a = run_eval._FCOption(action={"action": "commit", "path_id": "forward", "answer": "6"}, gate_id="c1a", is_correct=False)
opt_b = run_eval._FCOption(action={"action": "commit", "path_id": "forward", "answer": "7"}, gate_id="c1a", is_correct=True)
opt_c = run_eval._FCOption(action={"action": "commit", "path_id": "forward", "answer": "8"}, gate_id="c1a", is_correct=False)
menu_abc = run_eval._FCMenu(labels=["A", "B", "C"], options={"A": opt_a, "B": opt_b, "C": opt_c})

scored = run_eval._fc_score_turn(llm_json_lp, menu_abc, "logprobs")
check("score_turn: picks the highest-logprob label (B)", scored["label"] == "B")
check("score_turn: resolves to the CORRECT option (answer=7)", scored["option"].action["answer"] == "7")
check("score_turn: selection_method is 'logprobs'", scored["selection_method"] == "logprobs")
check("score_turn: option_count == 3", scored["option_count"] == 3)
check("score_turn: top1_margin is positive", scored["top1_margin"] is not None and scored["top1_margin"] > 0)

scored_grammar = run_eval._fc_score_turn({"choices": [{"message": {"content": "C is my pick"}}]}, menu_abc, "grammar")
check("score_turn (grammar): parses the leading label character from raw text", scored_grammar["label"] == "C")
check("score_turn (grammar): no probability vector on this path", scored_grammar["probability_vector"] is None)
check("score_turn (grammar): selection_method reports 'grammar'", scored_grammar["selection_method"] == "grammar")

scored_fallback = run_eval._fc_score_turn(llm_json_none, menu_abc, "logprobs")
check("score_turn: logprobs requested but unavailable -> falls back to text parse (picks C, "
      "the raw content of llm_json_none)", scored_fallback["label"] == "C")
check("score_turn: fallback is visible in selection_method",
      scored_fallback["selection_method"] == "logprobs-unavailable-text-fallback")

scored_garbage = run_eval._fc_score_turn({"choices": [{"message": {"content": "???"}}]}, menu_abc, "grammar")
check("score_turn: undecodable text defaults to the menu's first label instead of crashing",
      scored_garbage["label"] == menu_abc.labels[0])
check("score_turn: the default is flagged in selection_method", "undecodable-default" in scored_garbage["selection_method"])

# --- _fc_apply_dispatch: mirrors engine/runner.py's Session.commit() state machine ---
st_correct = run_eval._FCState(deg=NAV3, current_node_id="start")
run_eval._fc_apply_dispatch(
    st_correct, {"action": "commit", "path_id": "forward", "answer": "7"},
    {"outcome": "ok", "node_id": "n1", "gate_id": "c1a"})
check("apply_dispatch (correct): advances to the engine's reported node_id", st_correct.current_node_id == "n1")
check("apply_dispatch (correct): records the gate result", st_correct.gate_results.get("c1a") == "7")
check("apply_dispatch (correct): pushes the OLD node onto the traversal stack", st_correct.traversal_stack == ["start"])

st_wrong = run_eval._FCState(deg=NAV3, current_node_id="start")
run_eval._fc_apply_dispatch(
    st_wrong, {"action": "commit", "path_id": "forward", "answer": run_eval.NONE_OF_THESE},
    {"outcome": "locked", "node_id": "start"})
check("apply_dispatch (locked/wrong): stays put", st_wrong.current_node_id == "start")
check("apply_dispatch (locked/wrong): gate_results untouched", st_wrong.gate_results == {})
check("apply_dispatch (locked/wrong): traversal_stack untouched", st_wrong.traversal_stack == [])

st_back = run_eval._FCState(deg=NAV3, current_node_id="n1", traversal_stack=["start"])
run_eval._fc_apply_dispatch(
    st_back, {"action": "commit", "path_id": "back", "answer": ""}, {"outcome": "back", "node_id": "start"})
check("apply_dispatch (back): reverts to the engine's reported node_id", st_back.current_node_id == "start")
check("apply_dispatch (back): pops the traversal stack", st_back.traversal_stack == [])

st_observe = run_eval._FCState(deg=NAV3, current_node_id="start")
run_eval._fc_apply_dispatch(st_observe, {"action": "observe", "path_id": "", "answer": ""}, {"outcome": "ok", "node_id": "start"})
check("apply_dispatch (observe): a no-op — node/gate_results/traversal_stack all unchanged",
      st_observe.current_node_id == "start" and st_observe.gate_results == {} and st_observe.traversal_stack == [])

# --- NONE_OF_THESE always scores WRONG against the real score_gate (not just against our own
#     stub distractor generator — the actual engine-side scorer) ---
check("NONE_OF_THESE fails score_gate against a numeric expected", score_gate(run_eval.NONE_OF_THESE, "7") is False)
check("NONE_OF_THESE fails score_gate against boolean TRUE", score_gate(run_eval.NONE_OF_THESE, "TRUE") is False)
check("NONE_OF_THESE fails score_gate against boolean FALSE", score_gate(run_eval.NONE_OF_THESE, "FALSE") is False)

# --- _fc_expected_calibration_error ---
ece = run_eval._fc_expected_calibration_error([(0.9, True), (0.9, True), (0.1, False), (0.1, False)])
check("ECE: a symmetric miscalibration-by-0.1 example yields exactly 0.1", ece is not None and abs(ece - 0.1) < 1e-9)
check("ECE: no records -> None", run_eval._fc_expected_calibration_error([]) is None)
perfect = run_eval._fc_expected_calibration_error([(1.0, True)] * 5 + [(0.0, False)] * 5)
check("ECE: perfectly-calibrated example yields ~0", perfect is not None and abs(perfect) < 1e-9)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
