"""--force-observe-after-wrong (rung iii, MCV chunk 05 — the forced-observe-after-WRONG rule
owed by 04's Decision: "whether an enforced look-after-failure rule ... closes the gap for
wipe-curated-norefresh"). Proves the two pure functions the run_eval.py interceptor is built
from: (1) _forces_observe_next classifies exactly the engine outcomes that mean a gate commit
was scored WRONG and nothing else — in particular no terminal outcome (out_of_lives,
budget_exhausted, impossible) ever forces anything, matching the loop's `if completed: break`
running before there is a next turn; (2) _apply_force_observe unconditionally overrides the
proposed action to plain observe when pending, discarding whatever the model proposed (even an
observe it already chose, or a legal commit at a different gate) — the whole point being to test
the LOOK, not merely to forbid an illegal guess (contrast --look-gate). No live server, no
network, no engine/session dependency — pure functions only, same as context_policy.py's own
message-construction smoke."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import run_eval  # noqa: E402

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")


# --- _forces_observe_next: exactly {"locked", "wrong"} force the next turn; nothing else does ---
for outcome in ("locked", "wrong"):
    check(f"_forces_observe_next({outcome!r}) is True — a gate commit the engine scored WRONG",
          run_eval._forces_observe_next(outcome) is True)

for outcome in ("ok", "exit", "back", "loop_trapped", None):
    check(f"_forces_observe_next({outcome!r}) is False — not a wrong gate commit",
          run_eval._forces_observe_next(outcome) is False)

# Terminal outcomes in particular: the rule must not force a turn that cannot exist. This is a
# behavioural fact about the caller (the loop's `if completed: break` fires before the next
# turn's interceptor could ever consume the pending flag) that this function must still get
# right in isolation, since a stray True here would matter the moment the loop is restructured.
for outcome in ("out_of_lives", "budget_exhausted", "impossible"):
    check(f"_forces_observe_next({outcome!r}) is False — terminal outcome, no next turn to force",
          run_eval._forces_observe_next(outcome) is False)

check("_forces_observe_next is exact, not a substring/prefix match ('lock' != 'locked')",
      run_eval._forces_observe_next("lock") is False)


# --- _apply_force_observe: the pure interceptor core ---

# Not pending: the action passes through completely unchanged (identity, not just equality —
# nothing here should ever rebuild a proposed action it isn't overriding).
commit_action = {"action": "commit", "path_id": "forward", "answer": "12"}
out_action, fired = run_eval._apply_force_observe(commit_action, False)
check("not pending: action passes through unchanged", out_action is commit_action)
check("not pending: fired is False", fired is False)

# Pending, model proposed a commit: unconditionally overridden to plain observe, whatever the
# commit's path_id/answer were — this is the whole mechanism, not a validity check on the guess.
out_action, fired = run_eval._apply_force_observe(
    {"action": "commit", "path_id": "forward", "answer": "99"}, True)
check("pending + commit proposed: overridden to a plain observe", out_action == {"action": "observe"})
check("pending + commit proposed: fired is True", fired is True)

# Pending, model already proposed observe: still counted as fired — the rule is unconditional,
# it does not special-case "the model already agreed" (contrast --look-gate, which only ever
# intercepts an illegal commit and is a no-op on an observe).
out_action, fired = run_eval._apply_force_observe({"action": "observe"}, True)
check("pending + observe already proposed: still returns observe", out_action == {"action": "observe"})
check("pending + observe already proposed: still counts as fired", fired is True)

# Pending, model proposed note: still overridden — the rule does not exempt non-commit,
# non-observe actions either.
out_action, fired = run_eval._apply_force_observe({"action": "note", "text": "left a marker"}, True)
check("pending + note proposed: overridden to observe (note is not exempt)",
      out_action == {"action": "observe"})
check("pending + note proposed: fired is True", fired is True)

# Not pending, model proposed observe: passes through unchanged, not fired — the counter must
# only count turns the rule actually changed something on, not every observe that occurs anyway.
out_action, fired = run_eval._apply_force_observe({"action": "observe"}, False)
check("not pending + observe proposed anyway: not counted as fired", fired is False)


print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
