"""Prove classify_failures sees interference on derived gates, not only on the init ladder.

Before 2026-09-12 the asked variable was parsed out of gate prose (`value of ([A-H])`), which
returned None on every rsn_ gate and on syn_final and silently skipped stale-value there. No live
server: synthetic turns_log rows walked against the real degs/rev-2.yaml ladder.

rev-2 values used below: init A=3 B=6 C=3 D=3 E=8 F=5 G=9 H=6; then C→6 (gate 10), F→6 (14),
E→4 (17), F→7 (20), F→4 (24), A→6 (27), C→3 (31).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_failures as cf  # noqa: E402
import e1a_table1  # noqa: E402

DEGS = str(Path(__file__).resolve().parent.parent / "degs")
GATES, HIST = cf.load_ladder("rev-2", DEGS)
POS = {g.gate_id: i for i, g in enumerate(GATES)}

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")


def row(gate_id, given, observe=True):
    """Clear every gate before `gate_id` correctly (observe, commit), then answer it wrong."""
    turns = []
    for _ in range(POS[gate_id]):
        turns += [{"action_parsed": {"action": "observe"}, "engine_text": ""},
                  {"action_parsed": {"action": "commit", "answer": "x"}, "engine_text": "CORRECT"}]
    if observe:
        turns.append({"action_parsed": {"action": "observe"}, "engine_text": ""})
    turns.append({"action_parsed": {"action": "commit", "answer": str(given)}, "engine_text": "WRONG"})
    return {"deg_id": "rev-2", "turns_log": turns}


def classify(gate_id, given, observe=True):
    _, gid, primary, _, found = cf.classify_run(row(gate_id, given, observe), GATES, HIST)["wrongs"][0]
    assert gid == gate_id, (gid, gate_id)
    return primary, found


# --- the manifest, not the prose, names what a gate asks ---
check("every rev-2 gate resolves at least one asked variable",
      all(cf.asked_vars(g) for g in GATES))
check("rsn_1 asks E, C, F, H (the prose regex returned None here)",
      cf.asked_vars(GATES[POS["rsn_1"]]) == ["E", "C", "F", "H"])

# --- stale-answer: reusing the earlier ask's answer at a re-ask ---
p, found = classify("use_2", 6)  # use_1 was C+D = 3+3 = 6; C is now 6, so the answer is 9
check(f"use_2 given use_1's answer → primary stale-answer (got {p})", p == "stale-answer")
check(f"  …and the one-revision overlap stays visible as stale-value[C] (got {found})",
      "stale-value[C]" in found)

# --- stale-value on derived gates, by operand-swap re-evaluation ---
p, found = classify("rsn_2", 12)  # max(C+G, D+H) = max(6+9, 3+6) = 15; with superseded C=3 → 12
check(f"rsn_2 given max(3+9, 3+6) → stale-value[C] (got {found})", found[:1] == ["stale-value[C]"])
p, found = classify("syn_final", 38)  # true 41; with superseded A=3 → 38
check(f"syn_final with A at its old value → stale-value[A] (got {found})", found[:1] == ["stale-value[A]"])
p, found = classify("syn_final", 44)  # C=6 or F=7 both reproduce 44
check(f"syn_final 44 names every operand that reproduces it → stale-value[C,F] (got {found})",
      found[:1] == ["stale-value[C,F]"])

# --- stale-value on a sets gate includes the value held immediately before it ---
p, found = classify("rev_c_1", 3)  # C was 3, the gate states C is now 6
check(f"rev_c_1 given C's previous value → stale-value[C] (got {p}, {found})", p == "stale-value")

# --- the substrate-generic class still wins, unchanged ---
p, found = classify("use_2", 6, observe=False)
check(f"an unobserved commit is unobserved-guess whatever else matches (got {p})", p == "unobserved-guess")

# --- other-var-value and other-wrong ---
p, _ = classify("use_1", 9)  # C+D = 6; 9 is G's current value; no variable has been revised yet
check(f"use_1 given G's current value → other-var-value (got {p})", p == "other-var-value")
p, _ = classify("rsn_1", 6)  # E=8 > C=6 → F=5; 6 is H, the untaken branch
check(f"rsn_1 given the untaken branch's value → other-var-value (got {p})", p == "other-var-value")
p, found = classify("use_1", 77)
check(f"use_1 given 77 → other-wrong with no matches (got {p}, {found})", p == "other-wrong" and not found)

# --- e1a_table1 reads the nested classes dict (it silently summed 0 before) ---
n = e1a_table1.compute_unobserved_guesses({("x", "control"): [row("use_2", 6, observe=False)]}, DEGS)
check(f"e1a_table1 counts the unobserved guess (got {n})", n == {("x", "control"): 1})

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
