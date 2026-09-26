"""point-and-click chunk 01: prove engine/distractors.py's candidate generator (1) never produces
a distractor equal to the expected answer or to another distractor, (2) generates the documented
rule set (off-by-N, wrong sign, swapped operands, the prior gate's value, guaranteed-fresh filler)
in its documented priority order, where derivable, (3) degrades to boolean's single negation and
to zero distractors for a non-numeric gate instead of fabricating junk, and (4) the canonical
nav-3 solve still reaches 20/20 with distractors present (the module's own `selftest()`, re-run
here so a pytest-less `cli/test_*.py` sweep catches a regression too). No network, no GPU.

Every expected list below was hand-traced against the generator's fixed rule order (see the
module docstring) rather than just asserting properties — the priority order is exactly the part
a future refactor is most likely to silently reorder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.distractors import NONE_OF_THESE, generate_distractors, selftest  # noqa: E402
from engine.graph import Gate  # noqa: E402

FAILS = 0


def check(d, ok):
    global FAILS
    FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")


# --- basic arithmetic gate (nav-3's c1a: "Calculate 3 + 4" -> answer "7"), no dependencies ---
g_c1a = Gate(problem="Calculate 3 + 4", answer="7", gate_id="c1a")

d3 = generate_distractors(g_c1a, "7", {}, {}, n=3)
# Trace: off-by-1 up=8, down=6 (2 items) -> off-by-2 up=9 fills the 3rd slot; -2=5, wrong-sign=-7,
# swap (no answer_fn) and prior (no deps) never fire because n=3 is already full.
check("c1a n=3: exact traced order [8, 6, 9]", d3 == ["8", "6", "9"])
check("c1a n=3: expected (7) never appears among its own distractors", "7" not in d3)
check("c1a n=3: all distractors pairwise distinct", len(d3) == len(set(d3)))

d5 = generate_distractors(g_c1a, "7", {}, {}, n=5)
# Trace: 8, 6, 9, 5 (off-by-1/2) then wrong-sign -7 fills the 5th slot.
check("c1a n=5: exact traced order [8, 6, 9, 5, -7]", d5 == ["8", "6", "9", "5", "-7"])

# --- zero-valued expected answer: wrong-sign (-0 == 0) must be skipped, exercising the
#     guaranteed-fresh FILLER rule (offset >= 3) to reach n=5 ---
g_zero = Gate(problem="Calculate 5 - 5", answer="0", gate_id="zero1")
d_zero = generate_distractors(g_zero, "0", {}, {}, n=5)
# Trace: 1, -1, 2, -2 (off-by-1/2, 4 items) -> wrong-sign skipped (exp_num == 0) -> no answer_fn
# (swap) -> no deps (prior) -> filler offset=3: +3 fills the 5th slot.
check("zero-expected n=5: filler rule reached, traced order [1, -1, 2, -2, 3]",
      d_zero == ["1", "-1", "2", "-2", "3"])
check("zero-expected: wrong-sign (-0) never appears as a distinct-looking distractor", "-0" not in d_zero)

# --- dependent chain gate (nav-3's real d5: depends_on=[d4, c1c],
#     "Subtract your c1c answer from the previous gate's answer" ->
#     answer_fn "int(d4) - int(c1c)"; d4 is the unnamed 'previous gate', listed first) ---
g_d5 = Gate(problem="", answer="",
            problem_template="Subtract your c1c answer from the previous gate's answer",
            answer_fn="int(d4) - int(c1c)", depends_on=["d4", "c1c"], gate_id="d5")
gate_results_d5 = {"d4": "30", "c1c": "18"}
expected_d5 = g_d5.resolved_answer(gate_results_d5, {})
check("d5: resolved_answer via answer_fn is 12 (30 - 18)", expected_d5 == "12")

d_d5_n6 = generate_distractors(g_d5, expected_d5, gate_results_d5, {}, n=6)
# Trace: 13, 11, 14, 10 (off-by-1/2, 4 items) -> wrong-sign -12 (5th) -> swap (c1c - d4 = 18 - 30
# = -12, already `seen`, discarded as a duplicate of wrong-sign — the documented degeneracy) ->
# prior-gate's-value = d4's own resolved value (30, FIRST dep, the unnamed 'previous gate') fills
# the 6th slot.
check("d5 n=6: exact traced order [13, 11, 14, 10, -12, 30]",
      d_d5_n6 == ["13", "11", "14", "10", "-12", "30"])
check("d5 n=6: the subtraction swap-value (-12) is not duplicated (dedup against wrong-sign)",
      d_d5_n6.count("-12") == 1)
check("d5 n=6: expected (12) absent from its distractors", "12" not in d_d5_n6)

# --- synthesis gate: >1 dep, commutative op, no single 'previous gate' slot (nav-3's real s1:
#     depends_on=[c1c, c2c], "Add your c1c answer to your c2c answer" -> "int(c1c) + int(c2c)") ---
g_s1 = Gate(problem="", answer="",
            problem_template="Add your c1c answer to your c2c answer",
            answer_fn="int(c1c) + int(c2c)", depends_on=["c1c", "c2c"], gate_id="s1")
gate_results_s1 = {"c1c": "18", "c2c": "22"}
expected_s1 = g_s1.resolved_answer(gate_results_s1, {})
check("s1: resolved_answer via answer_fn is 40 (18 + 22)", expected_s1 == "40")

d_s1_n6 = generate_distractors(g_s1, expected_s1, gate_results_s1, {}, n=6)
# Trace: 41, 39, 42, 38 (off-by-1/2) -> wrong-sign -40 (5th) -> swap on a commutative + reproduces
# the expected value itself (22 + 18 == 18 + 22 == 40) and is correctly self-collision-discarded,
# contributing NOTHING new -> prior-gate's-value = c1c (FIRST dep, 18) fills the 6th slot.
check("s1 n=6: exact traced order [41, 39, 42, 38, -40, 18]",
      d_s1_n6 == ["41", "39", "42", "38", "-40", "18"])
check("s1 n=6: the commutative swap-value (40, == expected) never appears — correctly discarded",
      "40" not in d_s1_n6)

# --- boolean gates: exactly one real negation, never padded to n ---
g_bool_t = Gate(problem="Evaluate: TRUE AND TRUE", answer="TRUE", gate_id="bt1")
check("boolean TRUE: exactly one distractor (FALSE), not padded to n=3",
      generate_distractors(g_bool_t, "TRUE", {}, {}, n=3) == ["FALSE"])
g_bool_f = Gate(problem="Evaluate: NOT TRUE", answer="FALSE", gate_id="bt2")
check("boolean FALSE: exactly one distractor (TRUE), not padded to n=5",
      generate_distractors(g_bool_f, "FALSE", {}, {}, n=5) == ["TRUE"])

# --- non-numeric, non-boolean gate: no fabricated junk ---
g_str = Gate(problem="Recite the maintenance code", answer="SIGMA-8", gate_id="code1")
check("string gate: zero distractors (out of scope, no fabrication)",
      generate_distractors(g_str, "SIGMA-8", {}, {}, n=3) == [])

# --- NONE_OF_THESE is a menu-assembly sentinel, never emitted BY the generator itself ---
check("NONE_OF_THESE never appears in any generated list above",
      all(NONE_OF_THESE not in lst
          for lst in (d3, d5, d_zero, d_d5_n6, d_s1_n6)))

# --- the canonical nav-3 walkthrough (this module's own selftest — the distractor generator
#     must not itself break a 20/20 solve when distractors are present) ---
check("nav-3 canonical solution still reaches 20/20 with distractors present", selftest())

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
