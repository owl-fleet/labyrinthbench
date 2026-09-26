"""Deterministic candidate-answer distractor generation for gate menus.

point-and-click chunk 01 (forced-choice arm, nav-3): the harness builds a commit menu of
candidate answers instead of letting the model free-generate one. This module owns the
DISTRACTOR-VALUE generation only — pure and deterministic, no RNG. Menu-level randomization
(option shuffle, label assignment, label-set rotation) is the CALLER's concern
(cli/run_eval.py), kept separate on purpose: the experimenter's one frozen lever (this file,
pre-registered before any row is read — see knowledge/projects/plans/point-and-click/
01-forced-choice-arm-nav3.md) must never be entangled with the harness's independently-seeded
position/label controls.

Rules, tried in this fixed priority order until `n` distinct distractors (all != expected, all
distinct from each other) are collected:
  1. off-by-1  (expected + 1, expected - 1)
  2. off-by-2  (expected + 2, expected - 2)
  3. wrong sign (-expected)
  4. swapped operands — best-effort re-evaluation of a simple binary answer_fn
     (`int(A) <op> int(B)`) with A and B swapped. On nav-3's chain this degenerates to
     `expected` itself for the commutative ops (+, *) — discarded as a non-distractor — and to
     the SAME value as "wrong sign" for subtraction, so it rarely contributes anything NEW on
     this particular DEG. Kept anyway for generality (a future DEG may use non-commutative,
     non-subtractive binary gates where it is genuinely distinct) and as documentation of what
     was considered.
  5. the gate's own most-recent dependency's resolved value ("the prior gate's answer") — the
     literal earlier rung's value, when numeric and distinct from `expected`.
  6. guaranteed-fresh filler offsets (expected +/- 3, 4, 5, ...) so the generator NEVER comes up
     short of `n`, even for a degenerate gate where rules 1-5 collide down to nothing new.

Boolean gates (`expected` in {TRUE, FALSE}) have exactly ONE real opposite value — their
distractor list is that single negation, never padded with typed junk to hit `n`. nav-3 itself
has no boolean gates (it is a pure arithmetic ramp); this branch exists for the OTHER DEGs that
do (`engine/gate_bank.make_boolean_gate`) and for chunk 01's own text, which names both gate
kinds explicitly. Non-numeric, non-boolean gates (e.g. a maintenance-code string gate) return no
distractors — fabricating plausible-looking string junk is out of scope here.

`NONE_OF_THESE` is the sentinel "none of the above" answer string: it fails every branch of
`engine.gate_bank.score_gate` (not true/false, not float-parseable, and not a substring/exact
match of any real gate answer in this codebase), so a commit submitted with this answer always
scores WRONG — the same life-costing consequence as a genuine wrong guess in generative mode,
which is what keeps forced-choice and generative failure semantics comparable.

Self-check (this file's stand-in for a `sandbox/selftest.py`-style proof — LabyrinthBench's own
`selftest` facility lives in `sandbox/` for the unrelated docker permission-puzzle rungs; there is
no equivalent for the DEG/gate engine, so this is that proof for nav-3):

    python3 engine/distractors.py
"""
from __future__ import annotations

import re
from typing import Optional

from .graph import Gate

# Verified against nav-3's own gate set (all numeric) and against TRUE/FALSE — see
# cli/test_distractors.py and cli/test_forced_choice_menu.py. Caveat for future generality beyond
# chunk 01: score_gate's string-gate branch is a SUBSTRING check, so a hypothetical string gate
# whose expected answer were itself one of this sentinel's words (e.g. "of", "these") would
# collide with it. Menus only offer this sentinel for numeric/boolean gates today (see
# generate_distractors' non-numeric branch above), so it does not arise on nav-3, but a future
# string-gated DEG using this same menu-assembly path should re-check before reusing it as-is.
NONE_OF_THESE = "NONE-OF-THESE"


def _is_boolean(expected: str) -> bool:
    return expected.strip().lower() in ("true", "false")


def _resolve_operand(name: str, gate_results: dict, var_ledger: dict) -> Optional[float]:
    v = var_ledger.get(name, gate_results.get(name))
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_BINOP_RE = re.compile(
    r"^\s*(?:int|float)\(\s*(?P<a>\w+)\s*\)\s*(?P<op>[+\-*/])\s*(?:int|float)\(\s*(?P<b>\w+)\s*\)\s*$"
)


def _swapped_operand_value(gate: Gate, gate_results: dict, var_ledger: dict) -> Optional[float]:
    """Best-effort: parse a simple `int(A) <op> int(B)` answer_fn and recompute with A and B
    swapped. Returns None when answer_fn isn't in this exact shape (a plain literal-problem gate
    like c1a/c2a, a >2-operand synthesis gate such as s1, or any non-binary expression) — those
    gates rely on the other rules instead. See the module docstring for why this frequently
    degenerates to a duplicate of another rule on nav-3 specifically."""
    if not gate.answer_fn:
        return None
    m = _BINOP_RE.match(gate.answer_fn)
    if not m:
        return None
    a = _resolve_operand(m.group("a"), gate_results, var_ledger)
    b = _resolve_operand(m.group("b"), gate_results, var_ledger)
    if a is None or b is None:
        return None
    op = m.group("op")
    try:
        if op == "+":
            return b + a
        if op == "-":
            return b - a
        if op == "*":
            return b * a
        if op == "/":
            return (b / a) if a != 0 else None
    except ZeroDivisionError:
        return None
    return None


def _prior_gate_value(gate: Gate, gate_results: dict, var_ledger: dict) -> Optional[float]:
    """'The prior gate's answer' distractor: nav-3's derivative chain (d1..d13) always lists the
    immediately-preceding, textually-unnamed gate FIRST in depends_on and the explicitly-NAMED
    reach-back gate second — e.g. d2: depends_on=[d1, c2a], template 'Add your c2a answer to the
    previous gate's answer': d1 ('the previous gate') is first, c2a (named) is second. Take deps
    in DECLARATION order and use the first one that resolves, so a gate that doesn't follow this
    convention (e.g. s1's synthesis dependencies, which name two parallel-chain endpoints with no
    'previous' slot at all) still degrades gracefully instead of guessing the wrong operand."""
    for name in gate.dep_ids:
        v = _resolve_operand(name, gate_results, var_ledger)
        if v is not None:
            return v
    return None


def generate_distractors(gate: Gate, expected: str, gate_results: dict,
                          var_ledger: Optional[dict] = None, n: int = 3) -> list[str]:
    """Deterministic list of up to `n` wrong-answer strings for this gate, distinct from
    `expected` and from each other. Pure function: no randomness, no I/O."""
    var_ledger = var_ledger or {}
    if _is_boolean(expected):
        other = "FALSE" if expected.strip().upper() == "TRUE" else "TRUE"
        return [other]

    try:
        exp_num = float(expected)
    except ValueError:
        return []  # non-numeric, non-boolean gate — out of scope, no fabricated distractors

    is_int = float(expected).is_integer()

    def _fmt(x: float) -> str:
        return str(int(x)) if is_int and float(x).is_integer() else str(x)

    seen = {expected.strip()}
    out: list[str] = []

    def _try(x: float) -> None:
        if len(out) >= n:
            return
        s = _fmt(x)
        if s not in seen:
            seen.add(s)
            out.append(s)

    _try(exp_num + 1)
    _try(exp_num - 1)
    _try(exp_num + 2)
    _try(exp_num - 2)
    if exp_num != 0:
        _try(-exp_num)
    swapped = _swapped_operand_value(gate, gate_results, var_ledger)
    if swapped is not None:
        _try(swapped)
    prior = _prior_gate_value(gate, gate_results, var_ledger)
    if prior is not None:
        _try(prior)

    offset = 3
    while len(out) < n and offset < 1000:  # pragma: no cover — safety valve, unreachable in practice
        _try(exp_num + offset)
        _try(exp_num - offset)
        offset += 1

    return out


def selftest(deg_path: Optional[str] = None) -> bool:
    """Walk nav-3's canonical (all-correct) solution gate by gate, generating a distractor menu
    at each rung, and assert: (1) the expected answer is always present in its own menu, (2) every
    candidate in a menu is pairwise distinct (no accidental collision with a distractor), (3)
    NONE_OF_THESE never leaks into the generated distractor list itself (it is added by the
    caller's menu assembly, not by this module), and (4) picking the correct candidate at every
    rung still climbs all 20 gates to the exit — i.e. the distractor generator cannot itself
    corrupt a passing run. Doubles as documentation of the reference solution."""
    from pathlib import Path as _Path

    from .graph import load_deg

    path = _Path(deg_path) if deg_path else _Path(__file__).resolve().parent.parent / "degs" / "nav-3.yaml"
    deg = load_deg(path)
    gate_results: dict = {}
    var_ledger: dict = {}
    node_id = deg.start_node_id
    passed = 0
    ok = True
    for _ in range(deg.optimal_commits):
        node = deg.node(node_id)
        if node.terminal:
            break
        path_obj = node.paths[0]
        gate = path_obj.gate
        expected = gate.resolved_answer(gate_results, var_ledger)
        if expected == "__UNRESOLVABLE__":
            print(f"    ! {gate.gate_id}: dependency not yet resolved")
            ok = False
            break
        distractors = generate_distractors(gate, expected, gate_results, var_ledger, n=3)
        menu = [expected] + distractors
        if expected not in menu:
            print(f"    ! {gate.gate_id}: expected {expected!r} missing from its own menu {menu!r}")
            ok = False
        if len(menu) != len(set(menu)):
            print(f"    ! {gate.gate_id}: duplicate candidates in menu {menu!r}")
            ok = False
        if NONE_OF_THESE in distractors:
            print(f"    ! {gate.gate_id}: NONE_OF_THESE leaked into the generated distractor list")
            ok = False
        # Pick the correct candidate (proves the generator doesn't corrupt a passing run).
        if gate.gate_id:
            gate_results[gate.gate_id] = expected
        if gate.sets_var:
            var_ledger[gate.sets_var] = expected
        node_id = path_obj.destination
        passed += 1
    reached_exit = node_id == "exit"
    ok = ok and passed == deg.optimal_commits and reached_exit
    print(f"engine/distractors selftest: {passed}/{deg.optimal_commits} gates, "
          f"{'PASS' if ok else 'FAIL'} (reached {node_id!r})")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
