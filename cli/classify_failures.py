"""classify_failures.py — wrong-answer mechanism classifier over rev-2 results JSONL.

The registered mechanism metric for the look-gate + cohort addendum
(private lab notebook). Promoted from the
2026-07-05 Wali-navigator failure postmortem, which found 46/48 wrong answers across
both arms were UNOBSERVED GUESSES (a commit answering a gate at a node never observed
since arrival, on gates whose answers are stated in the problem text).

For each wrong commit in each run it assigns ONE primary class, first match wins:
  unobserved-guess  — answered a gate at a node not observed since arrival
  stale-answer      — answer equals this question's true answer at an EARLIER ask of it
                      (a re-ask: same answer_fn over the same dependencies) — reusing
                      your own earlier computed answer
  stale-value       — interference: on a stated gate, the answer equals a superseded value
                      of the variable it sets; on a derived gate, re-evaluating answer_fn
                      with ONE depended-on variable at a superseded value reproduces it
  other-var-value   — answer equals the current value of some variable, but is not the true
                      answer (on a derived gate this includes answering one operand, or the
                      untaken branch of a conditional)
  other-wrong       — observed, wrong, none of the above
With --detail, every class a wrong commit matched is listed alongside the primary, so
overlaps stay visible (a re-ask with one revision in between is both stale-answer and
stale-value; stale-answer wins because the re-ask motif exists to provoke it).

Asked variables come from the manifest — a gate's depends_on, else its sets_var — and
answers are evaluated by the engine's own Gate.resolved_answer and compared with its own
score_gate. Before 2026-09-12 they were parsed out of the problem prose, which silently
disabled stale-value on every reasoning and synthesis gate.

Reads the DEG yaml (for the ladder + variable timeline) and one-or-more results JSONL
files; prints per-run rows and per-file aggregates. Read-only.

  python cli/classify_failures.py \
      --deg rev-2 /results/rev2-look-gate-14b.jsonl /results/rev2-control-14b-topup.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.gate_bank import score_gate  # noqa: E402
from engine.graph import Gate  # noqa: E402

CLASSES = ("unobserved-guess", "stale-answer", "stale-value", "other-var-value", "other-wrong")
_UNRESOLVABLE = "__UNRESOLVABLE__"


def _gate(g: dict) -> Gate:
    return Gate(problem=g.get("problem", ""), answer=str(g.get("answer", "")),
                gate_id=g.get("gate_id", "?"), problem_template=g.get("problem_template"),
                answer_fn=g.get("answer_fn"), depends_on=g.get("depends_on"),
                seed=g.get("seed") or {}, sets_var=g.get("sets_var"))


def load_ladder(deg_id: str, degs_dir: str):
    """Walk the DEG from start following gated paths → ordered [Gate] and a per-variable value
    timeline {var: [(ladder_idx, value)]}, bound the way the engine binds its var_ledger."""
    deg = yaml.safe_load(open(os.path.join(degs_dir, f"{deg_id}.yaml")))
    node_by_id = {n["id"]: n for n in deg["nodes"]}
    gates: list[Gate] = []
    cur = deg["nodes"][0]["id"]
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        node = node_by_id.get(cur)
        if not node:
            break
        nxt = None
        for p in node.get("paths", []):
            if p.get("gate"):
                gates.append(_gate(p["gate"]))
                nxt = p.get("destination")
                break
        cur = nxt
    var_history: dict[str, list[tuple[int, str]]] = {}
    ledger: dict[str, str] = {}
    results: dict[str, str] = {}
    for i, g in enumerate(gates):
        ans = _resolve(g, results, ledger)
        results[g.gate_id] = ans
        if g.sets_var:
            ledger[g.sets_var] = ans
            var_history.setdefault(g.sets_var, []).append((i, ans))
    return gates, var_history


def _resolve(gate: Gate, gate_results: dict, ledger: dict) -> str:
    try:
        return gate.resolved_answer(gate_results, ledger)
    except Exception:
        return _UNRESOLVABLE


def asked_vars(gate: Gate) -> list[str]:
    return gate.dep_ids or ([gate.sets_var] if gate.sets_var else [])


def classify_run(row: dict, gates, var_history):
    def hist(var, gi):
        # A gate that sets a variable states its new value, so the binding is current AT that gate.
        return [v for i, v in var_history.get(var, []) if i <= gi]

    def ledger_at(gi):
        return {var: h[-1] for var in var_history if (h := hist(var, gi))}

    def superseded(var, gi):
        h = hist(var, gi)
        return list(dict.fromkeys(v for v in h[:-1] if v != h[-1])) if h else []

    def results_before(gi):
        out, ledger = {}, {}
        for j, g in enumerate(gates[:gi]):
            out[g.gate_id] = _resolve(g, out, ledger)
            if g.sets_var:
                ledger[g.sets_var] = out[g.gate_id]
        return out

    def matches(gi, given):
        gate = gates[gi]
        ledger, results = ledger_at(gi), results_before(gi)
        true = _resolve(gate, results, ledger)

        def hit(cand):
            return cand not in (None, _UNRESOLVABLE) and cand != true and score_gate(given, cand)

        found = []
        if gate.answer_fn and any(
                g.answer_fn == gate.answer_fn and g.dep_ids == gate.dep_ids
                and hit(_resolve(g, results_before(j), ledger_at(j)))
                for j, g in enumerate(gates[:gi])):
            found.append("stale-answer")
        if gate.answer_fn:
            swapped = [v for v in gate.dep_ids if v in var_history and any(
                hit(_resolve(gate, results, {**ledger, v: x})) for x in superseded(v, gi))]
            if swapped:
                found.append(f"stale-value[{','.join(swapped)}]")
        elif gate.sets_var and any(hit(x) for x in superseded(gate.sets_var, gi)):
            found.append(f"stale-value[{gate.sets_var}]")
        if any(hit(val) for val in ledger.values()):
            found.append("other-var-value")
        return found

    gate_idx = 0
    observed_here = True  # control/look-gate both bootstrap an observe
    classes: dict[str, int] = {}
    wrongs: list[tuple] = []  # (1-based ladder pos, gate_id, primary class, given, all matched)
    observes = commits = 0
    for t in row.get("turns_log", []):
        ap = t.get("action_parsed") or {}
        act = ap.get("action")
        etext = t.get("engine_text", "") or ""
        if act == "observe":
            observes += 1
            observed_here = True
        elif act == "commit" and str(ap.get("answer") or "").strip():
            commits += 1
            given = str(ap.get("answer", ""))
            if "WRONG" in etext:
                gid = gates[gate_idx].gate_id if gate_idx < len(gates) else "?"
                found = matches(gate_idx, given) if gate_idx < len(gates) else []
                if not observed_here:
                    found = ["unobserved-guess"] + found
                k = found[0].split("[")[0] if found else "other-wrong"
                classes[k] = classes.get(k, 0) + 1
                wrongs.append((gate_idx + 1, gid, k, given, found))
            elif "CORRECT" in etext:
                gate_idx += 1
                observed_here = False
        elif act == "commit":
            observed_here = False
    return {"observes": observes, "commits": commits, "classes": classes, "wrongs": wrongs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", default="rev-2")
    ap.add_argument("--degs-dir", default=os.environ.get("DEGS_DIR", "/app/degs"))
    ap.add_argument("--label", help="only rows whose run_label matches (for interleaved multi-arm files)")
    ap.add_argument("--detail", action="store_true", help="print each wrong commit's ladder position + gate_id")
    ap.add_argument("files", nargs="+", help="results JSONL file(s)")
    args = ap.parse_args()

    gates, var_history = load_ladder(args.deg, args.degs_dir)
    blind = [g.gate_id for g in gates if not asked_vars(g)]
    print(f"DEG {args.deg}: {len(gates)} gates; init ladder = read-and-echo through the first sets_var run")
    if blind:
        print(f"  {len(blind)} gate(s) ask no variable — only unobserved-guess / other-wrong can fire there: "
              f"{', '.join(blind)}")
    print()
    grand: dict[str, int] = {}
    tot_obs = tot_cmt = 0
    for path in args.files:
        rows = [json.loads(l) for l in open(path) if l.strip() and "error" not in json.loads(l)]
        if args.label:
            rows = [r for r in rows if r.get("run_label") == args.label]
        label_s = f" label={args.label}" if args.label else ""
        print(f"== {os.path.basename(path)}{label_s} ({len(rows)} runs) ==")
        fobs = fcmt = 0
        fclasses: dict[str, int] = {}
        for r in rows:
            st = classify_run(r, gates, var_history)
            fobs += st["observes"]; fcmt += st["commits"]
            for k, v in st["classes"].items():
                fclasses[k] = fclasses.get(k, 0) + v
            lg = r.get("look_gate_interceptions")
            lg_s = f" look_gate_intercepts={lg}" if lg is not None else ""
            print(f"  depth={r.get('ramp_depth'):>2}  obs/cmt={st['observes']}/{st['commits']}"
                  f"  wrong={st['classes']}{lg_s}")
            if args.detail:
                for pos, gid, k, given, found in st["wrongs"]:
                    also_s = f"  also={','.join(found[1:])}" if len(found) > 1 else ""
                    print(f"      gate {pos:>2} ({gid}): {found[0] if found else k}  given={given}{also_s}")
        print(f"  FILE: obs/commit={fobs/max(fcmt,1):.2f}  wrong-classes={fclasses}\n")
        tot_obs += fobs; tot_cmt += fcmt
        for k, v in fclasses.items():
            grand[k] = grand.get(k, 0) + v
    print(f"GRAND: obs/commit={tot_obs/max(tot_cmt,1):.2f}  wrong-classes={grand}")


if __name__ == "__main__":
    main()
