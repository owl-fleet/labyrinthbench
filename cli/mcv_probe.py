#!/usr/bin/env python3
"""MCV probe — marginal-context-value fork runner (rung i: post-hoc h=1).

Replays decision points minted by `run_eval.py --dump-context` against context-strategy
variants and scores each fork's next-gate answer against DEG ground truth. No engine
interaction: the corpus IS the state; ground truth is computed from the DEG yaml chain.

Modes (plan: knowledge/projects/plans/marginal-context-value/):
  This script is MEASUREMENT mode at horizon 1 only. On-policy trajectories and
  optimization/beam mode are later chunks — do not bolt them on here.

Usage (from the sandbox container):
  python cli/mcv_probe.py --corpus /results/mcv/corpus-r0.jsonl \
      --deg degs/nav-3.yaml --base-url http://192.168.0.11:11434/v1 \
      --model qwen3:14b --output /results/mcv/probe-r0.jsonl [--strategies s0,s1,...]
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import httpx
import yaml

# ---------------------------------------------------------------- ground truth

def gate_chain(deg_path: str) -> list[dict]:
    """Ordered [(gate_id, problem_text, answer)] along the corridor, answer_fn resolved."""
    deg = yaml.safe_load(Path(deg_path).read_text())
    answers: dict[str, str] = {}
    chain = []
    for node in deg["nodes"]:
        for path in node.get("paths", []):
            gate = path.get("gate")
            if not gate:
                continue
            gid = gate["gate_id"]
            if "answer" in gate:
                ans = str(gate["answer"])
            else:
                # answer_fn like "int(c1b) + 6" — evaluate against earlier answers only.
                ans = str(eval(gate["answer_fn"], {"__builtins__": {"int": int, "abs": abs}}, dict(answers)))
            answers[gid] = ans
            chain.append({"gate_id": gid,
                          "problem": gate.get("problem") or gate.get("problem_template") or "",
                          "answer": ans})
    return chain


def gate_for_turn(rec: dict, chain: list[dict]) -> dict | None:
    """Identify which gate this decision point faces.

    Fresh exposure: the gate's problem text appears verbatim in the LAST user (engine)
    message — the paths listing from observe() restates it (linear corridor => problems
    are unique, so the match is unambiguous).

    Retry/gate-failure turns (instrument debt, chunk 01 RESULTS): a wrong LOCK answer does
    not open the gate, and the engine's feedback ("--- LOCKED ---\nGate answer: WRONG —
    the gate does not open.") never restates the problem text — the node's flavor
    `description` line (e.g. "Gate 3.") is NOT the gate's `problem` ("Add 6 to your c1b
    answer"). The old substring match therefore always missed retries, marking every one
    un-probe-able — exactly the turns mainlines die on (s0_identity scored 92% on
    probe-able turns yet the mainlines died at gates ~7-8; see 01-corpus-and-probe-mvp.md).

    Fixed by falling back to the DEG's LINEAR CHAIN POSITION on a LOCKED turn: nav-3 is a
    strict corridor with LOCK-only gates (no wrong_destination routing — see nav-3.yaml),
    so a locked-retry turn is, by construction, still facing the first unsolved gate in
    chain order. solved_ledger() already derives that position deterministically from
    committed-CORRECT history in the same call_messages — no LLM call, no engine
    interaction, consistent with the frozen-decision-point design (measurement mode only).

    This intentionally does NOT fire on pure exploration/backtrack turns (no LOCKED marker,
    no problem text) — those stay un-probe-able, same as before, to avoid misattributing a
    wandering turn to a stale already-solved (or not-yet-seen) gate. Do not generalize the
    LOCKED fallback to routing DEGs (wrong_destination gates render a different "--- WRONG
    ---" header and actually move the model) without revisiting this — chain position is
    only exact ground truth for a pure lock corridor."""
    last_user = next((m["content"] for m in reversed(rec["call_messages"]) if m["role"] == "user"), "")
    for g in chain:
        if g["problem"] and g["problem"] in last_user:
            return g
    if "--- LOCKED ---" in last_user:
        solved = solved_ledger(rec["call_messages"], chain)
        if len(solved) < len(chain):
            return chain[len(solved)]
    return None

# ---------------------------------------------------------------- ledger extraction

_COMMIT_RE = re.compile(r'"action"\s*:\s*"commit".*?"answer"\s*:\s*"?([\w.\-]+)"?', re.S)


def solved_ledger(messages: list[dict], chain: list[dict]) -> list[tuple[str, str]]:
    """Committed-CORRECT answers in corridor order. An assistant commit followed by an
    engine 'Gate answer: CORRECT' is a solve; nav-3 is linear so solves map onto the
    chain prefix in order."""
    solves = 0
    pairs = []
    msgs = messages
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if not _COMMIT_RE.search(m["content"] or ""):
            continue
        nxt = next((x["content"] for x in msgs[i + 1:] if x["role"] == "user"), "")
        if "Gate answer: CORRECT" in nxt:
            if solves < len(chain):
                pairs.append((chain[solves]["gate_id"], chain[solves]["answer"]))
            solves += 1
    return pairs


def ledger_line(pairs: list[tuple[str, str]]) -> str:
    return "recall = " + " | ".join(f"{g}={a}" for g, a in pairs) if pairs else "recall = (none yet)"

# ---------------------------------------------------------------- strategies
# Each strategy: (messages) -> messages. Channels available in the accumulating arm:
# system prompt | engine user turns (dominant noise: map/room/gate blocks) | assistant
# final texts (actions). Reasoning never re-enters context via /v1, so it is NOT a channel.

def s0_identity(msgs, chain):
    return msgs


def s1_ledger_only(msgs, chain):
    """Wiped-curated approximation: system + recall ledger + current engine text."""
    sys_p = msgs[0]
    last_user = next((m for m in reversed(msgs) if m["role"] == "user"), None)
    lg = ledger_line(solved_ledger(msgs, chain))
    return [sys_p, {"role": "user", "content": lg + "\n\n" + (last_user["content"] if last_user else "")}]


def s2_drop_old_engine(msgs, chain):
    """Keep system + ALL assistant turns + only the LAST engine turn (kills observation repetition)."""
    keep = [msgs[0]] + [m for m in msgs[1:-1] if m["role"] == "assistant"]
    return keep + [msgs[-1]]


def s3_window3(msgs, chain):
    """System + last 3 user/assistant exchanges (recency window)."""
    return [msgs[0]] + msgs[max(1, len(msgs) - 6):]


def s4_ledger_plus_full(msgs, chain):
    """ADDITIVE rescue: full noisy history + the recall ledger appended to the current turn.
    Does adding the concentrate rescue without any removal?"""
    lg = ledger_line(solved_ledger(msgs, chain))
    out = [dict(m) for m in msgs]
    out[-1] = {"role": "user", "content": lg + "\n\n" + out[-1]["content"]}
    return out


def s5_shuffled_history(msgs, chain):
    """Content-position control: same tokens, shuffled middle exchanges (seeded)."""
    if len(msgs) <= 4:
        return msgs
    mid = msgs[1:-1]
    rng = random.Random(1337)
    pairs = [mid[i:i + 2] for i in range(0, len(mid) - 1, 2)]
    rng.shuffle(pairs)
    flat = [m for p in pairs for m in p]
    return [msgs[0]] + flat + [msgs[-1]]


STRATEGIES = {
    "s0_identity": s0_identity,
    "s1_ledger_only": s1_ledger_only,
    "s2_drop_old_engine": s2_drop_old_engine,
    "s3_window3": s3_window3,
    "s4_ledger_plus_full": s4_ledger_plus_full,
    "s5_shuffled_history": s5_shuffled_history,
}

# ---------------------------------------------------------------- fork execution

_ACTION_RE = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.S)


def parse_action(text: str) -> dict | None:
    m = None
    for m in _ACTION_RE.finditer(text or ""):
        pass  # last JSON-looking action block wins, matching run_eval's forgiving posture
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_llm(client: httpx.Client, model: str, messages: list[dict]) -> dict:
    r = client.post("/chat/completions",
                    json={"model": model, "messages": messages, "stream": False, "temperature": 0})
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--deg", required=True, help="DEG yaml path (ground-truth chain)")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--turns", default="", help="optional 'run:turn,run:turn' filter")
    ap.add_argument("--limit", type=int, default=0, help="stop after N fork calls (smoke)")
    args = ap.parse_args()

    chain = gate_chain(args.deg)
    strategies = {k: STRATEGIES[k] for k in args.strategies.split(",")}
    only = set(args.turns.split(",")) if args.turns else None

    records = [json.loads(l) for l in open(args.corpus)]
    client = httpx.Client(base_url=args.base_url, timeout=300.0)
    out = open(args.output, "a")
    done = skipped = 0

    for rec in records:
        key = f"{rec['run']}:{rec['turn']}"
        if only and key not in only:
            continue
        gate = gate_for_turn(rec, chain)
        if gate is None:
            skipped += 1
            continue
        for name, fn in strategies.items():
            if args.limit and done >= args.limit:
                break
            forked = fn(rec["call_messages"], chain)
            t0 = time.time()
            try:
                resp = call_llm(client, args.model, forked)
            except Exception as e:  # one dead fork must not kill the sweep
                out.write(json.dumps({"key": key, "strategy": name, "error": str(e)}) + "\n")
                out.flush()
                continue
            msg = resp["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            action = parse_action(text)
            committed = bool(action and action.get("action") == "commit")
            correct = committed and str(action.get("answer")) == gate["answer"]
            out.write(json.dumps({
                "key": key, "run": rec["run"], "turn": rec["turn"],
                "strategy": name, "gate_id": gate["gate_id"], "expected": gate["answer"],
                "action": action, "committed": committed, "correct": correct,
                "ctx_chars": sum(len(m["content"]) for m in forked),
                "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
                "latency_s": round(time.time() - t0, 1),
            }) + "\n")
            out.flush()
            done += 1
            print(f"  {key} {name} gate={gate['gate_id']} commit={committed} correct={correct}")
    print(f"forks: {done}  gate-unmatched turns skipped: {skipped}")


if __name__ == "__main__":
    main()
