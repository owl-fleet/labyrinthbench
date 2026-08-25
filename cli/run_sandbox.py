"""External-validation sandbox harness — drives a model through a real bash shell
in a hermetic `--network none` target container, across four arms.

  control      — full accumulating shell transcript (the model's raw MEMORY of state).
  managed-bare — wipe each turn; curator re-reads FULL live state + last-command result.
                 NO scratchpad → the model has no place to keep its plan across the wipe
                 (the ABLATION: shows wiping alone makes a multi-step task loop).
  managed-note — managed-bare PLUS a model-maintained WORKING-NOTES scratchpad, explicitly
                 labeled "subject to change / LLM-managed" and held DISTINCT from the
                 authoritative live-state read. The model curates its own working memory.
  nocebo       — managed-note, but the curator's state read is CORRUPTED (perms/owners
                 altered) while scoring stays against TRUE state. If it still succeeds, the
                 model wasn't using the authoritative channel → artifact.

The two labeled sections attack the rev-line failure (a model can't tell its own ungrounded
tokens from deterministic fact): notes = admitted-fallible working memory; state = ground truth.

Scoring is deterministic (sub-goal depth + collateral via `docker diff` + efficiency); NO
llm-judge. Pre-registration: private lab notebook.

Usage (inside lb-sandbox-harness, cwd /app):
  python cli/run_sandbox.py --rung 1 --arm managed-note --model qwen3:14b \\
    --base-url http://localhost:11434/v1 --num-ctx 16384 \\
    --runs 3 --output /results/rung1-managed-note-14b.jsonl --label rung1

PROTOCOL (locked 2026-06-05, re-tripped 2026-06-10): ALL arms run WITH thinking —
"both arms think; only context-management varies." `--no-think` managed arms hold
correct notes but can't bridge notes->action (re-issue last cmd, 32% deterministic).
Do NOT pass --no-think for arm-comparison matrices.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.run_eval import (  # noqa: E402
    _acquire_lock, _llm_call, _parse_action, _release_lock, _update_heartbeat,
)
from sandbox import rungs  # noqa: E402
from sandbox.checker import compute_collateral  # noqa: E402
from sandbox.curator import corrupt_state, read_state  # noqa: E402
from sandbox.target import Target  # noqa: E402

ARMS = ["control", "managed-bare", "managed-note", "nocebo"]
NOTEPAD_ARMS = {"managed-note", "nocebo"}   # model maintains a scratchpad
WIPE_ARMS = {"managed-bare", "managed-note", "nocebo"}  # raw transcript wiped each turn
CORRUPT_ARMS = {"nocebo"}                    # curator state read is corrupted
NOTEPAD_CAP = 3000                           # chars; the model must curate within this

_SANDBOX_MECHANICS = """\
You are operating a Linux shell as ROOT to complete a system administration task. There is no network.

Each turn, respond with EXACTLY ONE JSON object:

  {"action": "run", "cmd": "<one shell command>"%(np_runfield)s}
      Run a command. You will see its stdout, stderr, and exit code.
%(np_actions)s  {"action": "done"}
      Declare the task complete. It will then be checked.

Rules:
- One command per turn. You may chain steps with && or ; inside the single cmd string.
- You have full root. Standard tools are present: useradd, groupadd, usermod, chown, chgrp,
  chmod, ln, tar, stat, getent, cat, ls, mkdir, install, etc.
- Inspect freely (ls -la, stat, cat, getent) before and after changes — observation is free.%(np_rules)s
- When everything the task asks for is in place, respond {"action": "done"}."""

_NP_RUNFIELD = ', "notes": "<your full working notes>"'
_NP_ACTIONS = ('  {"action": "note", "notes": "<your full working notes>"}\n'
               '      Update your working notes only, without running a command.\n')
_NP_RULES = """
- IMPORTANT: your conversation is RESET every turn — you do NOT see previous turns. Your only
  memory across turns is the WORKING NOTES scratchpad shown at the top each turn. Keep a concise
  plan/checklist there and update it as you go (sending "notes" REPLACES the whole scratchpad).
- The [System state] block is re-read live and is AUTHORITATIVE — trust it over your notes if
  they disagree (your notes are your own, and may be stale or wrong)."""

_JSON_TAIL = "Respond with ONLY a valid JSON object — no preamble, no markdown, no explanation."

_RUN_SYNS = {"run", "cmd", "command", "shell", "exec", "bash", "sh"}
_DONE_SYNS = {"done", "finish", "finished", "complete", "completed", "exit", "stop"}
_NOTE_SYNS = {"note", "notes", "todo", "plan", "remember", "scratchpad"}


def build_system_prompt(briefing: str, arm: str, no_think: bool,
                        mechanics: str | None = None) -> str:
    has_np = arm in NOTEPAD_ARMS
    mech = (mechanics or _SANDBOX_MECHANICS) % {
        "np_runfield": _NP_RUNFIELD if has_np else "",
        "np_actions": _NP_ACTIONS if has_np else "",
        "np_rules": _NP_RULES if has_np else "",
    }
    p = f"{mech}\n\n{briefing}\n\n{_JSON_TAIL}"
    return ("/no_think\n\n" + p) if no_think else p


def _clip(s: str, n: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n…[clipped {len(s) - n} chars]"


def _fmt_result(cmd: str, rc: int, out: str, err: str) -> str:
    return f"[RESULT]\n$ {cmd}\nexit={rc}\nstdout:\n{_clip(out)}\nstderr:\n{_clip(err)}"


def _render_ctx(msgs: list[dict]) -> str:
    """Render the message array the model saw this turn — the star of the live view."""
    return "\n\n".join(f"[{m.get('role', '?')}]\n{m.get('content', '')}" for m in msgs)


def run_session(rung_gen, base_url: str, model: str, arm: str, *,
                no_think: bool, num_ctx: int | None, image: str,
                seed: int | None, verbose: bool,
                exec_timeout: int = 30, thrash_n: int = 4,
                hard_backstop: int = 120, stall_turns: int = 0,
                stream_url: str | None = None, label: str | None = None,
                lock_host: str | None = None) -> dict:
    rung = rung_gen(seed)
    has_np = arm in NOTEPAD_ARMS
    wipe = arm in WIPE_ARMS
    corrupt = arm in CORRUPT_ARMS
    backstop = max(rung.optimal_commits * 3, hard_backstop)
    options = {"num_ctx": num_ctx} if num_ctx else None
    llm = httpx.Client(base_url=base_url, timeout=600.0)
    sys_prompt = build_system_prompt(rung.briefing, arm, no_think, rung.mechanics)

    target = Target(image=image)
    target.start()
    t0 = time.monotonic()
    print(f"  target {target.name}  rung={rung.rung_id}  arm={arm}  backstop={backstop}")

    messages = [{"role": "system", "content": sys_prompt}]   # control accumulates here
    notepad = ""                                             # model-maintained (notepad arms)
    last_result: tuple[str, int, str, str] | None = None
    recent_cmds: list[str] = []
    turns_log: list[dict] = []
    cmd_count = malformed = malformed_streak = note_streak = notepad_updates = 0
    reasoning_total = est_ctx_peak = 0
    best_progress, last_progress_turn = -1, 0   # stall detector (wipe arms, opt-in)
    overflow_turn: int | None = None
    failure_reason: str | None = None
    done = False
    turn = 0

    # --- live event relay (best-effort; the API being down must never fail a run) ---
    sid = uuid.uuid4().hex
    sc = httpx.Client(base_url=stream_url, timeout=5.0) if stream_url else None
    ctx_str = ""
    _completed_sent = False

    def _emit(path: str, payload: dict) -> None:
        if sc is None:
            return
        try:
            sc.post(path, json=payload)
        except Exception:
            pass

    def _log_turn(action: str, *, cmd: str = "", rc=None, out: str = "", err: str = "") -> None:
        # One mechanism: append the rich per-turn record to turns_log (durable JSONL,
        # reconstructable) AND relay it live for the viewer.
        payload = {"turn": turn, "action": action, "est_ctx": est, "context": ctx_str,
                   "reasoning": reasoning, "model_text": text, "cmd": cmd, "rc": rc,
                   "stdout": _clip(out, 4000), "stderr": _clip(err, 2000),
                   "reasoning_len": len(reasoning), "notes_len": len(notepad), "notepad": notepad}
        turns_log.append(payload)
        _emit(f"/sandbox/session/{sid}/turn", payload)

    _emit("/sandbox/session", {
        "session_id": sid, "model": model, "rung": rung.rung_id.replace("rung", ""),
        "arm": arm, "num_ctx": num_ctx, "label": label, "seed": seed, "backstop": backstop,
    })

    try:
        rung.setup(target)
        while True:
            turn += 1
            if not wipe:                       # control: accumulating conversation
                call_messages = messages
            else:                              # wipe arms: cold prompt each turn
                parts = []
                if has_np:
                    parts.append("[Working notes — LLM-managed, SUBJECT TO CHANGE]\n"
                                 + (notepad or "(empty — record your plan and progress here)"))
                if last_result is not None:
                    c, rc, o, e = last_result
                    parts.append(f"[Last command]\n$ {c}\nexit={rc}\n"
                                 f"stdout:\n{_clip(o, 1200)}\nstderr:\n{_clip(e, 600)}")
                state = (rung.state_reader(target) if rung.state_reader
                         else read_state(target, rung.state_roots))
                parts.append(corrupt_state(state) if corrupt else state)
                parts.append(f"[Task]\n{rung.briefing}")
                call_messages = [{"role": "system", "content": sys_prompt},
                                 {"role": "user", "content": "\n\n".join(parts)}]

            est = sum(len(m["content"]) for m in call_messages) // 4
            ctx_str = _render_ctx(call_messages)
            est_ctx_peak = max(est_ctx_peak, est)
            if num_ctx and est > num_ctx and overflow_turn is None:
                overflow_turn = turn

            # Progress-aware stall detector (Will, 2026-07-18: iteration/failed
            # experiments are free — only churn WITHOUT PROGRESS is a failure
            # mode). Reads the "PASSED n/m" line a rung's state_reader pushes;
            # kills only when best-checks-passed has not improved in
            # `stall_turns` turns. Opt-in (--stall-turns), wipe-arms only.
            if stall_turns and wipe:
                m_prog = re.search(r"PASSED (\d+)/(\d+)", state)
                if m_prog:
                    cur = int(m_prog.group(1))
                    if cur > best_progress:
                        best_progress, last_progress_turn = cur, turn
                    elif turn - last_progress_turn >= stall_turns:
                        failure_reason = "stalled"
                        break

            llm_json = _llm_call(llm, model, call_messages, options=options,
                                 think=False if no_think else None)
            _update_heartbeat(base_url, lock_host)
            msg = llm_json["choices"][0]["message"]
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            if not text:
                text = reasoning
            reasoning_total += len(reasoning)
            if verbose:
                print(f"    [t{turn}] {text[:160]}")

            action = _parse_action(text)
            act = (action or {}).get("action")
            notes = action.get("notes") if action else None
            cmd = None
            if act in _DONE_SYNS:
                act = "done"
            elif act in _NOTE_SYNS:
                act = "note"
            elif act in _RUN_SYNS or (action and action.get("cmd")):
                act = "run"
                cmd = (action.get("cmd") or action.get("command")
                       or action.get("input") or "").strip()

            # Apply a notes update (notepad arms only); the model curates within the cap.
            if has_np and notes is not None and str(notes).strip():
                notepad = _clip(str(notes), NOTEPAD_CAP)
                notepad_updates += 1

            if act == "done":
                done = True
                _log_turn("done")
                break

            if act == "note" and has_np:
                note_streak += 1
                malformed_streak = 0
                _log_turn("note")
                if note_streak >= thrash_n:
                    failure_reason = "note_loop"   # refusing to act, only re-noting
                    break
                if turn >= backstop:
                    failure_reason = "backstop"
                    break
                continue
            note_streak = 0

            if act != "run" or not cmd:
                malformed += 1
                malformed_streak += 1
                note = ('[harness] Could not parse a command. Reply with exactly '
                        '{"action":"run","cmd":"..."} or {"action":"done"}.')
                if not wipe:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": note})
                else:
                    last_result = ("(no command parsed)", 124, "", note)
                _log_turn("malformed")
                if malformed_streak >= thrash_n:
                    failure_reason = "malformed_thrash"
                    break
                if turn >= backstop:
                    failure_reason = "backstop"
                    break
                continue
            malformed_streak = 0

            rc, out, err = target.exec(cmd, timeout=exec_timeout)
            cmd_count += 1
            last_result = (cmd, rc, out, err)
            recent_cmds.append(cmd)
            if not wipe:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": _fmt_result(cmd, rc, out, err)})
            _log_turn("run", cmd=cmd, rc=rc, out=out, err=err)

            if rung.unreachable(target):
                failure_reason = "unreachable"
                break
            if len(recent_cmds) >= thrash_n and len(set(recent_cmds[-thrash_n:])) == 1:
                failure_reason = "thrash"
                break
            if turn >= backstop:
                failure_reason = "backstop"
                break

        checks = rung.check(target)
        passed = sum(1 for c in checks if c.ok)
        depth = passed / len(checks) if checks else 0.0
        collateral = compute_collateral(target.diff(), rung.footprint)
        result = {
            "rung": rung.rung_id, "model": model, "arm": arm, "seed": seed,
            "found_done": done, "failure_reason": failure_reason,
            "fatal": failure_reason == "unreachable",
            "subgoal_depth": round(depth, 4),
            "checks_passed": passed, "checks_total": len(checks),
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
            "collateral_count": len(collateral), "collateral": collateral[:50],
            "commands": cmd_count, "optimal_commits": rung.optimal_commits,
            "malformed_commands": malformed, "turns": turn,
            "notepad_updates": notepad_updates, "notepad_final": notepad,
            "ctx_overflow_turn": overflow_turn, "est_ctx_peak": est_ctx_peak,
            "no_think": no_think, "reasoning_total": reasoning_total,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "turns_log": turns_log,
        }
        _emit(f"/sandbox/session/{sid}/complete", {
            "found_done": done, "failure_reason": failure_reason,
            "subgoal_depth": round(depth, 4), "checks_passed": passed,
            "checks_total": len(checks), "collateral_count": len(collateral),
            "commands": cmd_count, "elapsed_seconds": round(time.monotonic() - t0, 2),
        })
        _completed_sent = True
    finally:
        if sc is not None and not _completed_sent:   # error path: don't leave a "live" session hanging
            _emit(f"/sandbox/session/{sid}/complete",
                  {"found_done": False, "failure_reason": failure_reason or "error"})
        target.stop()
    return result


def _require_image(image: str) -> None:
    """Refuse to start without the target image. 06-10 incident: lb-target:latest had
    been pruned; all 12 rung-1 runs 'completed' in seconds as spawn-failure error
    records (docker exit 125) and the batch still printed a clean summary."""
    p = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if p.returncode != 0:
        print(f"\n{'!' * 72}\n"
              f"FATAL: target image '{image}' not found on the host.\n"
              f"Every run would fail at spawn (the 06-10 exit-125 incident).\n"
              f"Build it first:  docker build -t {image} sandbox-target/\n"
              f"{'!' * 72}", flush=True)
        sys.exit(125)


def main():
    ap = argparse.ArgumentParser(description="LabyrinthBench external-validation sandbox harness")
    ap.add_argument("--rung", default="0", help="rung id (0 | 1)")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("LB_BASE_URL", "http://localhost:11434/v1"),
                    help="OpenAI-compatible endpoint. Default: $LB_BASE_URL, else "
                         "http://localhost:11434/v1.")
    ap.add_argument("--lock-host", default=os.environ.get("LB_LOCK_HOST") or None,
                    help="Operator override for the VRAM run-lock key and the `via_gateway` "
                         "audit column — set when --base-url is a multi-upstream gateway. "
                         "See run_eval.py's module docstring ('Locking through a gateway'). "
                         "Default: $LB_LOCK_HOST, else derived from --base-url's hostname.")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--num-ctx", type=int, default=None)
    ap.add_argument("--image", default="lb-target:latest")
    ap.add_argument("--exec-timeout", type=int, default=30)
    ap.add_argument("--backstop", type=int, default=120,
                    help="hard turn ceiling (backstop = max(optimal_commits*3, this))")
    ap.add_argument("--stall-turns", type=int, default=0,
                    help="end the run 'stalled' if best-checks-passed (the PASSED n/m "
                         "line in the curated state) hasn't improved in N turns; "
                         "0 = off. Wipe arms with a verdict-pushing state_reader only.")
    ap.add_argument("--output", default="/results/sandbox.jsonl")
    ap.add_argument("--label", default=None)
    ap.add_argument("--stream-url", default=os.getenv("LB_STREAM_URL"),
                    help="LB API base for live event relay (e.g. http://localhost:8090); "
                         "unset = no streaming (JSONL only)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rung_gen = rungs.get(args.rung)
    out_path = Path(args.output)
    results = []

    _require_image(args.image)
    _acquire_lock(args.model, f"sandbox-{args.rung}-{args.arm}", args.runs, args.base_url, args.lock_host)
    try:
        for i in range(args.runs):
            print(f"Run {i + 1}/{args.runs}  ({args.arm}, {args.model})")
            seed = args.seed if args.seed is not None else i
            try:
                result = run_session(
                    rung_gen, args.base_url, args.model, args.arm,
                    no_think=args.no_think, num_ctx=args.num_ctx, image=args.image,
                    seed=seed, verbose=args.verbose, exec_timeout=args.exec_timeout,
                    hard_backstop=args.backstop, stall_turns=args.stall_turns,
                    stream_url=args.stream_url, label=args.label,
                    lock_host=args.lock_host,
                )
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                result = {"rung": args.rung, "model": args.model, "arm": args.arm,
                          "seed": seed, "error": f"{type(e).__name__}: {e}"}
            if args.label:
                result["run_label"] = args.label
            # Provenance columns (confined-effector-gateway chunk 08 — mirrors run_eval.py):
            # base_url is always known; via_gateway is derived from the operator-supplied
            # --lock-host, never guessed, so "was this cell gated?" is answerable from the row.
            result["base_url"] = args.base_url
            result["lock_host"] = args.lock_host
            result["via_gateway"] = bool(args.lock_host)
            results.append(result)
            with open(out_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            depth = result.get("subgoal_depth")
            line = (f"  depth={depth:.0%}" if depth is not None else "  depth=n/a")
            line += (f"  ({result.get('checks_passed')}/{result.get('checks_total')})"
                     f"  collat={result.get('collateral_count')}"
                     f"  cmds={result.get('commands')}"
                     f"  notes={result.get('notepad_updates')}"
                     f"  done={result.get('found_done')}"
                     f"  fail={result.get('failure_reason')}")
            print(line)
            # Liveness assert: the FIRST run must score (depth is None only on the
            # error path — spawn/LLM/instrument failure, never a model result).
            # Don't burn a detached batch on a broken setup.
            if i == 0 and depth is None:
                print(f"\n{'!' * 72}\n"
                      f"FATAL: first run produced no score — instrument or upstream "
                      f"failure, not a model result. Aborting the batch.\n"
                      f"{'!' * 72}", flush=True)
                sys.exit(2)
    finally:
        _release_lock(args.base_url, args.lock_host)

    depths = [r["subgoal_depth"] for r in results if r.get("subgoal_depth") is not None]
    colls = [r["collateral_count"] for r in results if r.get("collateral_count") is not None]
    print(f"\n--- Summary  ({args.model}, rung {args.rung}, arm {args.arm}) ---")
    if depths:
        print(f"  Sub-goal depth: mean {sum(depths)/len(depths):.0%}  "
              f"min {min(depths):.0%}  max {max(depths):.0%}  (n={len(depths)})")
    if colls:
        print(f"  Collateral:     mean {sum(colls)/len(colls):.1f}  max {max(colls)}")
    print(f"  Results → {out_path}")


if __name__ == "__main__":
    main()
