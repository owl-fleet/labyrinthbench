"""run_wali.py — treatment-arm driver for the Wali-as-navigator experiment.

Inverted driver: one LB session + ONE chat request to the dedicated Wali navigator endpoint,
whose own ReAct loop drives maze_act calls against the LB /act API. This harness never
resends history — Wali's loop owns the context management, which is the variable under
test. LB scores the session externally (GET /score — the guardian's verdict).

Writes one results-JSONL row per run, schema-compatible with run_eval.py's score_data
(same field names → pooled analysis unchanged) plus the validity-gate fields the prereg
requires: wali_parse_failures, divergent_turns, pct_invalid_turns.

Prereg (LOCKED) and runbook: private lab notebook.

Usage (inside a sandbox container that can reach the navigator endpoint and
mounts /results):

  python cli/run_wali.py --deg alpha-1 --output /results/wali-smoke.jsonl --label wali-smoke
  python cli/run_wali.py --deg rev-2 --output /results/rev2-wali-14b.jsonl --label rev2-wali-14b
"""
from __future__ import annotations

import argparse
import json
import os
import time

import httpx

# Same import pattern as run_eval's accum_mem: works as `python cli/run_wali.py` or `-m cli.run_wali`.
try:
    from cli.run_eval import _acquire_lock, _release_lock, _update_heartbeat
except ImportError:
    from run_eval import _acquire_lock, _release_lock, _update_heartbeat

# Tool-flavored translation of run_eval.SYSTEM_MECHANICS (pull off) — same informational
# content as the control arm's mechanics block, with the JSON-in-content response format
# replaced by the maze_act tool contract. Adapter-side framing, same class as the control
# harness prompt; the DEG briefing is appended verbatim below it.
MAZE_MECHANICS = """\
Your objective: reach EXIT. That is the only goal.

Each node has paths forward. Paths are open (free to take) or gated (solve a problem to proceed).
A correct gate answer takes you forward. A wrong answer takes you the wrong direction.
Gate problems and whether each path is open or gated are visible via observe.

Act by calling the maze_act tool — exactly ONE action per call:

  maze_act(action="observe")
      See your current location, all paths (with gate problems shown), and remaining budget.

  maze_act(action="commit", path_id="X")
      Take open path X.

  maze_act(action="commit", path_id="X", answer="Y")
      Answer the gate on path X and proceed.
      X = the path label from observe (e.g. "forward", "left") — NOT a gate option number.
      Y = your answer to the gate problem (e.g. "1", "2", "42", "TRUE").

  maze_act(action="commit", path_id="back")
      Return to your previous location. Costs one step.

  maze_act(action="note", text="Y")
      Store a persistent note. Returned in every future observe. Free.

Rules:
- commit costs one step. observe and note are free.
- Exhaust your step budget without reaching EXIT = failure.
- Dead ends have no forward paths — use commit with path_id="back" to return.
- Keep acting until the maze reports completed=true. Do NOT stop to summarize or ask
  questions — every response must be a maze_act call until the session is complete."""

# Tail-position instruction, appended AFTER the briefing — the analog of run_eval's
# _JSON_TAIL, which is the final line of the control arm's system prompt (amendment 4b:
# prompt-parity; without it the last thing the treatment model reads is the briefing's
# answer-format sentence, not the response contract).
MAZE_TAIL = "Respond with ONLY a maze_act tool call — no prose — until the maze reports completed=true."

# Engine event types with a guaranteed 1:1 SessionEvent emit per successful call. `inspect`
# is excluded: its "back"/invalid-path branches return in-band without emitting, so it
# cannot be position-aligned against the server-side history.
_EMIT_TYPES = {"observe", "commit", "note", "pull"}


def _consume_stream(resp: httpx.Response, lock_base_url: str) -> dict:
    """Consume wali's SSE stream for one episode. Returns the harness-side view:
    ordered maze_act calls (with parse health + paired results), the eval_summary
    counters, and the narration/reasoning text attributed per call."""
    calls: list[dict] = []          # {"action", "args", "parse_failed", "ok", "summary", ...}
    pending: list[dict] = []        # tool_call events awaiting their tool_result
    eval_summary: dict = {}
    text_buf: list[str] = []
    reasoning_buf: list[str] = []
    final_text = ""

    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except Exception:
            continue

        etype = data.get("type")
        if etype == "tool_call":
            call = {
                "name": data.get("name", ""),
                "action": (data.get("args") or {}).get("action"),
                "args": data.get("args") or {},
                "parse_failed": bool(data.get("parse_failed")),
                "model_text": "".join(text_buf),
                "model_reasoning": "".join(reasoning_buf),
                "ok": None,
                "summary": "",
                "steps_used": None,
            }
            text_buf.clear()
            reasoning_buf.clear()
            calls.append(call)
            pending.append(call)
        elif etype == "tool_result":
            _update_heartbeat(lock_base_url)
            if pending:
                call = pending.pop(0)
                summary = data.get("summary", "")
                call["summary"] = summary
                if summary.startswith("error:"):
                    call["ok"] = False
                else:
                    # maze_act summaries are '{"ok":..,"completed":..,..} | <engine text>'
                    try:
                        head = json.loads(summary.split(" | ", 1)[0])
                        call["ok"] = head.get("ok") is not False
                        call["steps_used"] = head.get("steps_used")
                    except Exception:
                        call["ok"] = True  # non-structured summary — assume dispatched fine
        elif etype == "eval_summary":
            eval_summary = data
        elif etype == "chat_lapse":
            # No-tool-call turn while the maze was incomplete (amendment 4). Wali nudged and
            # continued; log the lapse prose as a pseudo-turn for pathology tracing.
            calls.append({
                "name": "", "action": None, "args": {}, "parse_failed": False,
                "chat_lapse": True,
                "model_text": "".join(text_buf),
                "model_reasoning": "".join(reasoning_buf),
                "ok": None, "summary": "[chat_lapse — nudged to continue]", "steps_used": None,
            })
            text_buf.clear()
            reasoning_buf.clear()
        elif etype == "thinking":
            reasoning_buf.append(data.get("content", ""))
        elif etype in ("hud_state",):
            continue
        else:
            try:
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    text_buf.append(delta)
            except Exception:
                continue

    final_text = "".join(text_buf)
    return {"calls": calls, "eval_summary": eval_summary, "final_text": final_text}


def _divergent_turns(calls: list[dict], score_events: list[dict]) -> int | None:
    """Count positions where Wali's internal view of the action taken disagrees with the
    engine's server-side history — the prereg's adapter-honesty check. Both sequences are
    filtered to emit-guaranteed action types and successful (ok) calls; mismatched
    positions plus any length delta count as divergent."""
    if score_events is None:
        return None
    wali_seq = [c["action"] for c in calls
                if c.get("ok") and not c["parse_failed"] and c.get("action") in _EMIT_TYPES]
    lb_seq = [e.get("action") for e in score_events if e.get("action") in _EMIT_TYPES]
    mism = sum(1 for a, b in zip(wali_seq, lb_seq) if a != b)
    return mism + abs(len(wali_seq) - len(lb_seq))


def run_episode(deg_id: str, lb_url: str, wali_url: str, model: str,
                lock_base_url: str, episode_timeout: float, verbose: bool = False) -> dict:
    t_start = time.monotonic()
    client = httpx.Client(base_url=lb_url, timeout=30.0)

    # 1. Plain LB session — no engine HUD flags; the orchestration under test is Wali's own.
    sess = client.post("/session", json={"deg_id": deg_id, "model": model})
    sess.raise_for_status()
    sdata = sess.json()
    session_id = sdata["session_id"]
    step_budget = sdata["step_budget"]
    print(f"  session {session_id[:8]} deg={deg_id} step_budget={step_budget}")

    # 2. One request; Wali's loop does the rest. session_id rides the request so the
    #    orchestrator can bind maze_act to this LB session. `thinking` is deliberately
    #    omitted (upstream default = thinking-on, the control arm's exact regime).
    system_content = MAZE_MECHANICS + "\n\n" + sdata["briefing"].strip() + "\n\n" + MAZE_TAIL
    body = {
        "model": model,
        "stream": True,
        "session_id": session_id,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": (
                "You are connected to a live maze session. Solve it: reach EXIT. "
                "Use the maze_act tool; begin now."
            )},
        ],
    }
    stream_timeout = httpx.Timeout(connect=10.0, read=episode_timeout, write=30.0, pool=10.0)
    with httpx.Client(timeout=stream_timeout) as wali:
        with wali.stream("POST", f"{wali_url}/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            view = _consume_stream(resp, lock_base_url)

    calls = view["calls"]
    eval_summary = view["eval_summary"]
    parse_failures = eval_summary.get("parse_failures", sum(1 for c in calls if c["parse_failed"]))
    total_calls = eval_summary.get("tool_calls", len(calls))
    if verbose:
        for i, c in enumerate(calls):
            print(f"  [call {i}] {c['action']} ok={c['ok']} parse_failed={c['parse_failed']}")

    # 3. Score of record — the engine's verdict, never the executor's. A session Wali left
    #    incomplete (MAX_ITERATIONS exhausted before exit/out-of-lives) has no /score yet;
    #    fall back to /state for the same deterministic quantities, loudly flagged.
    st = client.get(f"/session/{session_id}/state")
    state = st.json() if st.status_code == 200 else {}
    score_events = None
    if state.get("completed"):
        score_resp = client.get(f"/score/{session_id}")
        score_resp.raise_for_status()
        score_data = score_resp.json()
        score_events = score_data.get("events")
    else:
        print("  WARNING: session not completed (Wali loop ended first) — scoring from /state")
        score_data = {
            "session_id": session_id,
            "found_exit": False,
            "failure_reason": "wali_loop_exhausted",
            "steps_to_exit": None,
            "ramp_depth": len(state.get("gate_results", {})),
            "step_budget": step_budget,
            "score_source": "state_incomplete",
        }

    # 4. Row assembly — run_eval.py field names verbatim so pooling works unchanged.
    score_data["model"] = model
    score_data["deg_id"] = deg_id
    score_data["turns"] = eval_summary.get("iterations", len(calls))
    score_data["elapsed_seconds"] = round(time.monotonic() - t_start, 2)
    score_data["arm"] = "treatment"
    score_data["turns_log"] = [
        {"turn": i, "model_text": c["model_text"], "model_reasoning": c["model_reasoning"],
         "action_parsed": c["args"], "engine_text": c["summary"], "parse_failed": c["parse_failed"],
         "chat_lapse": c.get("chat_lapse", False)}
        for i, c in enumerate(calls)
    ]
    score_data["final_text"] = view["final_text"]
    divergent = _divergent_turns(calls, score_events)
    chat_lapses = eval_summary.get("chat_lapses", sum(1 for c in calls if c.get("chat_lapse")))
    iterations = eval_summary.get("iterations", len(calls))
    score_data["wali_parse_failures"] = parse_failures
    score_data["wali_chat_lapses"] = chat_lapses
    score_data["divergent_turns"] = divergent
    score_data["wali_tool_calls"] = total_calls
    # Validity gate quantity (prereg: >10% of episode turns fall back → run measures the
    # adapter, not the orchestration). Numerator = fell-back turns (parse failures + chat
    # lapses) plus any view divergence; denominator = episode turns (loop iterations).
    if iterations:
        score_data["pct_invalid_turns"] = round(
            (parse_failures + chat_lapses + (divergent or 0)) / iterations, 4)
    else:
        score_data["pct_invalid_turns"] = 1.0
    client.close()
    return score_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", default="alpha-1")
    ap.add_argument("--lb-url", default="http://localhost:8090")
    ap.add_argument("--wali-url", default="http://localhost:11440")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--output", default="/results/results.jsonl")
    ap.add_argument("--label", default=None)
    ap.add_argument("--lock-base-url",
                    default=os.environ.get("LB_LOCK_HOST") or os.environ.get("LB_BASE_URL", "http://localhost:11434/v1"),
                    help="GPU host the agent model runs on — used only for the eval lock/heartbeat "
                         "(/eval-status visibility), not for any request. Default: $LB_LOCK_HOST, "
                         "else $LB_BASE_URL, else http://localhost:11434/v1 — this flag IS already "
                         "the operator-supplied lock-key override run_eval.py's --lock-host adds "
                         "elsewhere (see its module docstring 'Locking through a gateway'); name "
                         "it as the physical host, not a gateway, or runs through a shared gateway "
                         "will over-serialize against unrelated hardware.")
    ap.add_argument("--episode-timeout", type=float, default=600.0,
                    help="Max seconds between SSE chunks before the episode aborts.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _acquire_lock(f"wali:{args.model}", args.deg, args.runs, args.lock_base_url)
    try:
        for i in range(args.runs):
            print(f"Run {i + 1}/{args.runs} — wali({args.model}) on {args.deg}")
            try:
                result = run_episode(args.deg, args.lb_url, args.wali_url, args.model,
                                     args.lock_base_url, args.episode_timeout, args.verbose)
            except Exception as e:
                print(f"  ERROR: {e}")
                result = {"model": args.model, "deg_id": args.deg, "arm": "treatment", "error": str(e)}
            if args.label:
                result["run_label"] = args.label
            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")

            status = "EXIT ✓" if result.get("found_exit") else "DNF ✗"
            print(f"  {status}  ramp_depth={result.get('ramp_depth', '?')}  "
                  f"turns={result.get('turns', '?')}  "
                  f"pct_invalid={result.get('pct_invalid_turns', '?')}  "
                  f"(parse_failures={result.get('wali_parse_failures', '?')}, "
                  f"chat_lapses={result.get('wali_chat_lapses', '?')}, "
                  f"divergent={result.get('divergent_turns', '?')})")
    finally:
        _release_lock(args.lock_base_url)
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
