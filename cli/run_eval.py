"""LabyrinthBench CLI harness.

Runs a model through one or more DEG sessions and reports scores.

Usage:
  python run_eval.py --model qwen3.5:2b --base-url http://localhost:11434/v1 [options]

Options:
  --model         Model name (passed in the messages body)
  --base-url      OpenAI-compatible base URL (default: http://localhost:11434/v1)
  --maze-url      LabyrinthBench API URL (default: http://localhost:8090)
  --deg           DEG id to run (default: alpha-1)
  --runs          Number of independent sessions (default: 1)
  --no-think      Prepend /no_think to the system prompt (for Qwen3 thinking models)
  --verbose       Print full model responses
  --output        JSONL output file (default: /results/results.jsonl)
  --db-url        TimescaleDB connection string (optional; skips insert if omitted)
  --label         Run label tag stored in DB (e.g. baseline-20260520)
  --inject-history  Append harness-tracked decision history to each turn's user message
  --kos-prompt    Prepend structured navigation state (confirmed dead ends) before each observation
  --stateless     Wipe model context between turns; inject [Navigation State + Decision History +
                  Observation] as a fresh cold prompt each turn. Eliminates context snowball.
                  Automatically tracks history and dead ends regardless of other flags.
  --num-ctx       Ollama num_ctx (KV cache size). Use 16384 for phi4-reasoning.
  --context-policy  Named ContextPolicy (cli/context_policy.py) — arms as config, not a harness
                  fork. Mutually exclusive with --overlay-only/--stateless/--inject-history/
                  --kos-prompt (those stay as the untouched legacy flag matrix).
  --policy-code-ref  Repo URL/commit for the exact policy code used this run (leaderboard
                  integrity — auto-derived from this checkout's HEAD when omitted).
  --n-ctx-slot    Journal-verified n_ctx_slot (int) for this run's base_url host — operator-
                  supplied from `journalctl -u ollama | grep n_ctx_slot` (scripts/e1a-run-row.sh
                  has the SSH+grep recipe). NOT auto-detected: ollama's /v1 endpoint silently
                  drops --num-ctx (options), so the CLI flag can never be trusted as ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Cross-run memory faculty for the LB Design 2 accumulation eval. Importable whether run as
# `python cli/run_eval.py` (sibling on sys.path[0]) or `python -m cli.run_eval` (package).
try:
    from cli import accum_mem
except ImportError:
    import accum_mem

# Pluggable per-arm context management (lb-post-release chunk 02) — same import pattern.
try:
    from cli import context_policy
except ImportError:
    import context_policy

# Serving-stack provenance — what actually served this run, not just the tag we asked for.
try:
    from cli import provenance
except ImportError:
    import provenance

_DEFAULT_DB_URL = os.environ.get("DB_URL", "")
# Cold loads of large models (60-90 GB) can exceed 10 minutes before the first
# byte arrives; a flat 600s read timeout cancels them mid-load. Override via env.
_LLM_TIMEOUT_SECS = float(os.environ.get("LB_LLM_TIMEOUT", "1800"))
_HEARTBEAT_STALE_SECS = 600  # lock is hung if heartbeat older than this


def _lock_path(base_url: str) -> Path:
    host = urlparse(base_url).hostname or "local"
    return Path(f"/results/.eval_lock_{host.replace('.', '_')}")


def _acquire_lock(model, deg, runs, base_url):
    lock = _lock_path(base_url)
    if lock.exists():
        existing = json.loads(lock.read_text())
        try:
            os.kill(existing["pid"], 0)
            # PID alive — check heartbeat to distinguish running vs. hung
            last_hb = existing.get("last_heartbeat", existing.get("started", ""))
            try:
                hb_age = time.time() - time.mktime(time.strptime(last_hb, "%Y-%m-%dT%H:%M:%S"))
            except Exception:
                hb_age = 0
            if hb_age > _HEARTBEAT_STALE_SECS:
                print(
                    f"WARNING: lock held by PID {existing['pid']} but heartbeat is "
                    f"{hb_age/60:.0f}m old — process appears hung. Overriding lock."
                )
            else:
                raise SystemExit(
                    f"ERROR: eval already running on {urlparse(base_url).hostname} — "
                    f"{existing['model']} on {existing['deg']}\n"
                    f"PID {existing['pid']}, started {existing['started']}, "
                    f"last heartbeat {last_hb}\n"
                    f"Check http://localhost:8090/eval-status"
                )
        except ProcessLookupError:
            pass  # stale lock from dead process
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    lock.write_text(json.dumps({
        "model": model, "deg": deg, "runs": runs,
        "base_url": base_url, "pid": os.getpid(),
        "started": now, "last_heartbeat": now,
    }, indent=2))


def _update_heartbeat(base_url: str) -> None:
    lock = _lock_path(base_url)
    if not lock.exists():
        return
    try:
        data = json.loads(lock.read_text())
        data["last_heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        lock.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _release_lock(base_url: str):
    _lock_path(base_url).unlink(missing_ok=True)


# Action mechanics — invariant across all DEGs. %-template (precedent: run_sandbox._SANDBOX_MECHANICS):
# with pull off, both placeholders render byte-identical to the pre-pull prompt (pinned by hash check).
SYSTEM_MECHANICS = """\
Your objective: reach EXIT. That is the only goal.

Each node has paths forward. Paths are open (free to take) or gated (solve a problem to proceed).
A correct gate answer takes you forward. A wrong answer takes you the wrong direction.
Gate problems and whether each path is open or gated are visible in observe().

Actions — respond with exactly ONE JSON object per turn:

  {"action": "observe"}
      See your current location, all paths (with gate problems shown), and remaining budget.

  {"action": "commit", "path_id": "X"}
      Take open path X.

  {"action": "commit", "path_id": "X", "answer": "Y"}
      Answer the gate on path X and proceed.
      X = the path label from observe() (e.g. "forward", "left") — NOT a gate option number.
      Y = your answer to the gate problem (e.g. "1", "2", "42", "TRUE").
      Example: if observe() shows
        forward: select route  [GATE: 1 = use_metric_query, 2 = use_doc_search]
      then commit with: {"action": "commit", "path_id": "forward", "answer": "1"}

  {"action": "commit", "path_id": "back"}
      Return to your previous location. Costs one step.

  {"action": "note", "text": "Y"}
      Store a persistent note. Returned in every future observe(). Free.
%(pull_action)s
Rules:
%(observe_rule)s- %(cost_rule)s
- Exhaust your step budget without reaching EXIT = failure.
- Dead ends have no forward paths — use commit("back") to return."""

# recommended-observe policy (prereg 15): a clear one-line imperative in the Rules block (fixed
# position, not the tail — amendment-4b lesson). "unique" implies non-inferability without naming
# the failure mode. LOCKED wording 2026-07-06.
_OBSERVE_RULE = "- Before you answer any gate, observe it — each gate's problem is unique.\n"

_COST_RULE_BASE = "commit() costs one step. observe() and note() are free."
_COST_RULE_PULL = "commit() and pull() each cost one step. observe() and note() are free."
_PULL_ACTION_BLOCK = """
  {"action": "pull"}
      Request the current values of all tracked variables. Returns the [STATE]
      ledger plus a fresh view of your current location. Costs one step.
"""

# Epistemic-label arm (the label-flip test): flipped-authority memory-status disclosure —
# the exact opposite pole of run_sandbox's "your notes are your own, and may be stale or wrong".
# Lives in the system prompt so a wiped (overlay-only) model sees it every turn.
VERIFIED_DISCLOSURE = """\
MEMORY STATUS: You have already worked through earlier gates on this ladder. Your context
is wiped every turn, so you have no internal memory of having done so. The current values
of all tracked variables are accessible at any time with {"action": "pull"}. These are
VERIFIED values — read live from the engine's authoritative state, not reconstructed from
memory. If anything you believe disagrees with a pulled value, the pulled value wins."""

# Task framing — overridden per-DEG by `meta.briefing`; this is the default when none is set.
DEFAULT_STRATEGY = """\
Strategy: EXIT is the goal. Gate accuracy is not. Survey your options before committing.
An open path is always worth considering. A wrong gate answer costs you steps to recover."""

_JSON_TAIL = "Respond with ONLY a valid JSON object. No preamble, no explanation."


def build_system_prompt(briefing: str = "", pull_state: bool = False, state_label: str = "",
                        recommend_observe: bool = False) -> str:
    """Mechanics + (VERIFIED disclosure when labeled) + (DEG briefing or default strategy) + JSON tail."""
    mech = SYSTEM_MECHANICS % {
        "pull_action": _PULL_ACTION_BLOCK if pull_state else "",
        "cost_rule": _COST_RULE_PULL if pull_state else _COST_RULE_BASE,
        "observe_rule": _OBSERVE_RULE if recommend_observe else "",
    }
    framing = briefing.strip() if briefing.strip() else DEFAULT_STRATEGY
    if state_label == "verified":
        framing = f"{VERIFIED_DISCLOSURE}\n\n{framing}"
    return f"{mech}\n\n{framing}\n\n{_JSON_TAIL}\n"


def _parse_available_paths(text: str) -> list[str]:
    """Extract path IDs listed in an --- OBSERVE --- response."""
    paths = []
    in_paths = False
    for line in text.splitlines():
        if line.strip() == "Paths:":
            in_paths = True
            continue
        if in_paths:
            m = re.match(r'^  (\S+):', line)
            if m:
                paths.append(m.group(1))
            elif line == "" or (line and not line.startswith(" ")):
                in_paths = False
    return paths


def _parse_location(text: str) -> str | None:
    m = re.search(r"^Location:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _is_dead_end(text: str) -> bool:
    t = text.lower()
    return "dead end" in t or "dead-end" in t or "dead_end" in t or "no forward" in t or "no exits" in t


def _build_kos_state_block(kos_state: dict) -> str:
    dead_ends = kos_state.get("confirmed_dead_ends", set())
    if dead_ends:
        dead_end_line = f"Confirmed dead ends (do not re-enter): {', '.join(sorted(dead_ends))}"
    else:
        dead_end_line = "Confirmed dead ends: none yet"
    return f"[Navigation State]\n{dead_end_line}\n\n[Observation]\n"


def _build_history_block(history: list[dict], max_entries: int = 8) -> str:
    if not history:
        return ""
    recent = history[-max_entries:]
    lines = ["[Decision History]"]
    for h in recent:
        loc = h.get("location") or "unknown"
        dead = " — DEAD END" if h.get("dead_end") else ""
        lines.append(f"  Turn {h['turn']}: {h['action_str']} → {loc}{dead}")
    return "\n" + "\n".join(lines) + "\n"


def _build_stateless_injection(kos_state: dict, decision_history: list[dict], engine_text: str) -> str:
    """Build a fresh cold-prompt user message: nav state + history + current observation."""
    dead_ends = kos_state.get("confirmed_dead_ends", set())
    dead_end_line = (
        f"Confirmed dead ends (do not re-enter): {', '.join(sorted(dead_ends))}"
        if dead_ends else "Confirmed dead ends: none yet"
    )
    parts = [f"[Navigation State]\n{dead_end_line}"]
    if decision_history:
        recent = decision_history[-8:]
        lines = ["[Decision History]"]
        for h in recent:
            loc = h.get("location") or "unknown"
            dead = " — DEAD END" if h.get("dead_end") else ""
            lines.append(f"  Turn {h['turn']}: {h['action_str']} → {loc}{dead}")
        parts.append("\n".join(lines))
    parts.append(f"[Observation]\n{engine_text}")
    return "\n\n".join(parts)


def _parse_action(text: str) -> dict | None:
    """Extract the first JSON object from the model's response."""
    text = text.strip()
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find first { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _insert_run(db_url: str, score: dict, label: str | None) -> None:
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError:
        print("  WARNING: psycopg2 not available — skipping DB insert")
        return
    try:
        conn = psycopg2.connect(db_url)
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO labyrinth_runs (
                    session_id, model, deg_id,
                    found_exit, steps_to_exit, step_budget, optimal_commits,
                    normalized_efficiency, gate_accuracy, path_correctness,
                    recovery_rate, chain_gate_count, chain_accuracy, knowledge_state_consistency,
                    note_used, elapsed_seconds, turns, run_label, base_url, n_ctx_slot, prov
                ) VALUES (
                    %(session_id)s, %(model)s, %(deg_id)s,
                    %(found_exit)s, %(steps_to_exit)s, %(step_budget)s, %(optimal_commits)s,
                    %(normalized_efficiency)s, %(gate_accuracy)s, %(path_correctness)s,
                    %(recovery_rate)s, %(chain_gate_count)s, %(chain_accuracy)s, %(knowledge_state_consistency)s,
                    %(note_used)s, %(elapsed_seconds)s, %(turns)s, %(run_label)s, %(base_url)s, %(n_ctx_slot)s,
                    %(prov)s
                )
                """,
                {
                    "session_id": score.get("session_id"),
                    "model": score.get("model"),
                    "deg_id": score.get("deg_id"),
                    "found_exit": score.get("found_exit", False),
                    "steps_to_exit": score.get("steps_to_exit"),
                    "step_budget": score.get("step_budget", 0),
                    "optimal_commits": score.get("optimal_commits", 0),
                    "normalized_efficiency": score.get("normalized_efficiency"),
                    "gate_accuracy": score.get("gate_accuracy"),
                    "path_correctness": score.get("path_correctness"),
                    "recovery_rate": score.get("recovery_rate"),
                    "chain_gate_count": score.get("chain_gate_count"),
                    "chain_accuracy": score.get("chain_accuracy"),
                    "knowledge_state_consistency": score.get("knowledge_state_consistency"),
                    "note_used": score.get("note_used", False),
                    "elapsed_seconds": score.get("elapsed_seconds"),
                    "turns": score.get("turns"),
                    "run_label": label,
                    # Provenance columns (lb-post-release chunk 02 — the standing gate): base_url is
                    # always known; n_ctx_slot is journal-verified by the operator, never the CLI flag
                    # (ollama's /v1 endpoint silently drops --num-ctx — see run_eval.py --help).
                    "base_url": score.get("base_url"),
                    "n_ctx_slot": score.get("n_ctx_slot"),
                    # Serving-stack identity tuple (2026-08-15) — engine/build/weights/template.
                    "prov": Json(score["prov"]) if score.get("prov") else None,
                },
            )
        conn.close()
        print(f"  DB: inserted run {str(score.get('session_id', ''))[:8]}")
    except Exception as e:
        print(f"  WARNING: DB insert failed — {e}")


def _asdict_or_none(obj) -> dict | None:
    return asdict(obj) if obj is not None else None


def _native_usage(native_json: dict) -> dict | None:
    """Normalize ollama's NATIVE /api/chat token counters (prompt_eval_count/eval_count) into an
    OpenAI-ish usage dict, keeping the raw duration fields — same silent-shape-divergence class as
    the documented `think`/`options` drops on /v1: the native and OpenAI-compat paths report usage
    under different keys entirely, so a caller that only checks `resp["usage"]` silently gets None
    on the native path unless this translation happens explicitly."""
    pe, ee = native_json.get("prompt_eval_count"), native_json.get("eval_count")
    if pe is None and ee is None:
        return None
    return {
        "prompt_tokens": pe,
        "completion_tokens": ee,
        "total_tokens": (pe or 0) + (ee or 0),
        "raw": {k: native_json.get(k) for k in (
            "prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration", "total_duration",
        )},
    }


def _llm_call(llm: httpx.Client, model: str, messages: list, retries: int = 3, options: dict | None = None, think: bool | None = None) -> dict:
    """Call the model with retry. When `think` is set we MUST use Ollama's NATIVE /api/chat endpoint:
    the OpenAI-compat /v1/chat/completions SILENTLY IGNORES a top-level `think` field, so `think:false`
    there does NOT suppress reasoning (verified on qwen3:14b — reasoning_len ~600 via /v1 vs 0 via
    /api/chat). We translate the native response into the OpenAI-shaped dict the caller expects. Models
    that don't support `think` (e.g. llama3.3) 400 on /api/chat → we fall back to /v1 (they don't think
    anyway). Non-Ollama OpenAI-compat servers (e.g. LM Studio) don't implement /api/chat at all, but
    some answer an unknown route with HTTP 200 + an OpenAI-style {"error": ...} body instead of a
    4xx/5xx — raise_for_status() never fires, so a bare `native_json.get("message", {})` silently
    extracts empty content and the caller reads it as a real (blank) answer, not a failure (stranger
    test 2026-08-04: 64 straight injected observes with nothing surfaced). An `error` key or a missing
    `message` on the native path is therefore also treated as a failed call → fall through to /v1,
    same as the HTTPStatusError case. The returned dict always carries `usage` (OpenAI-shaped) when
    the server reported one, on either path — token-usage capture (todo-ai backlog item) reads this
    per turn."""
    last_exc = None
    _base = str(llm.base_url).rstrip("/")
    _native = (_base[:-3] if _base.endswith("/v1") else _base) + "/api/chat"
    for attempt in range(retries):
        try:
            if think is not None:
                try:
                    payload = {"model": model, "messages": messages, "stream": False, "think": think}
                    if options:
                        payload["options"] = options
                    r = llm.post(_native, json=payload)
                    r.raise_for_status()
                    native_json = r.json()
                    if "error" in native_json or "message" not in native_json:
                        raise ValueError(
                            f"native /api/chat returned no message: {native_json.get('error', native_json)!r}"
                        )
                    m = native_json["message"]
                    return {
                        "choices": [{
                            "message": {"content": m.get("content", ""), "reasoning": m.get("thinking", "")},
                            # Native /api/chat reports truncation as done_reason="length". Surfaced
                            # under the OpenAI key so the caller has ONE thing to check (2026-08-22).
                            "finish_reason": native_json.get("done_reason"),
                        }],
                        "usage": _native_usage(native_json),
                    }
                except (httpx.HTTPStatusError, ValueError):
                    pass  # model likely doesn't support `think`, or the server has no native endpoint
                          # at all (200 + error body) → fall through to the OpenAI path
            payload = {"model": model, "messages": messages, "stream": False}
            if options:
                payload["options"] = options
            r = llm.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_exc = e
            print(f"  LLM call attempt {attempt+1}/{retries} failed: {type(e).__name__}: {e} — retrying")
            time.sleep(5 * (attempt + 1))
    raise last_exc


def _corrupt_state_line(text: str) -> str:
    """PLACEBO (falsification): rewrite the [STATE] ledger the model SEES to per-variable-shifted
    (wrong) values, while the engine still scores against the TRUE values. If a corrupted ledger
    still passes, the model wasn't depending on the ledger → the result is an artifact. The shift is
    per-variable (breaks order AND equality, unlike a uniform offset) and tracks the true value (stays
    plausible, so the model won't trivially detect-and-compensate)."""
    import re
    def _shift(mm):
        name, val = mm.group(1), int(mm.group(2))
        return f"{name} = {val + (ord(name[0]) % 5) + 1}"
    return re.sub(r"(\[STATE[^\]]*\])(.*)", lambda m: m.group(1) + re.sub(r"(\w+) = (\d+)", _shift, m.group(2)), text)


def _null_state_line(text: str) -> str:
    """PLACEBO (inert): the ledger still APPEARS (same header + variable names) but every value reads
    UNAVAILABLE — no information, no misinformation. Tests whether the form/wipe helps WITHOUT content
    (vs the nocebo, which tests whether content is used). Engine scores against TRUE values regardless."""
    import re
    return re.sub(r"(\[STATE[^\]]*\])(.*)", lambda m: m.group(1) + re.sub(r"(\w+) = (\d+)", lambda mm: f"{mm.group(1)} = UNAVAILABLE", m.group(2)), text)


def run_session(
    maze_url: str,
    base_url: str,
    model: str,
    deg_id: str,
    no_think: bool,
    verbose: bool,
    inject_history: bool = False,
    kos_prompt: bool = False,
    stateless: bool = False,
    options: dict | None = None,
    fog_radius: int | None = None,
    show_recall: bool = False,
    show_state: bool = False,
    overlay_only: bool = False,
    corrupt_ledger: bool = False,
    null_ledger: bool = False,
    pull_state: bool = False,
    state_stub: bool = False,
    state_label: str = "",
    arm: str | None = None,
    mem_ingest_url: str = "",
    deg_variant: str = "v0",
    run_index: int = 0,
    macguffin_slot: str = "",
    look_gate: bool = False,
    recommend_observe: bool = False,
    observe_cap: bool = False,
    context_policy_name: str | None = None,
    policy_code_ref: str | None = None,
    n_ctx_slot: int | None = None,
    api_key: str | None = None,
) -> dict:
    if overlay_only:
        stateless = True  # overlay-only = wipe the model's context each turn; the HUD is the entire context
    _maybe_corrupt = (_corrupt_state_line if corrupt_ledger
                      else _null_state_line if null_ledger
                      else (lambda t: t))
    client = httpx.Client(base_url=maze_url, timeout=60.0)
    llm = httpx.Client(base_url=base_url,
                       timeout=httpx.Timeout(_LLM_TIMEOUT_SECS, connect=30.0),
                       headers={"Authorization": f"Bearer {api_key}"} if api_key else None)
    t_start = time.monotonic()

    # Create session
    _sess_body = {"deg_id": deg_id, "model": model}
    if fog_radius is not None:
        _sess_body["fog_radius"] = fog_radius
    if show_recall:
        _sess_body["show_recall"] = True
    if show_state:
        _sess_body["show_state"] = True
    if pull_state:
        _sess_body["allow_pull"] = True
    if state_stub:
        _sess_body["state_stub"] = True
    if state_label:
        _sess_body["state_label"] = state_label
    resp = client.post("/session", json=_sess_body)
    resp.raise_for_status()
    session = resp.json()
    session_id = session["session_id"]
    step_budget = session["step_budget"]
    briefing = session.get("briefing", "")

    print(f"  Session {session_id[:8]}  DEG={deg_id}  budget={step_budget}")

    # The observe-cap (prereg 15) is decoupled from the interceptor: --look-gate implies it (forced
    # arm, unchanged), and --observe-cap enables it standalone (recommended/none arms) so termination
    # is identical across all three observe-policies.
    observe_cap = observe_cap or look_gate
    sys_prompt = ("/no_think\n\n" if no_think else "") + build_system_prompt(
        briefing, pull_state=pull_state, state_label=state_label, recommend_observe=recommend_observe)

    # ── Cross-run memory faculty (LB Design 2) — RETRIEVAL HOOK ───────────────────
    # Inject this arm's notes from past runs into the SYSTEM PROMPT (cross-run context is
    # semantically a briefing). The system prompt is identical across the within-run regimes
    # (overlay/stateless/default all rebuild call_messages from sys_prompt) and survives the
    # overlay-only wipe — so the cross-run axis stays orthogonal to the within-run HUD axis.
    # A0 returns "" and never calls /search (asserted via memory_retrievals==0).
    mem_client = None
    mem_debug: dict = {"arm": arm, "retrievals": 0}
    mem_written = False
    if arm:
        mem_client = accum_mem.MemoryClient(mem_ingest_url or accum_mem.DEFAULT_INGEST_URL)
        mem_block, mem_debug = accum_mem.retrieve_memory_block(
            mem_client, arm, accum_mem.build_query(deg_id, briefing), deg_id)
        if mem_block:
            sys_prompt = sys_prompt + "\n\n" + mem_block

    # Pluggable context policy (lb-post-release chunk 02). None = legacy flag-based construction
    # below (overlay_only/stateless/else), fully untouched — every already-published historical
    # arm keeps its exact code path. `messages` still exists for the legacy branches; a policy
    # owns its own state instead (AccumulatePolicy._messages, WipeCuratedPolicy needs none).
    policy: context_policy.ContextPolicy | None = None
    if context_policy_name:
        policy = context_policy.make_policy(context_policy_name, sys_prompt)

    messages = [{"role": "system", "content": sys_prompt}]

    # Bootstrap with observe
    obs = client.post("/act", json={"session_id": session_id, "action": "observe"})
    obs.raise_for_status()
    obs_data = obs.json()
    current_engine_text = _maybe_corrupt(obs_data["text"])
    if policy is not None:
        policy.seed(current_engine_text)
    elif not stateless:
        messages.append({"role": "user", "content": current_engine_text})

    turn = 0
    truncated_turns = 0   # turns the model was cut off mid-generation (finish_reason=length)
    norm_action_count = 0
    look_gate_interceptions = 0
    # Look-gate arm (the cheapest-instrument baseline): a deterministic pre-commit interceptor
    # that forbids answering a gate at a node not observed since arrival — the 2026-07-05 failure
    # postmortem found 46/48 wrong answers were unobserved guesses. observed_here starts True
    # because the loop bootstraps an observe above (line ~460), same as every control run.
    observed_here = True
    # Consecutive-observe cap (amendment, 2026-07-06): the look-gate's mirror-image pathology is
    # over-observation — at a gate it can't solve, the model re-observes forever (observe is free →
    # never dies, never exits → spins to the turn cap; one run looped 250× at one node / 125 min).
    # Cap consecutive observes at a node (reset on commit OR note); on the (cap+1)th, END the episode
    # and score the current depth (froze at gate K = climbed K-1). Empirical: healthy runs peak at
    # 3 consecutive observes, so 5 is safe headroom. Active only under --look-gate.
    _OBSERVE_CAP = 5
    consecutive_observes = 0
    observe_loop_terminated = False
    turns_log: list[dict] = []
    decision_history: list[dict] = []
    kos_state: dict = {"confirmed_dead_ends": set()}
    last_observe_paths: list[str] = _parse_available_paths(current_engine_text)
    completed = False
    while not completed:
        turn += 1
        # Call model — context-policy (chunk 02): the named policy owns message construction.
        # Legacy (policy is None): overlay-only wipes to the curated overlay; stateless cold-
        # prompts [NavState+History+Observation]; default accumulates the full conversation.
        turn_snap = None
        turn_telem = None
        if policy is not None:
            turn_snap = context_policy.TurnSnapshot(turn=turn, sys_prompt=sys_prompt, engine_text=current_engine_text)
            call_messages = policy.turn_start(turn_snap)
            # Telemetry reflects what was actually SENT this turn — compute right after turn_start,
            # not at turn_end, and stdout it immediately (extends wali/orchestrator.py's _hud_event
            # live-log pattern; the Boundary-II lesson is that this per-turn line is the only true
            # loss if a campaign dies mid-run and only the raw JSONL is recoverable after).
            turn_telem = policy.telemetry(turn_snap, call_messages)
            print(turn_telem.to_event(), end="")
        elif overlay_only:
            call_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": current_engine_text},
            ]
        elif stateless:
            call_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": _build_stateless_injection(kos_state, decision_history, current_engine_text)},
            ]
        else:
            call_messages = messages
        llm_json = _llm_call(llm, model, call_messages, options=options, think=False if no_think else None)
        _update_heartbeat(base_url)
        usage = llm_json.get("usage")
        choice = llm_json["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason")
        model_text = msg.get("content") or ""
        model_reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""

        # TRUNCATION IS AN INSTRUMENT FAILURE, NOT A MODEL FAILURE (2026-08-22).
        # A thinking model that hits its output ceiling spends the whole budget reasoning and
        # emits NO answer — measured on qwen3.6:27b: max_tokens 100 and 200 both returned
        # finish_reason="length" with empty content. The fallback below would then hand raw
        # reasoning text to _parse_action, which finds no JSON, warns, and injects an observe —
        # so a truncated turn was indistinguishable from the model declining to act, and nothing
        # in this file read finish_reason at all. Record it loudly instead.
        truncated = (finish_reason == "length")
        if truncated:
            print(f"  [turn {turn}] TRUNCATED: hit the output ceiling "
                  f"(finish_reason=length, completion_tokens={(usage or {}).get('completion_tokens')}) "
                  f"— this turn is an INSTRUMENT failure, not a model decision",
                  file=sys.stderr, flush=True)
            truncated_turns += 1

        if not model_text:
            model_text = model_reasoning

        if verbose:
            print(f"  [turn {turn}] model: {model_text[:200]}")

        if policy is None and not stateless:
            messages.append({"role": "assistant", "content": model_text})

        # Parse action
        action = _parse_action(model_text)
        if action is None:
            print(f"  [turn {turn}] WARNING: could not parse JSON from model response — injecting observe")
            action = {"action": "observe"}
        else:
            # Semantic normalizer — map common hallucinated action names to valid ones.
            # High-confidence remaps preserve intent; last-resort falls back to observe.
            MOVE_SYNONYMS    = {"move", "go", "navigate", "walk", "travel", "proceed", "take", "enter"}
            INSPECT_SYNONYMS = {"check_gate", "examine", "inspect_gate", "look_at", "inspect_path", "check_path", "check", "inspect"}
            OBSERVE_SYNONYMS = {"look", "survey", "scan", "view"}
            BACK_SYNONYMS    = {"retreat", "backtrack", "go_back", "return_to", "back_up"}
            NOTE_SYNONYMS    = {"remember", "record", "memo", "memorize"}
            PULL_SYNONYMS    = {"pull_state", "get_state", "read_state", "fetch_state", "query_state",
                                "request_state", "state", "ledger", "get_values", "get_variables", "check_state"}
            # pull only valid when the arm enables it — otherwise it falls to the observe fallback as today
            VALID_ACTIONS    = {"observe", "commit", "note"} | ({"pull"} if pull_state else set())

            act = action.get("action")
            if act in MOVE_SYNONYMS:
                path = (action.get("direction") or action.get("path") or
                        action.get("to") or action.get("destination") or
                        action.get("path_id") or "")
                action = {"action": "commit", "path_id": path, "answer": action.get("answer", "")}
                norm_action_count += 1
            elif act in INSPECT_SYNONYMS:
                # inspect is gone — fold into observe so model gets the gate info it wants
                action = {"action": "observe"}
                norm_action_count += 1
            elif act in OBSERVE_SYNONYMS:
                action = {"action": "observe"}
                norm_action_count += 1
            elif act in BACK_SYNONYMS:
                action = {"action": "commit", "path_id": "back"}
                norm_action_count += 1
            elif act in NOTE_SYNONYMS:
                text = action.get("text") or action.get("content") or action.get("note") or ""
                action = {"action": "note", "text": text}
                norm_action_count += 1
            elif pull_state and act in PULL_SYNONYMS:
                action = {"action": "pull"}
                norm_action_count += 1
            elif act not in VALID_ACTIONS:
                print(f"  [turn {turn}] WARNING: unrecognized action {act!r} — injecting observe")
                action = {"action": "observe"}
                norm_action_count += 1

        # Remap numeric path_id (e.g. "1", "2") to the actual path label from the last observe.
        # Models sometimes confuse gate option numbers with path labels.
        if action.get("action") == "commit":
            pid = str(action.get("path_id") or "")
            if pid.isdigit() and last_observe_paths:
                idx = int(pid) - 1
                if 0 <= idx < len(last_observe_paths):
                    action["path_id"] = last_observe_paths[idx]
                    norm_action_count += 1

        # Look-gate interceptor: forbid answering a gate at a node not observed since arrival.
        # Deterministic instrument, no memory help — replaces the illegal commit with an observe
        # (costs the turn, not a life; same shape as the 400→observe normalization). Gates only
        # answer-bearing commits: movement/back (answer="") is not a guess and passes through.
        if look_gate and action.get("action") == "commit" and str(action.get("answer") or "").strip() and not observed_here:
            look_gate_interceptions += 1
            if verbose:
                print(f"  [turn {turn}] LOOK-GATE: answer at unobserved node — injecting observe")
            action = {"action": "observe"}

        # Consecutive-observe cap: end the episode if the model would observe an (cap+1)th time in a
        # row without committing/noting — it has demonstrably frozen at this gate. Score the current
        # depth via /state below (froze at gate K = climbed K-1), tagged failure_reason=observe_loop.
        # Checked here (after the interceptor) so interceptor-injected observes count toward the cap.
        if observe_cap and action.get("action") == "observe" and consecutive_observes >= _OBSERVE_CAP:
            observe_loop_terminated = True
            print(f"  [turn {turn}] OBSERVE-CAP: {consecutive_observes} consecutive observes at one node — ending episode (observe_loop)")
            break

        # Dispatch — coerce None/int values to str so FastAPI doesn't 422
        def _s(v, default=""):
            return str(v) if v is not None else default
        last_user = next(
            (m["content"] for m in reversed(call_messages) if m.get("role") == "user"),
            "",
        )
        act_payload = {
            "session_id": session_id,
            "action": _s(action.get("action")) or "observe",
            "path_id": _s(action.get("path_id")),
            "answer": _s(action.get("answer")),
            "text": _s(action.get("text")),
            "injected_context": last_user,
        }
        try:
            act_resp = client.post("/act", json=act_payload)
            act_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                print(f"  [turn {turn}] WARNING: 400 from /act for action {act_payload.get('action')!r} — injecting observe")
                norm_action_count += 1
                fallback = client.post("/act", json={"session_id": session_id, "action": "observe", "path_id": "", "answer": "", "text": ""})
                fallback.raise_for_status()
                act_data = fallback.json()
                fallback_text = _maybe_corrupt(act_data.get("text", ""))
                if policy is not None:
                    # Mirrors the legacy default path's combined effect (assistant appended earlier
                    # in the turn + user=fallback_text appended here) as ONE turn_end call — see the
                    # module docstring on why the combined call is behavior-identical.
                    policy.turn_end(context_policy.TurnSnapshot(
                        turn=turn, sys_prompt=sys_prompt, engine_text=fallback_text, model_text=model_text))
                turns_log.append({
                    "turn": turn, "model_text": model_text, "model_reasoning": model_reasoning,
                    "action_parsed": action, "engine_text": f"[400→observe] {fallback_text}",
                    "usage": usage,
                    "context_telemetry": _asdict_or_none(turn_telem),
                })
                current_engine_text = fallback_text
                observed_here = True  # the fallback dispatched an observe
                consecutive_observes += 1  # counts toward the observe-cap like any other observe
                if policy is None and not stateless:
                    messages.append({"role": "user", "content": fallback_text})
                completed = act_data.get("completed", False)
                continue
            raise
        act_data = act_resp.json()

        # Track observation state for the look-gate: an observe/pull reveals the current node;
        # a commit moves (or bounces) us to a node we must re-observe before answering its gate.
        # consecutive_observes drives the observe-cap: it resets on any commit OR note (real
        # progress at a node), so observe→note→observe→commit never trips it — only a pure
        # observe run does.
        _dispatched = action.get("action")
        if _dispatched in ("observe", "pull"):
            observed_here = True
            consecutive_observes += 1
        elif _dispatched == "commit":
            observed_here = False
            consecutive_observes = 0
        elif _dispatched == "note":
            consecutive_observes = 0

        engine_text = _maybe_corrupt(act_data.get("text", ""))
        if verbose:
            print(f"  [turn {turn}] engine: {engine_text[:200]}")

        # Build decision history entry for commit actions (harness-side, engine unchanged)
        if (inject_history or stateless) and action.get("action") == "commit":
            loc = _parse_location(engine_text)
            is_dead = bool(loc and "DEAD_END" in engine_text)
            path = action.get("path_id", "")
            ans = action.get("answer", "")
            act_str = f'commit "{path}"' + (f' answer="{ans}"' if ans else "")
            decision_history.append({"turn": turn, "action_str": act_str, "location": loc, "dead_end": is_dead})

        # Update KOS state: track confirmed dead ends after commit actions
        if (kos_prompt or stateless) and action.get("action") == "commit":
            loc = _parse_location(engine_text)
            if loc and _is_dead_end(engine_text):
                kos_state["confirmed_dead_ends"].add(loc)

        current_engine_text = engine_text
        if "--- OBSERVE ---" in engine_text:
            last_observe_paths = _parse_available_paths(engine_text)

        if policy is not None:
            # Combined turn_end (assistant + this turn's observation) — see the module docstring
            # on why one call replicates the two separate legacy append() sites exactly.
            policy.turn_end(context_policy.TurnSnapshot(
                turn=turn, sys_prompt=sys_prompt, engine_text=engine_text, model_text=model_text))
            turns_log.append({
                "turn": turn, "model_text": model_text, "model_reasoning": model_reasoning,
                "action_parsed": action, "engine_text": engine_text,
                "usage": usage, "context_telemetry": _asdict_or_none(turn_telem),
                "truncated": truncated,
            })
        elif stateless:
            turns_log.append({"turn": turn, "model_text": model_text, "model_reasoning": model_reasoning, "action_parsed": action, "engine_text": engine_text, "injected_history": None, "truncated": truncated})
        else:
            history_block = _build_history_block(decision_history) if inject_history else ""
            if kos_prompt:
                user_content = _build_kos_state_block(kos_state) + engine_text
            else:
                user_content = engine_text + history_block
            turns_log.append({"turn": turn, "model_text": model_text, "model_reasoning": model_reasoning, "action_parsed": action, "engine_text": engine_text, "injected_history": history_block or None, "truncated": truncated})
            messages.append({"role": "user", "content": user_content})
        completed = act_data.get("completed", False)

        if completed:
            break

        # overlay-only / any policy that cold-prompts each turn: refresh the overlay (map+recall+
        # node) for the next turn after a state-changing action, so the HUD the model sees is
        # always current. Pull is exempt — its response IS a current view (STATE + observe) and
        # must survive exactly one turn (the one-shot pull semantic: pull, then use it or lose it).
        if (overlay_only or (policy is not None and policy.needs_observe_refresh)) and action.get("action") not in ("observe", "pull"):
            _ob = client.post("/act", json={"session_id": session_id, "action": "observe", "path_id": "", "answer": "", "text": ""})
            if _ob.status_code == 200:
                current_engine_text = _maybe_corrupt(_ob.json().get("text", current_engine_text))
                if "--- OBSERVE ---" in current_engine_text:
                    last_observe_paths = _parse_available_paths(current_engine_text)

        # Safety: don't spin past 3× the step budget in model turns
        if turn > step_budget * 3:
            print(f"  WARNING: turn limit hit without completion")
            break

    # Retrieve score
    score_resp = client.get(f"/score/{session_id}")
    if score_resp.status_code == 400:
        # Not completed (observe-cap or turn cap) — score deterministically from /state so an
        # incomplete run keeps its real ramp_depth (gates passed) instead of a None. ramp_depth =
        # len(gate_results), matching runner.score(). (Amendment 2026-07-06: previously an
        # incomplete run recorded only {"error": ...} and lost its depth — the Wali arm hit the
        # same gap and worked around it in run_wali.py.)
        st = client.get(f"/session/{session_id}/state")
        state = st.json() if st.status_code == 200 else {}
        score_data = {
            "session_id": session_id,
            "found_exit": False,
            "failure_reason": "observe_loop" if observe_loop_terminated else "turn_limit_hit",
            "steps_to_exit": None,
            "ramp_depth": len(state.get("gate_results", {})),
            "step_budget": state.get("step_budget", step_budget),
            "score_source": "state_incomplete",
        }
    else:
        score_resp.raise_for_status()
        score_data = score_resp.json()

    score_data["model"] = model
    score_data["deg_id"] = deg_id
    score_data["turns"] = turn
    score_data["elapsed_seconds"] = round(time.monotonic() - t_start, 2)
    score_data["normalized_actions"] = norm_action_count
    score_data["look_gate"] = look_gate
    score_data["recommend_observe"] = recommend_observe
    score_data["observe_cap"] = observe_cap
    score_data["look_gate_interceptions"] = look_gate_interceptions
    score_data["look_gate_observe_cap"] = _OBSERVE_CAP if observe_cap else None
    score_data["observe_loop_terminated"] = observe_loop_terminated
    score_data["inject_history"] = inject_history
    score_data["kos_prompt"] = kos_prompt
    score_data["stateless"] = stateless
    score_data["pull_state"] = pull_state
    score_data["state_stub"] = state_stub
    score_data["state_label"] = state_label
    # Non-zero means the model was cut off mid-generation on that many turns, so those turns are
    # instrument artifacts, not model decisions. A row with truncated_turns > 0 must not be read
    # as a clean measurement of the arm (2026-08-22).
    score_data["truncated_turns"] = truncated_turns
    score_data["turns_log"] = turns_log

    # Provenance columns (lb-post-release chunk 02 — the standing gate): base_url is always
    # known; n_ctx_slot is the operator-supplied journal-verified value (never the CLI flag —
    # ollama's /v1 endpoint silently drops --num-ctx, same class as the documented `think` drop).
    score_data["base_url"] = base_url
    score_data["n_ctx_slot"] = n_ctx_slot

    # Context-policy provenance + generality class (leaderboard integrity — Will, 2026-07-14):
    # a run using the new interface declares which policy ran, its generality class, and the
    # inspectable code that produced it — "no answer-key smuggling" rides on this, not on trust.
    score_data["context_policy"] = context_policy_name
    score_data["policy_provenance"] = (
        context_policy.policy_provenance(context_policy_name, policy_code_ref)
        if context_policy_name else None
    )
    score_data["policy_summary"] = policy.task_end() if policy is not None else None

    # ── Cross-run memory faculty (LB Design 2) — WRITE-BACK HOOK ──────────────────
    # EVERY arm (incl. A0) writes the byte-identical deterministic record so A0/A1/A2 share an
    # identical store; only the read policy varies (flattery audit A-5). A0's store is write-only.
    # var_ledger from /session/{id}/state carries the forged constants (e.g. K) the dam reads.
    score_data["arm"] = arm
    score_data["memory_retrievals"] = mem_debug.get("retrievals", 0)
    score_data["memory_debug"] = mem_debug
    if arm and mem_client is not None:
        try:
            st = client.get(f"/session/{session_id}/state")
            session_state = st.json() if st.status_code == 200 else {}
        except Exception:
            session_state = {}
        session_state.setdefault("session_id", session_id)
        record = accum_mem.build_record(arm, deg_id, deg_variant, run_index,
                                        score_data, session_state, macguffin_slot or None)
        try:
            mem_client.ingest_record(record)
            mem_written = True
        except Exception as e:
            # Loud + unmissable: a dropped write silently breaks the cross-run learning curve.
            print(f"  !!! [mem] WRITE-BACK FAILED (arm={arm} run={run_index}): {e}")
        mem_client.close()
    score_data["memory_written"] = mem_written
    return score_data


def main():
    ap = argparse.ArgumentParser(description="LabyrinthBench CLI harness")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--maze-url", default="http://localhost:8090")
    ap.add_argument("--deg", default="alpha-1")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--run-offset", type=int, default=0,
                    help="Starting run_index offset. Lets --runs 1 invocations CONTINUE an arm's "
                         "learning curve across separate (interleaved) calls — the cross-run store is "
                         "stateful in the DB, so run_index just needs to keep counting. The hardened "
                         "interleaved driver (design2/gate_hardened.sh) passes --run-offset k.")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--output", default="/results/results.jsonl")
    ap.add_argument("--db-url", default=_DEFAULT_DB_URL)
    ap.add_argument("--label", default=None)
    # ── LB Design 2 accumulation eval — the cross-run memory axis ────────────────
    ap.add_argument("--arm", choices=["A0", "A1", "A2", "A2W", "A2R", "A3", "A4"], default=None,
                    help="Cross-run KOS memory arm. Omit = no cross-run memory (legacy behavior). "
                         "A0 control (write-only, never reads), A1 naive, A2 organism (route+dam+fork). "
                         "A2W = A2 + wide retrieval window (un-starve the dam); A2R = A2W + recency sort.")
    ap.add_argument("--mem-ingest-url", default=accum_mem.DEFAULT_INGEST_URL,
                    help="ingestion-worker base URL for the memory faculty (read/write memory store).")
    ap.add_argument("--deg-variant", default="v0", help="Variant tag stored in the run-record.")
    ap.add_argument("--macguffin-slot", default="",
                    help="fact-slot URI for the currency-dam stale-twin test (Gate 4), e.g. fact://lab/macguffin/k")
    ap.add_argument("--inject-history", action="store_true",
                    help="Append harness-tracked decision history to each engine response (engine unchanged)")
    ap.add_argument("--kos-prompt", action="store_true",
                    help="Prepend structured navigation state (confirmed dead ends) before each observation")
    ap.add_argument("--stateless", action="store_true",
                    help="Wipe model context between turns; inject cold [NavState+History+Observation] prompt each turn")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Ollama num_ctx (KV cache size). Use 16384 for phi4-reasoning.")
    ap.add_argument("--fog-radius", type=int, default=None,
                    help="Override the DEG's fog-of-war radius (awareness ladder: 0=blind, 2=default map).")
    ap.add_argument("--show-recall", action="store_true",
                    help="Externalize recorded gate answers into the overlay (HUD-as-working-memory arm).")
    ap.add_argument("--show-state", action="store_true",
                    help="Externalize the CURRENT variable ledger into the overlay (belief-revision arm). "
                         "Shows only current bindings (latest sets_var wins) — NOT the raw gate dump, so stale "
                         "values never re-enter the overlay. Use with --overlay-only for the managed arm.")
    ap.add_argument("--overlay-only", action="store_true",
                    help="Wipe model context every turn; cold-prompt with ONLY the curated overlay (map+recall+node). "
                         "Total context management — the HUD is the entire context. Use with --fog-radius + --show-recall.")
    ap.add_argument("--corrupt-ledger", action="store_true",
                    help="NOCEBO/falsification: show the model a per-variable-shifted (WRONG) [STATE] ledger while "
                         "scoring against TRUE values. If it still passes, the ledger wasn't load-bearing (artifact). "
                         "Use with --show-state.")
    ap.add_argument("--null-ledger", action="store_true",
                    help="PLACEBO (inert): the [STATE] ledger appears but every value reads UNAVAILABLE — tests "
                         "whether the form/wipe helps WITHOUT content. Use with --show-state.")
    ap.add_argument("--pull-state", action="store_true",
                    help="Pull-HUD: enable {\"action\": \"pull\"} — the model requests the full current [STATE] "
                         "ledger on demand. Costs one step. In overlay-only mode the pull response survives "
                         "exactly one turn (use it or lose it). Use with --overlay-only for the managed-pull arm.")
    ap.add_argument("--state-stub", action="store_true",
                    help="Hybrid arm: push a one-line stub into the overlay — tracked variable NAMES only, plus "
                         "how to pull values. Requires --pull-state.")
    ap.add_argument("--state-label", choices=["", "verified"], default="",
                    help="Epistemic-label arm: 'verified' flips the authority label — VERIFIED [STATE] header + "
                         "memory-status disclosure in the system prompt. Requires --pull-state.")
    ap.add_argument("--look-gate", action="store_true",
                    help="Look-gate arm (cheapest-instrument baseline): deterministically intercept an "
                         "answer-bearing commit at a node not observed since arrival, replacing it with an "
                         "observe (costs the turn, not a life). Implies --observe-cap. No memory help.")
    ap.add_argument("--recommend-observe", action="store_true",
                    help="Recommended-observe policy (prereg 15): add the locked observe-first rule to the "
                         "system prompt ('Before you answer any gate, observe it — each gate's problem is "
                         "unique.'). Instruction only, no enforcement.")
    ap.add_argument("--observe-cap", action="store_true",
                    help="End the episode + score from /state after 5 consecutive observes at a node "
                         "(the observe-loop pathology guard). Decoupled from --look-gate so all observe-"
                         "policies (prereg 15) share identical termination. --look-gate implies it.")
    # ── Pluggable context policy (lb-post-release chunk 02) ───────────────────────
    ap.add_argument("--context-policy", choices=sorted(context_policy.POLICIES), default=None,
                    help="Named ContextPolicy (cli/context_policy.py) — arms as config, not a "
                         "harness fork. wipe-curated/accumulate reproduce --overlay-only/default; "
                         "the rest are stubs that raise until their owning chunk lands. Mutually "
                         "exclusive with --overlay-only/--stateless/--inject-history/--kos-prompt.")
    ap.add_argument("--policy-code-ref", default=None,
                    help="Repo URL/commit for the exact policy code used this run (leaderboard "
                         "integrity, Will 2026-07-14). Auto-derived from this checkout's HEAD when "
                         "omitted for built-in policies.")
    ap.add_argument("--n-ctx-slot", type=int, default=None,
                    help="Journal-verified n_ctx_slot for this run's base_url host (operator-"
                         "supplied — see scripts/e1a-run-row.sh for the SSH+grep recipe). Never "
                         "auto-detected: the --num-ctx flag is silently dropped by ollama's /v1 "
                         "endpoint, so it cannot be trusted as ground truth.")
    ap.add_argument("--api-key", default=os.environ.get("LB_LLM_API_KEY"),
                    help="API key sent as 'Authorization: Bearer <key>' on every request to "
                         "--base-url (e.g. for key-authed cloud/gateway endpoints). Local Ollama/"
                         "LM Studio ignore it. Defaults to $LB_LLM_API_KEY; never logged or "
                         "persisted into --output/--db-url.")
    args = ap.parse_args()

    if (args.state_stub or args.state_label) and not args.pull_state:
        ap.error("--state-stub/--state-label require --pull-state")

    if args.context_policy and (args.overlay_only or args.stateless or args.inject_history or args.kos_prompt):
        ap.error("--context-policy is mutually exclusive with --overlay-only/--stateless/"
                 "--inject-history/--kos-prompt — those stay on the untouched legacy path.")

    llm_options = {"num_ctx": args.num_ctx} if args.num_ctx else None

    output_path = Path(args.output)
    results = []

    _acquire_lock(args.model, args.deg, args.runs, args.base_url)

    # Serving-stack identity, captured ONCE per invocation and stamped onto every row below.
    # Once, not per session: the stack cannot change mid-invocation, and the probe is HTTP the
    # scored run should not be paying for. Wrapped because describing a run must never fail it.
    try:
        prov = provenance.capture(args.base_url, args.model, api_key=args.api_key)
        print(provenance.summary(prov))
    except Exception as e:  # pragma: no cover — belt and braces; capture() already swallows
        prov = {"error": str(e)[:200]}
        print(f"  ! provenance capture failed (non-fatal): {e}")

    try:
        for i in range(args.runs):
            print(f"Run {i + 1}/{args.runs} (run_index={args.run_offset + i})")
            try:
                result = run_session(
                    maze_url=args.maze_url,
                    base_url=args.base_url,
                    model=args.model,
                    deg_id=args.deg,
                    no_think=args.no_think,
                    verbose=args.verbose,
                    inject_history=args.inject_history,
                    kos_prompt=args.kos_prompt,
                    stateless=args.stateless,
                    options=llm_options,
                    fog_radius=args.fog_radius,
                    show_recall=args.show_recall,
                    show_state=args.show_state,
                    overlay_only=args.overlay_only,
                    corrupt_ledger=args.corrupt_ledger,
                    null_ledger=args.null_ledger,
                    pull_state=args.pull_state,
                    state_stub=args.state_stub,
                    state_label=args.state_label,
                    arm=args.arm,
                    mem_ingest_url=args.mem_ingest_url,
                    deg_variant=args.deg_variant,
                    run_index=args.run_offset + i,
                    macguffin_slot=args.macguffin_slot,
                    look_gate=args.look_gate,
                    recommend_observe=args.recommend_observe,
                    observe_cap=args.observe_cap,
                    context_policy_name=args.context_policy,
                    policy_code_ref=args.policy_code_ref,
                    n_ctx_slot=args.n_ctx_slot,
                    api_key=args.api_key,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                result = {"model": args.model, "deg_id": args.deg, "error": str(e)}

            if args.label:
                result["run_label"] = args.label
            # Stamped on EVERY row, including error/DNF rows: a run that failed is exactly the one
            # you later need to attribute to a serving stack.
            result["prov"] = prov
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if args.db_url and "error" not in result:
                _insert_run(args.db_url, result, args.label)

            status = "EXIT ✓" if result.get("found_exit") else "DNF ✗"
            steps = result.get("steps_to_exit", "—")
            opt = result.get("optimal_commits", "?")
            gate_acc = result.get("gate_accuracy")
            gate_str = f"{gate_acc:.0%}" if gate_acc is not None else "n/a"
            line = f"  {status}  steps={steps}  optimal={opt}  gate_acc={gate_str}"
            if args.pull_state:
                line += f"  pulls={result.get('pull_count', '?')}"
            if result.get("chain_gate_count"):
                ca, kc = result.get("chain_accuracy"), result.get("knowledge_state_consistency")
                line += f"  chain_acc={f'{ca:.0%}' if ca is not None else 'n/a'}"
                line += f"  consistency={f'{kc:.0%}' if kc is not None else 'n/a'}"
            print(line)
            # A DNF (budget exhausted / trapped / out-of-lives — anything short of found_exit) used
            # to print only the same one-line status above, indistinguishable from a healthy run at
            # a glance and invisible to anything that only checks the exit code. Loud stderr here so
            # an unreachable endpoint or a broken harness doesn't read as quiet success (an "error"
            # result already gets its own ERROR line above — don't double-print for that case).
            if not result.get("found_exit") and "error" not in result:
                print(
                    f"  DNF ✗  reason={result.get('failure_reason', 'unknown')}"
                    f"  turns={result.get('turns', '?')}  ramp_depth={result.get('ramp_depth', '?')}",
                    file=sys.stderr,
                )
    finally:
        _release_lock(args.base_url)

    # Summary
    n = len(results)
    found = sum(1 for r in results if r.get("found_exit"))
    steps_list = [r["steps_to_exit"] for r in results if r.get("found_exit") and r.get("steps_to_exit")]
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else None

    print(f"\n--- Summary ({args.model}, DEG={args.deg}) ---")
    print(f"  Exit rate:  {found}/{n} ({found/n:.0%})")
    if avg_steps:
        print(f"  Avg steps:  {avg_steps:.1f}  (optimal={results[0].get('optimal_commits', '?')})")
    # chain-reasoning aggregates (only meaningful on dependent-chain DEGs)
    ca_vals = [r["chain_accuracy"] for r in results if r.get("chain_accuracy") is not None]
    kc_vals = [r["knowledge_state_consistency"] for r in results if r.get("knowledge_state_consistency") is not None]
    if ca_vals:
        print(f"  Chain acc:  {sum(ca_vals)/len(ca_vals):.0%}  (mean over {len(ca_vals)} runs w/ attempted chain gates)")
    if kc_vals:
        print(f"  Knowledge-state consistency: {sum(kc_vals)/len(kc_vals):.0%}  (executed the program vs guessed)")
    print(f"  Results written to: {output_path}")

    # Process exit code: every run above ALWAYS executes and gets written to --output regardless of
    # outcome (a DNF never truncates a multi-run aggregate flow — the loop has no early-exit on a
    # bad result), so a real campaign's partial DNF rate is data, not a harness fault, and stays
    # exit 0. But the process used to report success (exit 0) even when NOTHING found the exit —
    # for --runs 1 that's simply "the run DNF'd" reported as a pass. Zero successes across the
    # whole batch is unambiguous: fail loud.
    if found == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
