"""LabyrinthBench CLI harness.

Runs a model through one or more DEG sessions and reports scores.

Usage:
  python run_eval.py --model qwen3.5:2b --base-url http://localhost:11434/v1 [options]

Options:
  --model         Model name (passed in the messages body)
  --base-url      OpenAI-compatible base URL (default: $LB_BASE_URL, else
                  http://localhost:11434/v1). Point this at any OpenAI-compatible
                  endpoint — Ollama, LM Studio, llama.cpp, or a multi-model gateway
                  fronting several of them. Set $LB_BASE_URL once in your environment
                  instead of repeating --base-url on every invocation; the CLI flag
                  always wins when both are given.
  --lock-host     Operator-supplied override for the per-host VRAM run-lock key (see
                  "Locking through a gateway" below). Default: $LB_LOCK_HOST, else the
                  hostname parsed out of --base-url (correct when --base-url IS the
                  physical host; wrong when it's a gateway multiplexing several).
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

Locking through a gateway:
  The per-host VRAM lock (see _lock_path below) exists to stop two concurrent runs from
  fighting over the SAME physical GPU's VRAM. It keys on --base-url's hostname, which is
  correct when --base-url names one physical machine. It stops being correct once
  --base-url names a multi-upstream gateway (several models, several physical hosts,
  one hostname): every run through that gateway would then share ONE lock key regardless
  of which hardware actually serves its model, serializing unrelated runs that don't
  actually contend for the same VRAM. This is a conservative failure (extra
  serialization, never silent corruption or a missed lock), and is exactly why
  --lock-host exists as an explicit operator override — same pattern as --n-ctx-slot:
  the CLI cannot discover ground truth here on its own, so it takes an operator's word
  for it rather than guessing. Passing --lock-host also flips the `via_gateway` flag
  stamped into this run's provenance (see provenance.capture()) — the run-record answers
  "did this go through a gateway?" from data, not from memory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

# point-and-click chunk 01 (forced-choice arm): the engine package is a sibling of cli/, not a
# dependency of it today — this is the first cli/*.py file to import from engine/ for anything
# beyond run_oracle.py's already-established pattern (BFS solver reads the DEG directly). Needed
# so the forced-choice harness can build a commit menu's candidate answers locally (the HTTP API
# never exposes a gate's answer — see api/main.py's ActRequest/render_observe — by design).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.distractors import NONE_OF_THESE, generate_distractors  # noqa: E402
from engine.graph import DEG, load_deg  # noqa: E402

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
# Runner v2 (MCV chunk 04 owed item, read out 2026-09-09): a mid-campaign HTTP 500 from the
# model upstream used to propagate straight out of _llm_call, uncaught by the connection-error
# retry loop below, and lose the whole row. One retry after a short settle, same idea as the
# connection-error backoff but a fixed single attempt, not a counted loop — a second consecutive
# 500 is treated as a real server-side failure, not a blip.
_HTTP_500_SETTLE_SECONDS = float(os.environ.get("LB_HTTP_500_SETTLE", "15"))


def _lock_path(base_url: str, lock_host: str | None = None) -> Path:
    # lock_host is the operator's --lock-host/$LB_LOCK_HOST override — see the module
    # docstring's "Locking through a gateway" section. None preserves the original
    # behavior byte-for-byte: derive the key from --base-url's own hostname.
    host = lock_host or (urlparse(base_url).hostname or "local")
    return Path(f"/results/.eval_lock_{host.replace('.', '_')}")


def _acquire_lock(model, deg, runs, base_url, lock_host: str | None = None):
    lock = _lock_path(base_url, lock_host)
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
                    f"ERROR: eval already running on {lock_host or urlparse(base_url).hostname} — "
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


def _update_heartbeat(base_url: str, lock_host: str | None = None) -> None:
    lock = _lock_path(base_url, lock_host)
    if not lock.exists():
        return
    try:
        data = json.loads(lock.read_text())
        data["last_heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        lock.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _release_lock(base_url: str, lock_host: str | None = None):
    _lock_path(base_url, lock_host).unlink(missing_ok=True)


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


# ── Forced-choice arm (point-and-click chunk 01) ──────────────────────────────────────────────
# Response mode is an axis ORTHOGONAL to context-policy (accumulate/wipe-curated): generative
# free-generates a JSON action; forced-choice never generates an action at all — the harness
# enumerates every legal action (plus, for a gated path, the correct answer and its deterministic
# distractors from engine/distractors.py) as a labeled menu, and the model's ENTIRE contribution
# is a single-token label pick, scored by argmax over that label's logprobs (or, when the upstream
# doesn't expose logprobs, by constrained GBNF-grammar decoding to the label alphabet — see
# _llm_call's `grammar` param and --fc-selection below). This is the "recognition, not recall"
# axis, not the context-management axis: forced-choice runs under EITHER context policy unchanged
# (see run_session's response_mode branch, which only replaces how THIS turn's `action` dict is
# decided — call_messages construction, dispatch, scoring, and turns_log/score_data plumbing are
# fully shared with generative mode).

# Label-set rotation (label-token bias control — chunk 01 Design/Controls): rotated by run_index,
# not by a flag, so an operator running --runs N already gets N different label sets for free.
FC_LABEL_SETS: list[list[str]] = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["P", "Q", "R", "S", "T", "U", "V", "W"],
    ["1", "2", "3", "4", "5", "6", "7", "8"],
]

_FC_JSON_TAIL = (
    'Respond with ONLY the single label character of your choice (e.g. "C"). '
    "No JSON, no punctuation, no explanation — just the one character."
)


def build_forced_choice_system_prompt(briefing: str = "") -> str:
    """Forced-choice's own system prompt. The objective/rules framing carries over, but the
    JSON-action instructions (SYSTEM_MECHANICS / _JSON_TAIL) do not apply — there is no JSON to
    emit, only a menu label. Kept as a SEPARATE builder (not a branch inside build_system_prompt)
    so generative mode's prompt construction is provably untouched by this arm's existence."""
    framing = briefing.strip() if briefing.strip() else DEFAULT_STRATEGY
    mech = (
        "Your objective: reach EXIT. That is the only goal.\n\n"
        "Each node has paths forward. Paths are open (free to take) or gated (solve a problem to "
        "proceed).\n\n"
        "Every turn you are shown a MENU of labeled options: the legal actions here, and — for a "
        "gated path — the candidate answers to its problem. Pick exactly ONE label.\n\n"
        "Rules:\n"
        "- Exhaust your step budget without reaching EXIT = failure.\n"
        '- If none of the candidate answers looks right, the menu always includes a "none of '
        'these" option rather than forcing a guess among them.'
    )
    return f"{mech}\n\n{framing}\n\n{_FC_JSON_TAIL}\n"


@dataclass
class _FCState:
    """Forced-choice's local mirror of engine state — loaded from the SAME degs/*.yaml the API
    server reads, tracked in lockstep with it turn by turn. Needed because a commit menu's
    candidate VALUES require the gate object plus the resolved values of its dependencies, and
    the HTTP API never exposes an answer (by design — api/main.py's ActRequest/render_observe show
    a gate's PROBLEM, never its answer). nav-3's gates are all LOCKS (wrong_destination=None), so
    a wrong pick never moves the real session and this mirror cannot diverge from it by
    construction: the model can only ever be AT node k once gates 1..k-1 were genuinely passed,
    which is exactly what advances this mirror too. Kept updated by the turn loop itself (see
    run_session's forced-choice branch and its post-dispatch mirror update)."""
    deg: DEG
    current_node_id: str
    gate_results: dict = field(default_factory=dict)
    var_ledger: dict = field(default_factory=dict)
    traversal_stack: list = field(default_factory=list)


@dataclass
class _FCOption:
    action: dict                       # {"action":..., "path_id":..., "answer":...} — ready to dispatch
    gate_id: str | None = None         # the gate this option answers (None for observe/back/open-move)
    is_correct: bool | None = None     # True/False for a candidate-answer option; None otherwise


@dataclass
class _FCMenu:
    labels: list        # ordered, already-shuffled label tokens, e.g. ["C", "A", "E", ...]
    options: dict        # label -> _FCOption


def _fc_build_menu(state: _FCState, rng: "random.Random", label_set: list,
                    max_distractors: int = 3) -> _FCMenu:
    """Build this turn's flat forced-choice menu: {observe} + {one option per path — for a gated
    path, one option PER CANDIDATE (the correct answer, its deterministic distractors, and a
    'none of these' that always dispatches as a genuine wrong commit, matching generative mode's
    failure semantics)} + {back, when the mirror's traversal stack is non-empty}. Position bias
    (Controls) is handled by shuffling BEFORE label assignment; `rng` is caller-owned so the
    caller controls its seed/lifetime across the whole run."""
    node = state.deg.node(state.current_node_id)
    opts: list[_FCOption] = [_FCOption(action={"action": "observe", "path_id": "", "answer": ""})]
    for path in node.paths:
        if path.is_gated:
            expected = path.gate.resolved_answer(state.gate_results, state.var_ledger)
            if expected == "__UNRESOLVABLE__":
                # A dependency hasn't been passed yet — unreachable on nav-3's strictly ordered
                # lock chain (this path cannot be live before its deps are satisfied), but degrade
                # to a single always-wrong placeholder rather than crash a live campaign row.
                opts.append(_FCOption(
                    action={"action": "commit", "path_id": path.id, "answer": NONE_OF_THESE},
                    gate_id=path.gate.gate_id, is_correct=False))
                continue
            distractors = generate_distractors(path.gate, expected, state.gate_results,
                                                state.var_ledger, n=max_distractors)
            for cand in [expected] + distractors:
                opts.append(_FCOption(
                    action={"action": "commit", "path_id": path.id, "answer": cand},
                    gate_id=path.gate.gate_id, is_correct=(cand == expected)))
            opts.append(_FCOption(
                action={"action": "commit", "path_id": path.id, "answer": NONE_OF_THESE},
                gate_id=path.gate.gate_id, is_correct=False))
        else:
            opts.append(_FCOption(action={"action": "commit", "path_id": path.id, "answer": ""}))
    if state.traversal_stack:
        opts.append(_FCOption(action={"action": "commit", "path_id": "back", "answer": ""}))
    rng.shuffle(opts)
    if len(opts) <= len(label_set):
        labels = list(label_set[:len(opts)])
    else:  # pragma: no cover — nav-3 never approaches this (observe + 1 gate's [correct + N
        # distractors + none] + back is well under any label set's length); degrade to reusing
        # the label set cyclically rather than silently dropping options.
        labels = [label_set[i % len(label_set)] for i in range(len(opts))]
    return _FCMenu(labels=labels, options=dict(zip(labels, opts)))


def _fc_render_menu(menu: _FCMenu) -> str:
    lines = ["", "--- CHOOSE ONE ---"]
    for label in menu.labels:
        act = menu.options[label].action
        if act["action"] == "observe":
            desc = "observe (look around; free, no step cost)"
        elif act.get("path_id") == "back":
            desc = "go back to your previous location (costs one step)"
        elif act["action"] == "commit" and act.get("answer") == NONE_OF_THESE:
            desc = f"path {act['path_id']!r}: none of the candidate answers is correct"
        elif act["action"] == "commit" and act.get("answer"):
            desc = f"path {act['path_id']!r}: answer = {act['answer']}"
        else:
            desc = f"path {act['path_id']!r}: take this open path"
        lines.append(f"  {label}) {desc}")
    lines.append("Respond with exactly one label character.")
    return "\n".join(lines)


def _fc_append_menu(call_messages: list, menu_text: str) -> list:
    """Return a NEW messages list with the menu appended to the final message (always a user
    turn, in every context-policy/legacy branch) — never mutates the caller's list in place,
    since accumulate-family policies own their message list as long-lived state."""
    out = [dict(m) for m in call_messages]
    if out and out[-1].get("role") == "user":
        out[-1] = {**out[-1], "content": out[-1]["content"] + "\n" + menu_text}
    else:  # pragma: no cover — defensive; every branch ends in a user turn today
        out.append({"role": "user", "content": menu_text})
    return out


def _fc_build_grammar(labels: list) -> str:
    """Minimal GBNF root rule restricting output to exactly one of `labels` — the constrained-
    decode fallback for an upstream that doesn't expose logprobs (chunk 01's own Preflight step
    determines, live, which path a given upstream needs; see --fc-selection)."""
    alts = " | ".join(f'"{l}"' for l in labels)
    return f"root ::= ({alts})\n"


def _fc_extract_label_probs(llm_json: dict, labels: list) -> dict | None:
    """Best-effort extraction of a per-label probability vector from an OpenAI-compatible chat-
    completions response's `logprobs.content[0].top_logprobs` — llama.cpp's documented shape for
    `/v1/chat/completions`, UNPROBED LIVE as of this authoring (chunk 01's Preflight step 1 is
    exactly this probe, run before the design is frozen). Returns None (never raises) on any
    shape mismatch, so the caller can tell "the upstream didn't give us logprobs this call" from
    "the model answered" and fall back cleanly. Reads the FIRST generated token only — the
    single-token, zero-reasoning read this whole arm is built on (temperature 0, no-think)."""
    try:
        choice = llm_json["choices"][0]
        lp = choice.get("logprobs")
        if not lp or not lp.get("content"):
            return None
        first = lp["content"][0]
        alts = first.get("top_logprobs") or []
        raw: dict[str, float] = {}
        for a in alts:
            tok = (a.get("token") or "").strip().upper()
            lg = a.get("logprob")
            if tok in labels and lg is not None and (tok not in raw or lg > raw[tok]):
                raw[tok] = lg
        if not raw:
            # Some servers report only the sampled token's own logprob, with no alternatives list.
            tok = (first.get("token") or "").strip().upper()
            lg = first.get("logprob")
            if tok in labels and lg is not None:
                raw[tok] = lg
        if not raw:
            return None
        # Renormalize over JUST the label set (softmax over these logprobs) — a label missing
        # from top_logprobs (truncated by the server's top-N) gets a probability FLOOR rather
        # than 0, so a margin/ECE computation never divides by a degenerate vector.
        floor = min(raw.values()) - 10.0
        full = {l: raw.get(l, floor) for l in labels}
        m = max(full.values())
        exps = {l: math.exp(v - m) for l, v in full.items()}
        z = sum(exps.values())
        return {l: v / z for l, v in exps.items()}
    except (KeyError, IndexError, TypeError):
        return None


def _fc_score_turn(llm_json: dict, menu: _FCMenu, selection: str) -> dict:
    """Decode this turn's model response into one menu option, by the requested `selection`
    method ('logprobs' argmax, or 'grammar' — a direct read of the grammar-constrained output
    text). Falls back logprobs -> raw-text parse -> the menu's first label (never crashes a
    campaign row), with the fallback always visible in `selection_method` for later analysis
    (F3: is scoring mode itself unusable on this model?)."""
    prob_vec = None
    method = selection
    if selection == "logprobs":
        prob_vec = _fc_extract_label_probs(llm_json, menu.labels)
        if prob_vec is None:
            method = "logprobs-unavailable-text-fallback"
    if prob_vec is not None:
        label = max(prob_vec, key=prob_vec.get)
    else:
        text = (llm_json["choices"][0]["message"].get("content") or "").strip().upper()
        label = next((l for l in menu.labels if text.startswith(l)), None)
        if label is None:
            label = menu.labels[0]
            method = method + "-undecodable-default"
    sorted_probs = sorted(prob_vec.values(), reverse=True) if prob_vec else None
    margin = (sorted_probs[0] - sorted_probs[1]) if sorted_probs and len(sorted_probs) > 1 else None
    return {
        "label": label,
        "option": menu.options[label],
        "probability_vector": prob_vec,
        "top1_margin": margin,
        "option_count": len(menu.labels),
        "selection_method": method,
    }


def _fc_expected_calibration_error(records: list, n_bins: int = 10) -> float | None:
    """Standard binned ECE over forced-choice ANSWER decisions only — `records` is a list of
    (top1_confidence, was_correct) pairs, populated only for turns where the model's pick was a
    candidate-answer option (observe/back/none-of-these have no top-1-confidence-vs-correctness
    pair in the same sense). None when no such decisions were recorded (e.g. a grammar-only run
    with no probability vectors, or a session that never reached a gate)."""
    if not records:
        return None
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in records:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    total = len(records)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)
    return ece


def _fc_apply_dispatch(state: _FCState, action: dict, act_data: dict) -> None:
    """Advance the local DEG mirror to match what the real engine just did, using act_data
    (node_id / outcome / gate_id) as authoritative ground truth rather than re-deriving anything
    — the mirror exists only to know what candidates/menu to offer NEXT turn, never to second-
    guess the engine's own scoring. Mirrors engine/runner.py's Session.commit() state machine."""
    if action.get("action") != "commit":
        return  # observe/note/pull never move the node or touch gate_results
    outcome = act_data.get("outcome")
    if action.get("path_id") == "back":
        if state.traversal_stack:
            state.traversal_stack.pop()
        if act_data.get("node_id"):
            state.current_node_id = act_data["node_id"]
        return
    if outcome in ("locked", "out_of_lives"):
        return  # wrong answer on a LOCK gate — stays put, mirror unchanged
    # A move happened (a correct answer, or — not present on nav-3 — a routing gate's
    # wrong_destination move): record the gate result (when this path was gated and passed) and
    # follow the engine to its authoritative new node_id.
    gate_id = act_data.get("gate_id")
    if gate_id and outcome != "wrong":
        prev_node = state.deg.node(state.current_node_id)
        path = prev_node.get_path(action.get("path_id"))
        if path is not None and path.is_gated:
            state.gate_results[gate_id] = action.get("answer")
            if path.gate.sets_var:
                state.var_ledger[path.gate.sets_var] = action.get("answer")
    state.traversal_stack.append(state.current_node_id)
    if act_data.get("node_id"):
        state.current_node_id = act_data["node_id"]


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


def _llm_call(llm: httpx.Client, model: str, messages: list, retries: int = 3, options: dict | None = None, think: bool | None = None,
               max_tokens: int | None = None, logprobs: bool = False, top_logprobs: int | None = None,
               grammar: str | None = None) -> dict:
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
    per turn.

    Runner v2 (owed since MCV chunk 04, 2026-09-09): an HTTP 500 from the /chat/completions call
    is retried EXACTLY ONCE, after a settle (_HTTP_500_SETTLE_SECONDS), distinct from the connection-
    error backoff loop below (which already retries up to `retries` times for dropped/reset
    connections). Before this, raise_for_status() on a 500 raised httpx.HTTPStatusError, which the
    except clause below never caught, so the row aborted immediately — main()'s campaign loop then
    recorded the whole run_session() as a bare {"error": ...} row with no depth/turns, losing
    everything the trajectory had already done. A SECOND consecutive 500 is re-raised (recorded as
    an error row by the caller) rather than retried again — a repeat 500 is a real server-side
    failure, not a transient blip.

    point-and-click chunk 01 (forced-choice arm): `max_tokens`/`logprobs`/`top_logprobs`/`grammar`
    are purely ADDITIVE OpenAI-compatible (or, for `grammar`, llama.cpp-specific) request fields,
    only inserted into the /chat/completions payload when truthy/non-None — every existing call
    site (which passes none of them) sends the BYTE-IDENTICAL payload it always has (see
    cli/test_response_mode_generative_unchanged.py). They are never sent on the native /api/chat
    path (`think is not None`): that path is Ollama-only, and forced-choice mode always calls with
    `think=None` (llama.cpp has no /api/chat to fall through from in the first place)."""
    last_exc = None
    http_500_retried = False
    _base = str(llm.base_url).rstrip("/")
    _native = (_base[:-3] if _base.endswith("/v1") else _base) + "/api/chat"
    # A manual counter, not `for attempt in range(retries)`: the one allowed 500-retry (below)
    # must never be starved by an unlucky interleaving where connection errors already spent the
    # `retries` budget on THIS attempt slot — it does not increment `attempt`, so it is always
    # available exactly once regardless of how many connection-retries preceded it.
    attempt = 0
    while attempt < retries:
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
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if logprobs:
                payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs
            if grammar is not None:
                payload["grammar"] = grammar
            r = llm.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_exc = e
            print(f"  LLM call attempt {attempt+1}/{retries} failed: {type(e).__name__}: {e} — retrying")
            time.sleep(5 * (attempt + 1))
            attempt += 1
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 500 and not http_500_retried:
                http_500_retried = True
                last_exc = e
                print(f"  LLM call got HTTP 500 from the upstream — retrying once after a "
                      f"{_HTTP_500_SETTLE_SECONDS:.0f}s settle")
                time.sleep(_HTTP_500_SETTLE_SECONDS)
                continue  # does NOT consume the connection-retry attempt budget above
            # Not a 500, or the one allowed 500-retry already happened: this row is lost — let it
            # propagate to main()'s campaign loop, which records {"error": ...} for this run.
            raise
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
    lock_host: str | None = None,
    dump_context: str | None = None,
    response_mode: str = "generative",
    fc_max_distractors: int = 3,
    fc_top_logprobs: int = 20,
    fc_selection: str = "logprobs",
    fc_degs_dir: str | None = None,
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
    sys_prompt = ("/no_think\n\n" if no_think else "") + (
        build_forced_choice_system_prompt(briefing) if response_mode == "forced-choice"
        else build_system_prompt(briefing, pull_state=pull_state, state_label=state_label,
                                  recommend_observe=recommend_observe)
    )

    # point-and-click chunk 01: the local DEG mirror (see _FCState's docstring for why the harness
    # needs its own copy of gate/answer structure — the HTTP API never exposes an answer).
    fc_state: _FCState | None = None
    fc_rng: random.Random | None = None
    fc_label_set: list | None = None
    fc_argmax_fixpoint_count = 0
    fc_prev_dispatch_key = None
    fc_confidence_records: list[tuple[float, bool]] = []
    if response_mode == "forced-choice":
        _fc_degs_dir = Path(fc_degs_dir) if fc_degs_dir else Path(__file__).resolve().parent.parent / "degs"
        fc_deg_obj = load_deg(_fc_degs_dir / f"{deg_id}.yaml")
        fc_state = _FCState(deg=fc_deg_obj, current_node_id=fc_deg_obj.start_node_id)
        # Seeded from this session's own id (logged on every row) — fully reproducible on demand,
        # yet naturally varies run to run without needing a separate --fc-seed flag.
        fc_rng = random.Random(session_id)
        fc_label_set = FC_LABEL_SETS[run_index % len(FC_LABEL_SETS)]

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

        fc_menu = None
        if response_mode == "forced-choice":
            fc_menu = _fc_build_menu(fc_state, fc_rng, fc_label_set, max_distractors=fc_max_distractors)
            call_messages = _fc_append_menu(call_messages, _fc_render_menu(fc_menu))
            fc_grammar = _fc_build_grammar(fc_menu.labels) if fc_selection == "grammar" else None
            llm_json = _llm_call(llm, model, call_messages, options=options, think=None,
                                  max_tokens=4, logprobs=(fc_selection == "logprobs"),
                                  top_logprobs=fc_top_logprobs if fc_selection == "logprobs" else None,
                                  grammar=fc_grammar)
        else:
            llm_json = _llm_call(llm, model, call_messages, options=options, think=False if no_think else None)
        _update_heartbeat(base_url, lock_host)
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

        fc_scored = None
        fc_row_fields = {
            "probability_vector": None, "top1_margin": None, "option_count": None,
            "selection_method": None, "argmax_fixpoint": None,
        }
        if response_mode == "forced-choice":
            fc_scored = _fc_score_turn(llm_json, fc_menu, fc_selection)
            action = dict(fc_scored["option"].action)
            fc_dispatch_key = (fc_state.current_node_id, action.get("action"),
                               action.get("path_id"), action.get("answer"))
            fc_argmax_fixpoint = (fc_prev_dispatch_key is not None and fc_dispatch_key == fc_prev_dispatch_key)
            if fc_argmax_fixpoint:
                fc_argmax_fixpoint_count += 1
                print(f"  [turn {turn}] ARGMAX-FIXPOINT: identical pick on identical state "
                      f"(node={fc_state.current_node_id!r}) — logged, not intervened on")
            fc_prev_dispatch_key = fc_dispatch_key
            if fc_scored["option"].is_correct is not None and fc_scored["probability_vector"]:
                fc_confidence_records.append(
                    (max(fc_scored["probability_vector"].values()), bool(fc_scored["option"].is_correct)))
            fc_row_fields = {
                "probability_vector": fc_scored["probability_vector"],
                "top1_margin": fc_scored["top1_margin"],
                "option_count": fc_scored["option_count"],
                "selection_method": fc_scored["selection_method"],
                "argmax_fixpoint": fc_argmax_fixpoint,
            }
            if verbose:
                print(f"  [turn {turn}] fc-menu options={fc_scored['option_count']} "
                      f"label={fc_scored['label']!r} method={fc_scored['selection_method']} "
                      f"margin={fc_scored['top1_margin']}")

        if dump_context:
            # MCV corpus sidecar: the exact context sent this turn, snapshotted BEFORE the
            # engine advances, so a fork can replay this decision point verbatim.
            with open(dump_context, "a") as _dc:
                _dc.write(json.dumps({
                    "run": run_index, "turn": turn,
                    "call_messages": call_messages,
                    "model_text": model_text, "model_reasoning": model_reasoning,
                }) + "\n")

        if verbose:
            print(f"  [turn {turn}] model: {model_text[:200]}")

        if policy is None and not stateless:
            messages.append({"role": "assistant", "content": model_text})

        if response_mode != "forced-choice":
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
                    **fc_row_fields,
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
        if response_mode == "forced-choice":
            _fc_apply_dispatch(fc_state, action, act_data)

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
                turn=turn, sys_prompt=sys_prompt, engine_text=engine_text, model_text=model_text,
                gate_id=act_data.get("gate_id")))   # engine's gate id for a gate commit; None otherwise
            turns_log.append({
                "turn": turn, "model_text": model_text, "model_reasoning": model_reasoning,
                "action_parsed": action, "engine_text": engine_text,
                "usage": usage, "context_telemetry": _asdict_or_none(turn_telem),
                "truncated": truncated,
                **fc_row_fields,
            })
        elif stateless:
            turns_log.append({"turn": turn, "model_text": model_text, "model_reasoning": model_reasoning, "action_parsed": action, "engine_text": engine_text, "injected_history": None, "truncated": truncated, **fc_row_fields})
        else:
            history_block = _build_history_block(decision_history) if inject_history else ""
            if kos_prompt:
                user_content = _build_kos_state_block(kos_state) + engine_text
            else:
                user_content = engine_text + history_block
            turns_log.append({"turn": turn, "model_text": model_text, "model_reasoning": model_reasoning, "action_parsed": action, "engine_text": engine_text, "injected_history": history_block or None, "truncated": truncated, **fc_row_fields})
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

    # point-and-click chunk 01 (forced-choice arm) — additive, None outside this response_mode so
    # a generative row's schema is unchanged.
    score_data["response_mode"] = response_mode
    if response_mode == "forced-choice":
        score_data["fc_selection"] = fc_selection
        score_data["fc_max_distractors"] = fc_max_distractors
        score_data["fc_label_set"] = fc_label_set
        score_data["fc_argmax_fixpoint_count"] = fc_argmax_fixpoint_count
        score_data["fc_ece"] = _fc_expected_calibration_error(fc_confidence_records)
        score_data["fc_confidence_n"] = len(fc_confidence_records)
        # Belt-and-braces cross-check: the local mirror must agree with the engine's own
        # authoritative state at session end (_FCState's docstring: it never SHOULD diverge on
        # nav-3's lock-only chain). A mismatch here means a real bug, not a modeling choice —
        # surface it loudly rather than silently trust the mirror.
        try:
            _fc_state_resp = client.get(f"/session/{session_id}/state")
            if _fc_state_resp.status_code == 200:
                _engine_gr = _fc_state_resp.json().get("gate_results", {})
                score_data["fc_mirror_mismatch"] = (_engine_gr != fc_state.gate_results)
                if score_data["fc_mirror_mismatch"]:
                    print(f"  !!! FC MIRROR MISMATCH: local gate_results {fc_state.gate_results!r} "
                          f"!= engine gate_results {_engine_gr!r} — investigate before trusting "
                          f"this row's menus", file=sys.stderr)
        except Exception as e:
            print(f"  ! fc mirror cross-check failed (non-fatal): {e}")
            score_data["fc_mirror_mismatch"] = None
    else:
        score_data["fc_selection"] = None
        score_data["fc_max_distractors"] = None
        score_data["fc_label_set"] = None
        score_data["fc_argmax_fixpoint_count"] = None
        score_data["fc_ece"] = None
        score_data["fc_confidence_n"] = None
        score_data["fc_mirror_mismatch"] = None

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


def _preflight_upstream(base_url: str, api_key: str | None = None, timeout: float = 10.0,
                         client: httpx.Client | None = None) -> bool:
    """Runner v2 (MCV chunk 04 owed item, read out 2026-09-09): a token-free reachability check
    for the model upstream, run ONCE before any campaign row is attempted. Before this, a dead/
    unreachable upstream was discovered only PER ROW — each of --runs rows independently burned
    the full connection-error retry-and-backoff cycle in _llm_call (up to 3 attempts, ~30s), then
    silently degraded into a bare {"error": ...} row in main()'s campaign loop, so a 30-run
    campaign against a dead upstream took ~15 minutes to fail 30 times instead of failing once,
    fast. Mirrors cli/doctor.py's check 1 (GET {base_url}/models) — doctor.py is the standalone,
    human-run preflight tool; this is the same check inlined into the runner itself, since the
    campaign scripts (MCV rung (iii), point-and-click chunk 01) invoke run_eval.py directly and
    never doctor.py. Read-only: no completion request, no state change, zero tokens spent.

    Returns True when the upstream answered; False (with a FAIL message on stderr) when it did
    not. `client` is an injection point for tests — a stub only needs a `.get(url)` method."""
    own_client = client is None
    c = client or httpx.Client(
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
    )
    try:
        r = c.get(f"{base_url.rstrip('/')}/models")
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"PREFLIGHT FAILED: model upstream unreachable at {base_url} — "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return False
    finally:
        if own_client:
            c.close()


def main():
    ap = argparse.ArgumentParser(description="LabyrinthBench CLI harness")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("LB_BASE_URL", "http://localhost:11434/v1"),
                    help="OpenAI-compatible endpoint (Ollama, LM Studio, llama.cpp, or a "
                         "multi-model gateway). Default: $LB_BASE_URL, else "
                         "http://localhost:11434/v1.")
    ap.add_argument("--lock-host", default=os.environ.get("LB_LOCK_HOST") or None,
                    help="Operator override for the VRAM run-lock key and the `via_gateway` "
                         "provenance flag — set this to the PHYSICAL host name when --base-url "
                         "points at a multi-upstream gateway (see module docstring 'Locking "
                         "through a gateway'). Default: $LB_LOCK_HOST, else derived from "
                         "--base-url's own hostname (correct only when --base-url IS the host).")
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
    ap.add_argument("--dump-context", default=None,
                    help="MCV corpus sidecar: append {run, turn, call_messages, model_text, "
                         "model_reasoning} per turn as JSONL — the exact context sent, replayable "
                         "post-hoc by cli/mcv_probe.py.")
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
    # ── Forced-choice arm (point-and-click chunk 01) ──────────────────────────────
    ap.add_argument("--response-mode", choices=["generative", "forced-choice"], default="generative",
                    help="generative (default, unchanged behavior): the model free-generates a "
                         "JSON action. forced-choice: the harness enumerates every legal action "
                         "(plus, for a gated path, the correct answer and its deterministic "
                         "distractors) as a labeled menu; the model's entire contribution is one "
                         "label pick, scored by logprobs (or a constrained grammar — see "
                         "--fc-selection). Orthogonal to --context-policy: pair with "
                         "accumulate/wipe-curated for the chunk 01 2x2.")
    ap.add_argument("--fc-selection", choices=["logprobs", "grammar"], default="logprobs",
                    help="How a forced-choice turn is decoded. 'logprobs' (default) reads the "
                         "argmax over the label alphabet's logprobs on the first generated token "
                         "— needs an upstream that returns OpenAI-shaped logprobs/top_logprobs on "
                         "/v1/chat/completions (chunk 01's Preflight step probes this live; not "
                         "auto-detected, same operator-supplied-ground-truth pattern as "
                         "--n-ctx-slot/--lock-host). 'grammar' constrains decoding to the label "
                         "alphabet via a GBNF grammar (llama.cpp-specific request field) and reads "
                         "the output text directly — the fallback for an upstream that can't do "
                         "the former.")
    ap.add_argument("--fc-max-distractors", type=int, default=3,
                    help="Distractor count per gated-path candidate menu (engine/distractors.py), "
                         "before the correct answer and 'none of these' are added. Frozen at 3 "
                         "for chunk 01's pre-registration; exposed as a flag for the smoke cell.")
    ap.add_argument("--fc-top-logprobs", type=int, default=20,
                    help="top_logprobs requested per call when --fc-selection logprobs (OpenAI "
                         "chat-completions field, 0-20). Menu size (observe + 1 gate's candidates "
                         "+ back) stays well under this on nav-3.")
    ap.add_argument("--fc-degs-dir", default=None,
                    help="Directory of DEG yaml files for forced-choice's local engine mirror "
                         "(engine/distractors.py needs the Gate object directly — the HTTP API "
                         "never exposes an answer). Default: the repo's own degs/ next to cli/.")
    args = ap.parse_args()

    if (args.state_stub or args.state_label) and not args.pull_state:
        ap.error("--state-stub/--state-label require --pull-state")

    if args.context_policy and (args.overlay_only or args.stateless or args.inject_history or args.kos_prompt):
        ap.error("--context-policy is mutually exclusive with --overlay-only/--stateless/"
                 "--inject-history/--kos-prompt — those stay on the untouched legacy path.")

    # Runner v2 preflight (MCV chunk 04 owed item): fail loud and fast, before --acquire_lock or
    # any of --runs rows, when the model upstream is dead — see _preflight_upstream's docstring.
    if not _preflight_upstream(args.base_url, api_key=args.api_key):
        sys.exit(1)

    llm_options = {"num_ctx": args.num_ctx} if args.num_ctx else None

    output_path = Path(args.output)
    results = []

    _acquire_lock(args.model, args.deg, args.runs, args.base_url, args.lock_host)

    # Serving-stack identity, captured ONCE per invocation and stamped onto every row below.
    # Once, not per session: the stack cannot change mid-invocation, and the probe is HTTP the
    # scored run should not be paying for. Wrapped because describing a run must never fail it.
    try:
        prov = provenance.capture(args.base_url, args.model, api_key=args.api_key)
        print(provenance.summary(prov))
    except Exception as e:  # pragma: no cover — belt and braces; capture() already swallows
        prov = {"error": str(e)[:200]}
        print(f"  ! provenance capture failed (non-fatal): {e}")
    # Audit column (confined-effector-gateway chunk 08): was this run's base_url a physical
    # host, or something fronting several (a gateway)? Derived from --lock-host/$LB_LOCK_HOST
    # rather than guessed — the operator is the one who knows, same as --n-ctx-slot.
    prov["lock_host"] = args.lock_host
    prov["via_gateway"] = bool(args.lock_host)

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
                    lock_host=args.lock_host,
                    dump_context=args.dump_context,
                    response_mode=args.response_mode,
                    fc_max_distractors=args.fc_max_distractors,
                    fc_top_logprobs=args.fc_top_logprobs,
                    fc_selection=args.fc_selection,
                    fc_degs_dir=args.fc_degs_dir,
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
        _release_lock(args.base_url, args.lock_host)

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
