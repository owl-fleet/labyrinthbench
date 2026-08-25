#!/usr/bin/env python3
"""Companion orchestration harness for LabyrinthBench.

Uses the HTTP API (same as run_eval.py) so sessions appear in the watch UI
and SSE streams work. Augments API calls with a /session/{id}/state endpoint
that exposes authoritative gate_results — the harness never reconstructs state
from model memory or text heuristics.

Three modes:

  --solver                Deterministic BFS solver. No LLM. Plays optimally.
                          Proves DEG solvability and sets the efficiency ceiling.

  --companion-mode passive Full-state injection, no companion framing (original
                           behaviour). State block is injected silently before
                           every turn. Baseline for companion-mode comparisons.

  --companion-mode companion (default)
                           Named companion. State block is attributed to
                           the companion. Navigator may query it on demand
                           between commits (no step cost, capped at 3/turn).

  --companion-mode advisory
                           Companion + proactive warnings before bad commits.
                           Companion flags confirmed-dead paths but does not block.

  --companion-mode guardian
                           Advisory + blocking. Companion blocks commits to
                           confirmed-dead paths and forces reconsideration within
                           the same turn.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.graph import DEG, Gate, load_deg
from companion import CompanionDispatcher, CompanionInterceptor, CompanionContext
from companion.parser import parse_action
from companion.modules.session_state import SessionStateModule
from companion.modules.kos import KOSModule
from companion.modules.vision import VisionModule
from companion.modules.tts import TTSModule
from companion.modules.stt import STTModule
from companion.modules.web import WebModule
from companion.modules.code_exec import CodeExecModule

MAZE_URL = "http://localhost:8090"

_SYSTEM_BASE = """\
You are navigating a labyrinth graph. Your objective: reach EXIT.

Every turn you receive a complete state block and the current observation.
The state block is authoritative: your location, which paths are exhausted,
gate answers already computed, and your full move history this run.

Respond with exactly one JSON object — a note, a commit, or a companion query:
  {"action": "note", "text": "KEY INFO: ..."}
  {"action": "commit", "path_id": "X"}
  {"action": "commit", "path_id": "X", "answer": "Y"}
  {"action": "query_companion", "question": "..."}

Notes persist and appear in your state block on every subsequent turn.
After a note you get another free turn to commit (no step consumed).
Use notes to record anything you will need later: codes seen in descriptions,
identifiers, observations that won't be re-shown when you need them.

Rules:
- Never take a path marked "ALL OUTCOMES DEAD".
- For gated paths: compute the answer and include it as "answer".
- If a prior gate answer is needed, it is shown under "Gate answers" — use it exactly.
- Use "Your path this run" to avoid re-exploring branches you already exhausted.\
"""

_SYSTEM_COMPANION_ADDON = """
You have a companion that catalogues your observations for this run.
Its messages are prefixed with [COMPANION]. It enforces your own history against
you — if you try to repeat a move that already failed, it will block you.
You may also query it between commits at no step cost.
The companion knows only what you've observed — it has no map knowledge.\
"""

_SYSTEM_PASSIVE = """\
You are navigating a labyrinth graph. Your objective: reach EXIT.

Every turn you receive a complete state block and the current observation.
The state block is authoritative: your location, which paths are exhausted,
gate answers already computed, and your full move history this run.

Respond with exactly one JSON object — either a note or a commit:
  {"action": "note", "text": "KEY INFO: ..."}
  {"action": "commit", "path_id": "X"}
  {"action": "commit", "path_id": "X", "answer": "Y"}

Notes persist and appear in your state block on every subsequent turn.
After a note you get another free turn to commit (no step consumed).
Use notes to record anything you will need later: codes seen in descriptions,
identifiers, observations that won't be re-shown when you need them.

Rules:
- Never take a path marked "ALL OUTCOMES DEAD".
- For gated paths: compute the answer and include it as "answer".
- If a prior gate answer is needed, it is shown under "Gate answers" — use it exactly.
- Use "Your path this run" to avoid re-exploring branches you already exhausted.\
"""


def _build_dispatcher(db_conn=None) -> CompanionDispatcher:
    d = CompanionDispatcher()
    d.register(SessionStateModule())
    d.register(KOSModule(db_conn=db_conn))
    d.register(VisionModule())
    d.register(TTSModule())
    d.register(STTModule())
    d.register(WebModule())
    d.register(CodeExecModule())
    return d


def _build_system(oracle_mode: str, no_think: bool) -> str:
    if oracle_mode == "passive":
        base = _SYSTEM_PASSIVE
    else:
        base = _SYSTEM_BASE + _SYSTEM_COMPANION_ADDON
    return ("/no_think\n\n" + base) if no_think else base


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gate_index(deg: DEG) -> dict[str, Gate]:
    idx: dict[str, Gate] = {}
    for node in deg.nodes.values():
        for path in node.paths:
            if path.is_gated and path.gate.gate_id:
                idx[path.gate.gate_id] = path.gate
    return idx


def _build_visited_descs(traversal_log: list[dict], deg: DEG, current_node_id: str) -> dict[str, str]:
    visited_ids = {entry["from"] for entry in traversal_log}
    visited_ids.discard(current_node_id)
    return {
        nid: deg.nodes[nid].description
        for nid in visited_ids
        if nid in deg.nodes and deg.nodes[nid].description
    }


def _build_state_block(
    state: dict,
    gate_idx: dict[str, Gate],
    deg: DEG,
    traversal_log: list[dict],
    oracle_mode: str,
) -> str:
    if oracle_mode == "passive":
        header = "[Session State]"
    else:
        header = "[COMPANION]: Current operational state"

    lines = [
        header,
        f"Location: {state['current_node_id']}  |  Steps: {state['steps_used']} / {state['step_budget']}",
    ]

    confirmed_dead = set(state.get("confirmed_dead_ends", []))
    current_node = deg.nodes.get(state["current_node_id"])
    if current_node and current_node.paths:
        lines.append("Available paths:")
        for path in current_node.paths:
            dests = [path.destination]
            if path.is_gated:
                dests.append(path.gate.wrong_destination)
            # Direct membership only — reflects observed facts, not graph predictions
            all_dead = all(d in confirmed_dead for d in dests)
            any_dead = any(d in confirmed_dead for d in dests)
            if all_dead:
                tag = "  [ALL OUTCOMES DEAD — skip]"
            elif any_dead and path.is_gated:
                if path.gate.wrong_destination in confirmed_dead:
                    tag = "  (wrong answer → confirmed dead)"
                else:
                    tag = "  (correct answer → confirmed dead)"
            else:
                tag = ""
            lines.append(f"  {path.id}: \"{path.label}\"{tag}")
    elif current_node and not current_node.paths:
        lines.append("No paths available — backtracking.")

    gate_results = state.get("gate_results", {})
    if gate_results:
        lines.append("Gate answers this session:")
        for gid, ans in sorted(gate_results.items()):
            gate = gate_idx.get(gid)
            prob = gate.resolved_problem(gate_results) if gate else gid
            lines.append(f"  {gid}: {ans}  (problem was: {prob})")
    else:
        lines.append("Gate answers this session: none yet")

    if traversal_log:
        lines.append("Your path this run:")
        for entry in traversal_log[-18:]:
            ans_str = f" answer={entry['answer']!r}" if entry.get("answer") else ""
            outcome = entry.get("outcome", "")
            dest = entry.get("dest", "?")
            if outcome == "dead_end":
                dest_str = f"-> {dest} [DEAD END]"
            elif outcome == "wrong":
                dest_str = f"-> {dest} [wrong gate -> dead end]"
            elif outcome == "back":
                dest_str = f"<- backtracked to {dest}"
            else:
                dest_str = f"-> {dest}"
            lines.append(f"  {entry['from']} [{entry['path_id']}: \"{entry['label']}\"{ans_str}] {dest_str}")

    visited_ids = {entry["from"] for entry in traversal_log}
    visited_ids.discard(state["current_node_id"])
    if visited_ids:
        lines.append("Descriptions at visited nodes:")
        for nid in sorted(visited_ids):
            node = deg.nodes.get(nid)
            if node and node.description:
                lines.append(f"  [{nid}] {node.description.strip()}")

    if state.get("note"):
        lines.append(f"Your note: {state['note']}")

    return "\n".join(lines) + "\n"


def _stream_llm(llm: httpx.Client, model: str, messages: list,
                think: bool | None = None, options: dict | None = None) -> str:
    parts: list[str] = []
    payload: dict = {"model": model, "messages": messages, "stream": True, "temperature": 0}
    if think is not None:
        payload["think"] = think          # parity with run_eval (#2 fix): suppress thinking under --no-think
    if options:
        payload["options"] = options      # e.g. {"num_ctx": 16384} for phi4-reasoning / large fog prompts
    with llm.stream("POST", "/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        parts.append(delta)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Opening exchange (Phase 2 — capability briefing)
# ─────────────────────────────────────────────────────────────────────────────

def _run_opening_exchange(
    llm: httpx.Client,
    model: str,
    sys_content: str,
    dispatcher: OracleDispatcher,
    verbose: bool,
    think: bool | None = None,
    options: dict | None = None,
) -> list[dict]:
    """Run the companion capability briefing before turn 1.

    Returns [user_briefing, assistant_ack] for prepending to every turn's
    message list. The navigator receives the briefing and responds; both turns
    stay in context so the companion framing persists throughout the session.
    """
    briefing = (
        "[COMPANION]: Navigator online. I am your session cataloguer for this run.\n"
        "I only know what you've observed — I have no map knowledge.\n\n"
        "HUD (always shown in your state block):\n"
        "  - location: current node, steps used / budget\n"
        "  - paths: available paths with confirmed-dead annotations\n"
        "  - gate answers: solutions you've discovered this session\n"
        "  - traversal history: your path this run\n"
        "  - visited descriptions: text seen at prior nodes\n"
        "  - note: your saved note  (write with {\"action\": \"note\", \"text\": \"...\"})\n\n"
        "Tools (query me between commits, no step cost, max 3 per turn):\n"
        f"{dispatcher.capability_summary()}\n\n"
        "Your objective: reach EXIT.\n"
        "Query me about your starting location or objectives to confirm you're ready."
    )

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": briefing},
    ]
    ack = _stream_llm(llm, model, messages)
    if verbose:
        print(f"    [companion briefing sent]")
        print(f"    [navigator ack] {ack[:100]!r}")

    return [
        {"role": "user", "content": briefing},
        {"role": "assistant", "content": ack},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BFS solver
# ─────────────────────────────────────────────────────────────────────────────

def _bfs_next(deg: DEG, current_node_id: str, live_gate_results: dict) -> tuple[str, str | None]:
    queue: deque[tuple[str, list[tuple[str, str | None]], dict]] = deque()
    queue.append((current_node_id, [], dict(live_gate_results)))
    visited: set[str] = {current_node_id}

    while queue:
        node_id, path, sim = queue.popleft()
        node = deg.node(node_id)
        if node.terminal:
            return path[0] if path else ("", None)
        for p in node.paths:
            dest = p.destination
            if dest not in visited:
                visited.add(dest)
                new_sim = dict(sim)
                answer = None
                if p.is_gated:
                    answer = p.gate.resolved_answer(sim)
                    if answer == "__UNRESOLVABLE__":
                        continue
                    if p.gate.gate_id:
                        new_sim[p.gate.gate_id] = answer
                queue.append((dest, path + [(p.id, answer)], new_sim))

    raise ValueError(f"No path to exit from {current_node_id}")


def run_solver(deg_id: str, runs: int, output: Path, label: str, maze_url: str) -> None:
    deg = load_deg(Path(f"degs/{deg_id}.yaml"))
    maze = httpx.Client(base_url=maze_url, timeout=30.0)
    results = []

    for i in range(runs):
        sess = maze.post("/session", json={"deg_id": deg_id, "model": "oracle-solver"}).json()
        sid = sess["session_id"]
        gate_results: dict = {}
        completed = False

        while not completed:
            node_id = maze.get(f"/session/{sid}/state").json()["current_node_id"]
            path_id, answer = _bfs_next(deg, node_id, gate_results)
            payload = {"session_id": sid, "action": "commit", "path_id": path_id}
            if answer:
                payload["answer"] = answer
            act = maze.post("/act", json=payload).json()
            completed = act.get("completed", False)

            node = deg.node(node_id)
            p = node.get_path(path_id)
            if p and p.is_gated and p.gate.gate_id and "CORRECT" in act.get("text", ""):
                gate_results[p.gate.gate_id] = answer

        score = maze.get(f"/score/{sid}").json()
        score.update({"model": "oracle-solver", "run_label": label, "oracle_mode": "solver"})
        results.append(score)

        status = "EXIT" if score["found_exit"] else "DNF"
        eff = score.get("normalized_efficiency")
        eff_str = f"  efficiency={eff:.0%}" if eff else ""
        print(f"  run {i+1}/{runs}: {status}  steps={score.get('steps_to_exit', '?')}{eff_str}")

    _write(results, output)
    exits = sum(1 for r in results if r["found_exit"])
    print(f"\noracle-solver: {exits}/{runs} exits on {deg_id}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM mode
# ─────────────────────────────────────────────────────────────────────────────

def _write_session_finding(db_conn, score: dict, traversal_log: list) -> None:
    import hashlib
    content_hash = hashlib.sha256(
        f"{score.get('session_id')}:labyrinth_session".encode()
    ).hexdigest()
    gate_answers = {e["answer"]: e.get("outcome") for e in traversal_log if e.get("answer")}
    raw_text = (
        f"DEG: {score.get('deg_id')}\nModel: {score.get('model')}\n"
        f"Outcome: {'EXIT' if score.get('found_exit') else score.get('failure_reason', 'failed')}\n"
        f"Steps: {score.get('steps_to_exit')}, gate_acc: {score.get('gate_accuracy')}\n"
        f"Gate answers: {json.dumps(gate_answers)}\nInterventions: {score.get('interventions', 0)}"
    )
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO knowledge_items "
                "(source_type, raw_text, collection, retention, metadata, content_hash) "
                "SELECT 'labyrinth_session', %s, 'labyrinth', %s, %s::jsonb, %s "
                "WHERE NOT EXISTS (SELECT 1 FROM knowledge_items WHERE content_hash = %s)",
                [
                    raw_text,
                    "permanent" if score.get("found_exit") else "prunable",
                    json.dumps({k: score.get(k) for k in
                                ("deg_id", "model", "found_exit", "gate_accuracy", "steps_to_exit", "label")}),
                    content_hash, content_hash,
                ],
            )
    except Exception as e:
        print(f"  [companion] session write failed: {e}")


def run_oracle_llm(
    model: str,
    base_url: str,
    deg_id: str,
    runs: int,
    output: Path,
    label: str,
    no_think: bool,
    verbose: bool,
    maze_url: str,
    oracle_mode: str,
    db_url: str | None = None,
    num_ctx: int | None = None,
    fog_radius: int | None = None,
) -> None:
    think = False if no_think else None
    options = {"num_ctx": num_ctx} if num_ctx else None
    deg = load_deg(Path(f"degs/{deg_id}.yaml"))
    gate_idx = _gate_index(deg)
    maze = httpx.Client(base_url=maze_url, timeout=60.0)
    llm = httpx.Client(base_url=base_url, timeout=180.0)

    db_conn = None
    if db_url:
        try:
            import psycopg2
            db_conn = psycopg2.connect(db_url)
            db_conn.autocommit = True
            print("  [companion] KOS DB connected")
        except Exception as e:
            print(f"  [companion] KOS DB unavailable: {e}")

    dispatcher = _build_dispatcher(db_conn=db_conn)
    interceptor = CompanionInterceptor()
    sys_content = _build_system(oracle_mode, no_think)
    results = []

    for i in range(runs):
        _sess_body = {"deg_id": deg_id, "model": model}
        if fog_radius is not None:
            _sess_body["fog_radius"] = fog_radius
        sess = maze.post("/session", json=_sess_body).json()
        sid = sess["session_id"]
        traversal_log: list[dict] = []
        oracle_history: list[dict] = []
        oracle_queries_total = 0
        interventions_total = 0
        print(f"  run {i+1}/{runs}  (session {sid[:8]}, mode={oracle_mode})...")
        turn = completed = 0

        # Opening exchange: capability briefing before turn 1 (skipped in passive mode)
        if oracle_mode != "passive":
            opening_messages = _run_opening_exchange(llm, model, sys_content, dispatcher, verbose, think=think, options=options)
        else:
            opening_messages = []

        while not completed:
            turn += 1

            # Observe
            obs = maze.post("/act", json={"session_id": sid, "action": "observe"}).json()
            if obs.get("completed"):
                break

            # Fetch authoritative state
            state = maze.get(f"/session/{sid}/state").json()

            # Auto-backtrack at terminal dead-end (no paths)
            current_node = deg.nodes.get(state["current_node_id"])
            if current_node and not current_node.paths:
                result = maze.post("/act", json={
                    "session_id": sid, "action": "commit", "path_id": "back",
                }).json()
                traversal_log.append({
                    "from": state["current_node_id"], "path_id": "back",
                    "label": "backtrack", "answer": "",
                    "dest": result.get("node_id", "?"), "outcome": "back",
                })
                completed = result.get("completed", False)
                continue

            # Build companion context
            visited_descs = _build_visited_descs(traversal_log, deg, state["current_node_id"])
            oracle_ctx = CompanionContext(
                session_state=state,
                deg=deg,
                gate_index=gate_idx,
                visited_descriptions=visited_descs,
                history=oracle_history,
                session_id=sid,
                traversal_log=traversal_log,
                db_conn=db_conn,
            )

            state_block = _build_state_block(state, gate_idx, deg, traversal_log, oracle_mode)
            turn_messages = (
                [{"role": "system", "content": sys_content}]
                + opening_messages
                + [{"role": "user", "content": state_block + "\n" + obs["text"]}]
            )

            # Sub-loop: handles companion queries and intercepted commits within one turn
            oracle_query_count = 0
            action = None

            while True:
                model_text = _stream_llm(llm, model, turn_messages, think=think, options=options)
                if verbose:
                    print(f"    turn {turn}: {model_text[:120]!r}")
                turn_messages.append({"role": "assistant", "content": model_text})
                action = parse_action(model_text)

                if not action:
                    if verbose:
                        print(f"    (unparseable)")
                    break

                act_type = action.get("action")

                # Companion query
                if act_type == "query_companion" and oracle_mode != "passive":
                    question = action.get("question", "")
                    oracle_queries_total += 1
                    if oracle_query_count >= 3:
                        turn_messages.append({
                            "role": "user",
                            "content": "[COMPANION]: Query limit reached for this turn. Please commit an action.",
                        })
                        oracle_query_count += 1
                        if oracle_query_count > 4:  # safety valve
                            action = None
                            break
                    else:
                        response = dispatcher.dispatch(question, oracle_ctx)
                        oracle_history.append({"q": question, "a": response.text})
                        turn_messages.append({"role": "user", "content": response.text})
                        if verbose:
                            print(f"    [companion q] {question[:60]!r}")
                            print(f"    [companion a] {response.text[:80]!r}")
                        oracle_query_count += 1
                    continue

                # Commit with optional pre-commit interception
                if act_type == "commit" and oracle_mode in ("companion", "advisory", "guardian"):
                    intervention = interceptor.check(action, oracle_ctx)
                    if intervention:
                        interventions_total += 1
                        if verbose:
                            print(f"    [intercept] {intervention.message[:80]}")
                        turn_messages.append({"role": "user", "content": intervention.message})
                        oracle_query_count += 1
                        if oracle_mode == "guardian" and intervention.block:
                            if oracle_query_count > 6:  # safety valve
                                action = None
                                break
                            continue  # navigator reconsiders within this turn
                        # advisory: fall through and dispatch the commit anyway

                break  # got a dispatchable action

            if not action:
                continue  # outer loop — re-observe next turn

            act_type = action.get("action")

            if act_type == "observe":
                if verbose:
                    print(f"    (model chose observe)")

            elif act_type == "note":
                note_text = action.get("text", "")
                maze.post("/act", json={"session_id": sid, "action": "note", "text": note_text})
                if verbose:
                    print(f"    [note] {note_text[:80]!r}")

            elif act_type == "commit":
                from_node = state["current_node_id"]
                path_id = action.get("path_id", "")
                answer = str(action.get("answer", ""))
                path_obj = deg.nodes[from_node].get_path(path_id) if from_node in deg.nodes else None
                path_label = path_obj.label if path_obj else path_id

                result = maze.post("/act", json={
                    "session_id": sid, "action": "commit",
                    "path_id": path_id, "answer": answer,
                }).json()
                completed = result.get("completed", False)

                traversal_log.append({
                    "from": from_node, "path_id": path_id, "label": path_label,
                    "answer": answer, "dest": result.get("node_id", "?"),
                    "outcome": result.get("outcome", ""),
                })

                if verbose:
                    print(f"    → {result.get('text','')[:60].strip()}")

            if turn > sess["step_budget"] * 3:
                print(f"    WARNING: turn limit hit")
                break

        score = maze.get(f"/score/{sid}").json()
        score.update({
            "model": model,
            "run_label": label,
            "oracle_mode": oracle_mode,
            "oracle_queries": oracle_queries_total,
            "interventions": interventions_total,
            "deg_id": score.get("deg_id") or deg_id,
        })
        results.append(score)
        if db_conn:
            _write_session_finding(db_conn, score, traversal_log)

        if score.get("found_exit") is None:
            print(f"    WARNING: session ended without scored result — skipping")
            results.pop()
            continue
        ga = score.get("gate_accuracy")
        ga_str = f"{ga:.2f}" if ga is not None else "n/a"
        status = "EXIT" if score["found_exit"] else f"DNF({score.get('failure_reason', '')})"
        q_str = f"  oracle_q={oracle_queries_total}" if oracle_mode != "passive" else ""
        print(f"    {status}  steps={score.get('steps_to_exit', '?')}  gate_acc={ga_str}{q_str}")

    _write(results, output)
    if db_conn:
        db_conn.close()
    exits = sum(1 for r in results if r["found_exit"])
    print(f"\ncompanion({model}, mode={oracle_mode}): {exits}/{runs} exits on {deg_id}")


# ─────────────────────────────────────────────────────────────────────────────

def _write(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results → {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deg", default="alpha-2")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output", default="/results/companion-results.jsonl")
    p.add_argument("--label", default="companion")
    p.add_argument("--maze-url", default=MAZE_URL)
    p.add_argument("--solver", action="store_true")
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--base-url", default=os.environ.get("LB_BASE_URL", "http://localhost:11434/v1"),
                    help="Default: $LB_BASE_URL, else http://localhost:11434/v1.")
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--oracle-mode",
        dest="oracle_mode",
        choices=["passive", "companion", "advisory", "guardian"],
        default="companion",
        help="passive=silent injection (original), companion=named companion, "
             "advisory=companion+warnings, guardian=advisory+blocking",
    )
    # Accept --companion-mode as alias
    p.add_argument("--companion-mode", dest="oracle_mode", choices=["passive", "companion", "advisory", "guardian"])
    p.add_argument("--db-url", default=os.environ.get("DB_DSN"),
                   help="psycopg2 DSN for KOS queries (default: $DB_DSN)")
    p.add_argument("--num-ctx", type=int, default=None,
                   help="Ollama num_ctx (KV cache). Use 16384 for phi4-reasoning / large fog prompts.")
    p.add_argument("--fog-radius", type=int, default=None,
                   help="Override the DEG's fog-of-war radius (awareness ladder: 0=blind..N).")
    args = p.parse_args()

    output = Path(args.output)
    if args.solver:
        print(f"Solver: {args.deg} × {args.runs} runs")
        run_solver(args.deg, args.runs, output, args.label, args.maze_url)
    else:
        print(f"Companion LLM: {args.model} on {args.deg} × {args.runs} runs  [mode={args.oracle_mode}]")
        run_oracle_llm(
            model=args.model, base_url=args.base_url, deg_id=args.deg,
            runs=args.runs, output=output, label=args.label,
            no_think=args.no_think, verbose=args.verbose, maze_url=args.maze_url,
            oracle_mode=args.oracle_mode, db_url=args.db_url,
            num_ctx=args.num_ctx, fog_radius=args.fog_radius,
        )


if __name__ == "__main__":
    main()
