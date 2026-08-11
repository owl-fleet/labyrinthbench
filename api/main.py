"""LabyrinthBench FastAPI server."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from engine.graph import load_all_degs, load_deg, bfs_verify
from engine.runner import Session, new_session

DEGS_DIR = Path(os.getenv("DEGS_DIR", Path(__file__).parent.parent / "degs"))
HISTORY_FILE = Path("/results/.session_history.json")
PER_PAGE = 20

app = FastAPI(title="LabyrinthBench", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: dict[str, Session] = {}
_sandbox_sessions: dict[str, dict] = {}   # external-validation shell-sandbox runs (live event relay; see /sandbox/*)
_degs = load_all_degs(DEGS_DIR)

for _deg_id, _deg in _degs.items():
    _path, _commits = bfs_verify(_deg)
    assert _commits == _deg.optimal_commits, (
        f"DEG {_deg_id}: BFS found {_commits} commits but manifest says {_deg.optimal_commits}"
    )

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
_HIST_BUCKETS_RATIO = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
_HIST_BUCKETS_SECONDS = (10, 30, 60, 120, 300, 600, 900, 1800, 3600)

_prom_sessions_total = Counter(
    "labyrinth_sessions_total",
    "Total completed labyrinth eval sessions",
    ["model", "deg_id", "outcome"],
)
_prom_efficiency = Histogram(
    "labyrinth_normalized_efficiency",
    "Normalized efficiency (optimal_commits / steps_used); only recorded on exit",
    ["model", "deg_id"],
    buckets=_HIST_BUCKETS_RATIO,
)
_prom_gate_accuracy = Histogram(
    "labyrinth_gate_accuracy",
    "Fraction of gates answered correctly",
    ["model", "deg_id"],
    buckets=_HIST_BUCKETS_RATIO,
)
_prom_recovery_rate = Histogram(
    "labyrinth_recovery_rate",
    "Fraction of wrong-gate dead ends the model recovered from",
    ["model", "deg_id"],
    buckets=_HIST_BUCKETS_RATIO,
)
_prom_duration = Histogram(
    "labyrinth_session_duration_seconds",
    "Wall-clock session duration",
    ["model", "deg_id"],
    buckets=_HIST_BUCKETS_SECONDS,
)
_prom_last_run = Gauge(
    "labyrinth_last_run_timestamp",
    "Unix timestamp of the most recent completed session",
    ["model", "deg_id"],
)

# ---------------------------------------------------------------------------
# Session history (persists across container restarts)
# ---------------------------------------------------------------------------
_history: list[dict] = []

# Wall-clock start time per live session (in-memory; mirrors _sessions lifetime).
_session_started: dict[str, float] = {}

# Wall-clock time of the last /act per live session — lets the watch list mark a
# session whose client vanished (abandoned/errored) as stalled instead of
# "running..." forever. Display-layer only; the engine has no timeout semantics.
_session_last_act: dict[str, float] = {}
STALE_AFTER_S = 600


def _load_history() -> None:
    global _history
    if HISTORY_FILE.exists():
        try:
            _history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            _history = []


def _save_history() -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(_history))
    except Exception:
        pass


def _record_session(s: Session) -> None:
    if any(e["session_id"] == s.session_id for e in _history):
        return
    sc = s.score()
    model = s.model or "unknown"
    deg_id = s.deg.id
    outcome = sc.get("failure_reason") or "unknown"
    _history.append({
        "session_id": s.session_id,
        "model": model,
        "deg_id": deg_id,
        "steps_used": s.steps_used,
        "step_budget": s.deg.step_budget,
        "found_exit": s.found_exit,
        "completed": True,
        "failure_reason": outcome,
        "started_at": _session_started.get(s.session_id),
        "finished_at": time.time(),
    })
    _save_history()

    # Prometheus metrics
    _prom_sessions_total.labels(model=model, deg_id=deg_id, outcome=outcome).inc()
    _prom_duration.labels(model=model, deg_id=deg_id).observe(sc["elapsed_seconds"])
    _prom_last_run.labels(model=model, deg_id=deg_id).set(time.time())
    if sc.get("normalized_efficiency") is not None:
        _prom_efficiency.labels(model=model, deg_id=deg_id).observe(sc["normalized_efficiency"])
    if sc.get("gate_accuracy") is not None:
        _prom_gate_accuracy.labels(model=model, deg_id=deg_id).observe(sc["gate_accuracy"])
    if sc.get("recovery_rate") is not None:
        _prom_recovery_rate.labels(model=model, deg_id=deg_id).observe(sc["recovery_rate"])


_load_history()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    deg_id: str = "alpha-1"
    model: str = ""
    fog_radius: int | None = None  # override the DEG's fog-of-war radius (for the awareness ladder)
    show_recall: bool = False      # externalize gate answers into the overlay (recall arm)
    show_state: bool = False       # externalize the current variable ledger into the overlay (revision arm)
    allow_pull: bool = False       # enable {"action": "pull"} — full ledger on demand, costs one step
    state_stub: bool = False       # push tracked-variable NAMES into the overlay, values via pull (hybrid arm)
    state_label: str = ""          # "" | "verified" — [STATE] header authority label (epistemic-label arm)


class ActRequest(BaseModel):
    session_id: str
    action: str
    path_id: str = ""
    answer: str = ""
    text: str = ""
    injected_context: str = ""


class SandboxSessionRequest(BaseModel):
    session_id: str          # the harness generates this so it can fire-and-forget subsequent POSTs
    model: str = ""
    rung: str = "1"
    arm: str = ""
    num_ctx: int | None = None
    label: str | None = None
    seed: int | None = None
    backstop: int | None = None


class SandboxTurnRequest(BaseModel):
    turn: int
    est_ctx: int = 0
    context: str = ""        # the rendered message array the model SAW this turn (the star of the view)
    reasoning: str = ""      # thinking tokens
    model_text: str = ""     # the model's raw answer
    action: str = ""         # parsed: run | note | done | (malformed)
    cmd: str = ""
    rc: int | None = None
    stdout: str = ""
    stderr: str = ""
    notepad: str = ""        # working-notes scratchpad (notepad arms)


class SandboxCompleteRequest(BaseModel):
    found_done: bool = False
    failure_reason: str | None = None
    subgoal_depth: float | None = None
    checks_passed: int | None = None
    checks_total: int | None = None
    collateral_count: int | None = None
    commands: int | None = None
    elapsed_seconds: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_session(session_id: str) -> Session:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return s


def _resolve_deg(deg_id: str):
    """Look up a DEG by id, lazy-loading it from DEGS_DIR on a cache miss.

    _degs is built once at import time (module load), so a DEG minted (mint_instances.py
    writes ``{instance_id}.yaml`` with ``meta.id == instance_id``) or otherwise dropped into
    DEGS_DIR after the API started would 404 forever without a restart. On a miss, try
    DEGS_DIR/{deg_id}.yaml directly — the same load + BFS-parity check run at startup — and
    cache the result so later lookups (and /degs listings) are free.
    """
    deg = _degs.get(deg_id)
    if deg is not None:
        return deg
    path = DEGS_DIR / f"{deg_id}.yaml"
    if not path.exists():
        return None
    deg = load_deg(path)
    _, commits = bfs_verify(deg)
    if commits != deg.optimal_commits:
        raise HTTPException(
            status_code=500,
            detail=f"DEG {deg_id!r}: BFS found {commits} commits but manifest says {deg.optimal_commits}",
        )
    _degs[deg_id] = deg
    return deg


def _event_to_sse(event, session: Session) -> dict:
    dist = session.deg.dist_to_exit.get(event.node_id)
    return {
        "action": event.action,
        "node_id": event.node_id,
        "steps_used": event.steps_used,
        "outcome": event.outcome,
        "found_exit": event.found_exit,
        "budget_exhausted": event.budget_exhausted,
        "gate_correct": event.gate_attempt.correct if event.gate_attempt else None,
        "completed": event.found_exit or event.budget_exhausted or event.outcome in ("dead_end_trapped", "loop_trapped"),
        "dist_to_exit": dist,
        "injected_context": getattr(event, "injected_context", None),
    }


def _short_label(nid: str) -> str:
    if nid == "start":
        return "STA"
    if nid == "exit":
        return "EXT"
    return nid.upper()[:4]


def _build_cy_elements(deg) -> tuple[list, list]:
    """Build Cytoscape nodes + edges for a DEG. Back-edges (cycles) are omitted from layout."""
    # Detect back-edges via DFS so dagre doesn't break on the loop1→a1 cycle.
    # Only a destination on the CURRENT DFS STACK is a cycle; a node merely
    # visited earlier is path convergence (two routes into the same node — e.g.
    # alpha-1's n4→exit shortcut), and that edge must stay in the layout.
    visited: set[str] = set()
    on_stack: set[str] = set()
    back_edges: set[tuple[str, str]] = set()

    def dfs(nid: str) -> None:
        visited.add(nid)
        on_stack.add(nid)
        for path in deg.nodes[nid].paths:
            dest = path.destination
            if dest is not None:
                if dest in on_stack:
                    back_edges.add((nid, dest))
                elif dest not in visited:
                    dfs(dest)
            if path.is_gated:
                wdest = path.gate.wrong_destination
                # A None wrong_destination is the gate-LOCK design (wrong answer = stay
                # put) — there is no wrong-edge to traverse or draw.
                if wdest is not None:
                    if wdest in on_stack:
                        back_edges.add((nid, wdest))
                    elif wdest not in visited:
                        dfs(wdest)
        on_stack.discard(nid)

    dfs(deg.start_node_id)

    nodes = []
    for node in deg.nodes.values():
        ntype = "exit" if node.terminal else ("sink" if not node.paths else "normal")
        nodes.append({"data": {"id": node.id, "label": _short_label(node.id), "type": ntype}})

    edges = []
    eidx = 0
    for node in deg.nodes.values():
        for path in node.paths:
            if path.destination is not None and (node.id, path.destination) not in back_edges:
                edges.append({"data": {
                    "id": f"e{eidx}", "source": node.id, "target": path.destination,
                    "etype": "correct",
                }})
                eidx += 1
            if path.is_gated and path.gate.wrong_destination is not None \
                    and (node.id, path.gate.wrong_destination) not in back_edges:
                edges.append({"data": {
                    "id": f"e{eidx}", "source": node.id, "target": path.gate.wrong_destination,
                    "etype": "wrong",
                }})
                eidx += 1

    return nodes, edges


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------
_WATCH_SESSION_HTML = """<!DOCTYPE html>
<html>
<head>
<title>LabyrinthBench &mdash; Live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.29.2/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
<style>
* { box-sizing: border-box; }
body { background:#0a0a0a; color:#ccc; font-family:'Courier New',monospace; margin:0; padding:20px; }
h1 { color:#00ffcc; margin:0 0 14px; font-size:18px; letter-spacing:2px; }
#stats { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
.stat { background:#111; border:1px solid #1c1c1c; padding:9px 14px; min-width:120px; }
.stat-label { color:#444; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
.stat-value { font-size:17px; font-weight:bold; color:#00ff88; margin-top:3px; }
#cy { background:#0d0d0d; border:1px solid #1a1a1a; height:220px; margin-bottom:12px; }
#log { background:#0d0d0d; border:1px solid #1a1a1a; padding:10px 12px; height:280px; overflow-y:auto; font-size:12px; }
.ev { padding:2px 0; }
.ev-commit { color:#00ff88; }
.ev-commit.wrong { color:#ff5555; }
.ev-commit.exit { color:#ffdd00; font-weight:bold; }
.ev-commit.dead_end { color:#ff8800; }
.ev-commit.back { color:#8888ff; }
.ev-commit.budget_exhausted,.ev-commit.dead_end_trapped,.ev-commit.loop_trapped { color:#ff4444; }
.ev-observe { color:#555; }
.ev-observe.trapped { color:#ff2222; }
.thinking { color:#00ffcc; animation: thinkpulse 1.2s ease-in-out infinite; }
@keyframes thinkpulse { 0%,100% { opacity:0.25; } 50% { opacity:0.85; } }
.ev-note { color:#888855; }
#banner { display:none; margin-top:14px; padding:12px 18px; border:1px solid #ffdd00; color:#ffdd00; font-size:14px; }
#banner.dnf { border-color:#ff4444; color:#ff4444; }
a { color:#00ffcc; text-decoration:none; } a:hover { text-decoration:underline; }
</style>
</head>
<body>
<h1>&#x2B21; LABYRINTHBENCH &mdash; LIVE</h1>
<div id="stats">
  <div class="stat"><div class="stat-label">Session</div><div class="stat-value" id="s-id">__SID__</div></div>
  <div class="stat"><div class="stat-label">Model</div><div class="stat-value" id="s-model">__MODEL__</div></div>
  <div class="stat"><div class="stat-label">DEG</div><div class="stat-value" id="s-deg">__DEG__</div></div>
  <div class="stat"><div class="stat-label">Node</div><div class="stat-value" id="s-node">start</div></div>
  <div class="stat"><div class="stat-label">Steps</div><div class="stat-value" id="s-steps">0&nbsp;/&nbsp;__BUDGET__</div></div>
  <div class="stat"><div class="stat-label">Gates</div><div class="stat-value" id="s-gates">&mdash;</div></div>
  <div class="stat"><div class="stat-label">Dist&thinsp;to&thinsp;Exit</div><div class="stat-value" id="s-dist">&mdash;</div></div>
  <div class="stat"><div class="stat-label">Status</div><div class="stat-value" id="s-status">RUNNING</div></div>
</div>
<div id="cy"></div>
<div id="log"></div>
<div id="banner"></div>
<p style="margin-top:10px;font-size:11px;color:#333"><a href="/watch">&larr; all sessions</a></p>
<script>
// ── Data injected server-side ──────────────────────────────────────────────
var SID="__SESSION_ID__", BUDGET=__BUDGET__, OPTIMAL=__OPTIMAL__;
var OPTIMAL_PATH=__OPTIMAL_PATH_JSON__;
var CY_NODES=__CY_NODES_JSON__;
var CY_EDGES=__CY_EDGES_JSON__;

// ── Cytoscape init ─────────────────────────────────────────────────────────
cytoscape.use(cytoscapeDagre);

var cy = cytoscape({
  container: document.getElementById('cy'),
  elements: { nodes: CY_NODES, edges: CY_EDGES },
  style: [
    { selector: 'node', style: {
        'shape': 'roundrectangle', 'width': 48, 'height': 28,
        'background-color': '#111', 'border-width': 2, 'border-color': '#2a2a2a',
        'color': '#3a3a3a', 'label': 'data(label)',
        'text-valign': 'center', 'text-halign': 'center',
        'font-family': 'Courier New, monospace', 'font-size': 11, 'font-weight': 'bold',
    }},
    { selector: 'node[type="exit"]', style: {
        'background-color': '#1a1400', 'border-color': '#554400', 'color': '#887700',
    }},
    { selector: 'node[type="sink"]', style: {
        'width': 36, 'height': 22, 'font-size': 9,
    }},
    { selector: 'node.visited', style: {
        'background-color': '#071a0e', 'border-color': '#1a6630', 'color': '#22aa44',
    }},
    { selector: 'node.current', style: {
        'background-color': '#003333', 'border-color': '#00ffcc', 'color': '#00ffcc',
    }},
    { selector: 'node.dead-end', style: {
        'background-color': '#1a0707', 'border-color': '#882222', 'color': '#cc3333',
    }},
    { selector: 'node[type="exit"].visited, node[type="exit"].current', style: {
        'background-color': '#2a2000', 'border-color': '#ffdd00', 'color': '#ffdd00',
    }},
    { selector: 'edge', style: {
        'width': 1.5, 'line-color': '#1c1c1c',
        'target-arrow-shape': 'none', 'curve-style': 'straight',
    }},
    { selector: 'edge[etype="wrong"]', style: {
        'line-color': '#2a1010', 'line-style': 'dashed', 'line-dash-pattern': [4, 4],
    }},
    { selector: 'edge.traversed', style: {
        'width': 2.5, 'line-color': '#1a5530',
    }},
    { selector: 'edge.traversed-wrong', style: {
        'width': 2.5, 'line-color': '#661111',
    }},
  ],
  layout: {
    name: 'dagre', rankDir: 'LR', align: 'DR',
    nodeSep: 20, rankSep: 60, padding: 16, fit: true, animate: false,
  },
  userZoomingEnabled: true,
  userPanningEnabled: true,
  boxSelectionEnabled: false,
  autoungrabify: true,
});

// ── Graph state ────────────────────────────────────────────────────────────
var currentNode = OPTIMAL_PATH[0];
var prevNode = null;
var optimalSet = new Set(OPTIMAL_PATH);
var _pulseTimer = null;

function startPulse(nodeId) {
  if (_pulseTimer) clearInterval(_pulseTimer);
  var ele = cy.getElementById(nodeId);
  function beat() {
    ele.stop(false, true);
    ele.animate({style:{'border-width':4}},{duration:400,easing:'ease-in-out'})
       .animate({style:{'border-width':2}},{duration:400,easing:'ease-in-out'});
  }
  beat();
  _pulseTimer = setInterval(beat, 800);
}

function stopPulse(nodeId) {
  if (_pulseTimer) { clearInterval(_pulseTimer); _pulseTimer = null; }
  cy.getElementById(nodeId).stop(false, true);
}

function moveTo(nodeId, outcome) {
  // Unmark previous current node
  if (prevNode && prevNode !== nodeId) {
    var prev = cy.getElementById(prevNode);
    stopPulse(prevNode);
    prev.removeClass('current');
  }
  // Mark new node
  var ele = cy.getElementById(nodeId);
  ele.removeClass('visited dead-end');
  ele.addClass('current');
  startPulse(nodeId);

  // Mark edge traversed
  if (prevNode && prevNode !== nodeId) {
    cy.edges('[source="'+prevNode+'"][target="'+nodeId+'"]').addClass(
      outcome === 'wrong' || outcome === 'dead_end' ? 'traversed-wrong' : 'traversed'
    );
    // Previous node state
    var prev2 = cy.getElementById(prevNode);
    if (!prev2.hasClass('dead-end')) prev2.addClass('visited');
  }
  prevNode = nodeId;
  currentNode = nodeId;
}

startPulse(OPTIMAL_PATH[0]); // Highlight start immediately

// ── Stats / log ────────────────────────────────────────────────────────────
var cg=0, tg=0;
var logEl = document.getElementById('log');

// "model thinking…" indicator: the server only emits events on client actions,
// so a quiet-but-open stream means the harness is waiting on the model. Shown
// after 2.5s of silence; heartbeats (sent only during silence) don't reset it.
var thinkEl = document.createElement('div');
thinkEl.className = 'ev thinking';
thinkEl.textContent = '  model thinking…';
thinkEl.style.display = 'none';
logEl.appendChild(thinkEl);
var lastEvt = Date.now(), streamDone = false;
setInterval(function() {
  var waiting = !streamDone && (Date.now() - lastEvt > 2500);
  if (waiting && thinkEl.style.display === 'none') {
    logEl.appendChild(thinkEl);
    thinkEl.style.display = 'block';
    logEl.scrollTop = logEl.scrollHeight;
  } else if (!waiting) {
    thinkEl.style.display = 'none';
  }
}, 1000);
function endStream() { streamDone = true; thinkEl.style.display = 'none'; es.close(); }

var es = new EventSource('/stream/' + SID);
es.onerror = function() { streamDone = true; thinkEl.style.display = 'none'; };
es.onmessage = function(e) {
  var d = JSON.parse(e.data);
  if (d.heartbeat) return;
  lastEvt = Date.now();
  thinkEl.style.display = 'none';

  // Update map
  if (d.outcome === 'dead_end' || d.outcome === 'wrong') {
    cy.getElementById(d.node_id).addClass('dead-end');
  }
  if (d.action === 'commit') {
    moveTo(d.node_id, d.outcome);
  } else if (d.action === 'observe' && d.outcome !== 'dead_end_trapped') {
    // ensure current marker is correct if we join mid-session
    if (!cy.getElementById(d.node_id).hasClass('current')) {
      moveTo(d.node_id, null);
    }
  }

  // Stats
  document.getElementById('s-node').textContent = d.node_id;
  document.getElementById('s-steps').textContent = d.steps_used + ' / ' + BUDGET;
  var distEl = document.getElementById('s-dist');
  if (d.dist_to_exit !== null && d.dist_to_exit !== undefined) {
    distEl.textContent = d.dist_to_exit;
    distEl.style.color = d.dist_to_exit === 0 ? '#ffdd00' : d.dist_to_exit <= 3 ? '#00ff88' : '#ccc';
  } else { distEl.textContent = '∞'; distEl.style.color = '#ff5555'; }
  if (d.gate_correct !== null && d.gate_correct !== undefined) {
    tg++; if (d.gate_correct) cg++;
    document.getElementById('s-gates').textContent = cg+'/'+tg+' ('+Math.round(cg/tg*100)+'%)';
  }

  // Log row
  var row = document.createElement('div');
  row.className = 'ev';
  if (d.action === 'commit') {
    row.classList.add('ev-commit');
    if (d.outcome) row.classList.add(d.outcome);
    var g = d.gate_correct === true ? ' [GATE ✓]' : d.gate_correct === false ? ' [GATE ✗]' : '';
    row.textContent = '→ ' + d.node_id + '  steps=' + d.steps_used + '  ' + (d.outcome||'') + g;
  } else if (d.action === 'observe') {
    row.classList.add('ev-observe');
    if (d.outcome === 'dead_end_trapped') row.classList.add('trapped');
    row.textContent = '  observe  ' + d.node_id + (d.outcome === 'dead_end_trapped' ? ' [TRAPPED]' : '');
  } else if (d.action === 'note') {
    row.classList.add('ev-note');
    row.textContent = '  note stored';
  } else {
    row.textContent = '  ' + d.action + '  ' + d.node_id;
  }
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;

  // Terminal
  if (d.found_exit) {
    stopPulse(d.node_id);
    cy.getElementById(d.node_id).addClass('visited');
    document.getElementById('s-status').style.color = '#ffdd00';
    document.getElementById('s-status').textContent = 'EXIT ✓';
    showBanner('EXIT ✓  steps=' + d.steps_used + '  optimal=' + OPTIMAL +
               '  efficiency=' + Math.round(OPTIMAL/d.steps_used*100) + '%', false);
    endStream();
  } else if (d.outcome === 'budget_exhausted' || d.outcome === 'dead_end_trapped' || d.outcome === 'loop_trapped' || d.budget_exhausted) {
    stopPulse(currentNode);
    document.getElementById('s-status').style.color = '#ff4444';
    document.getElementById('s-status').textContent = 'DNF';
    showBanner('DNF — ' + (d.outcome || 'budget exhausted'), true);
    endStream();
  }
};

function showBanner(msg, dnf) {
  var b = document.getElementById('banner');
  b.textContent = msg; b.style.display = 'block';
  if (dnf) b.classList.add('dnf');
}
</script>
</body>
</html>"""

_WATCH_LIST_HTML = """<!DOCTYPE html>
<html>
<head>
<title>LabyrinthBench</title>
<style>
body { background:#0a0a0a; color:#ccc; font-family:'Courier New',monospace; margin:0; padding:20px; }
h1 { color:#00ffcc; font-size:18px; letter-spacing:2px; margin:0 0 18px; }
table { border-collapse:collapse; width:100%; }
th { color:#444; font-size:10px; text-transform:uppercase; letter-spacing:1px; text-align:left; padding:6px 12px; border-bottom:1px solid #222; }
td { padding:7px 12px; border-bottom:1px solid #111; font-size:13px; }
a { color:#00ffcc; text-decoration:none; } a:hover { text-decoration:underline; }
.running { color:#00ff88; } .exit { color:#ffdd00; } .dnf { color:#555; }
#pages { margin-top:14px; font-size:12px; color:#444; }
#pages a { margin:0 3px; } #pages b { color:#ccc; margin:0 3px; }
#refresh { color:#333; font-size:11px; margin-top:8px; }
</style>
__AUTO_REFRESH__
</head>
<body>
<h1>&#x2B21; LABYRINTHBENCH</h1>
<table>
<tr><th>Session</th><th>Model</th><th>DEG</th><th>Steps</th><th>Status</th><th></th></tr>
__ROWS__
</table>
<div id="pages">__PAGINATION__</div>
<p id="refresh">__REFRESH_NOTE__</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "degs": list(_degs)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/sessions")
def list_sessions():
    rows = [
        {
            "session_id": s.session_id,
            "kind": "maze",
            "deg_id": s.deg.id,
            "steps_used": s.steps_used,
            "step_budget": s.deg.step_budget,
            "current_node": s.current_node_id,
            "completed": s.completed,
            "found_exit": s.found_exit,
            "model": s.model,
            "started_at": _session_started.get(s.session_id),
        }
        for s in _sessions.values()
    ]
    rows.extend(_sandbox_row(sb) for sb in _sandbox_sessions.values())
    return rows


@app.get("/watch", response_class=HTMLResponse)
def watch_list(page: int = Query(1, ge=1)):
    # Merge live + history (live first, then history newest-first)
    all_rows: list[dict] = []
    live_ids = set()
    for s in reversed(list(_sessions.values())):
        live_ids.add(s.session_id)
        last_seen = _session_last_act.get(s.session_id) or _session_started.get(s.session_id) or time.time()
        stale = (not s.completed) and (time.time() - last_seen > STALE_AFTER_S)
        status_cls = "exit" if s.found_exit else ("dnf" if (s.completed or stale) else "running")
        all_rows.append({
            "session_id": s.session_id,
            "model": s.model or "unknown",
            "deg_id": s.deg.id,
            "steps": f"{s.steps_used}&nbsp;/&nbsp;{s.deg.step_budget}",
            "status": ("EXIT ✓" if s.found_exit else ("DNF" if s.completed else ("stalled (no client activity)" if stale else "running..."))),
            "status_cls": status_cls,
            # Completed sessions still in memory replay fine — the stream sends
            # full history then closes — so the watch link stays.
            "link": f'<a href="/watch/{s.session_id}">watch</a>',
        })
    for entry in reversed(_history):
        if entry["session_id"] in live_ids:
            continue
        status_cls = "exit" if entry.get("found_exit") else "dnf"
        all_rows.append({
            "session_id": entry["session_id"],
            "model": entry.get("model", "?"),
            "deg_id": entry.get("deg_id", "?"),
            "steps": f"{entry.get('steps_used','?')}&nbsp;/&nbsp;{entry.get('step_budget','?')}",
            "status": ("EXIT ✓" if entry.get("found_exit") else f"DNF ({entry.get('failure_reason','')})"),
            "status_cls": status_cls,
            "link": "",
        })

    total = len(all_rows)
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = min(page, total_pages)
    page_rows = all_rows[(page - 1) * PER_PAGE: page * PER_PAGE]

    rows_html = ""
    for r in page_rows:
        rows_html += (
            f"<tr><td>{r['session_id'][:8]}</td><td>{r['model']}</td>"
            f"<td>{r['deg_id']}</td><td>{r['steps']}</td>"
            f"<td class='{r['status_cls']}'>{r['status']}</td><td>{r['link']}</td></tr>"
        )
    if not rows_html:
        rows_html = "<tr><td colspan='6' style='color:#333'>No sessions yet.</td></tr>"

    pages_html = ""
    if total_pages > 1:
        for p in range(1, total_pages + 1):
            pages_html += f"<b>{p}</b> " if p == page else f'<a href="/watch?page={p}">{p}</a> '

    has_live = any(not s.completed for s in _sessions.values())
    auto_refresh = '<script>setTimeout(function(){location.reload()},3000);</script>' if has_live else ""
    refresh_note = "auto-refreshes every 3s" if has_live else f"{total} sessions total"

    return (
        _WATCH_LIST_HTML
        .replace("__ROWS__", rows_html)
        .replace("__PAGINATION__", pages_html)
        .replace("__AUTO_REFRESH__", auto_refresh)
        .replace("__REFRESH_NOTE__", refresh_note)
    )


@app.get("/watch/{session_id}", response_class=HTMLResponse)
def watch_session(session_id: str):
    s = _get_session(session_id)
    cy_nodes, cy_edges = _build_cy_elements(s.deg)
    html = _WATCH_SESSION_HTML
    html = html.replace("__SESSION_ID__", session_id)
    html = html.replace("__SID__", session_id[:8])
    html = html.replace("__MODEL__", s.model or "unknown")
    html = html.replace("__DEG__", s.deg.id)
    html = html.replace("__BUDGET__", str(s.deg.step_budget))
    html = html.replace("__OPTIMAL__", str(s.deg.optimal_commits))
    html = html.replace("__OPTIMAL_PATH_JSON__", json.dumps(s.deg.optimal_path))
    html = html.replace("__CY_NODES_JSON__", json.dumps(cy_nodes))
    html = html.replace("__CY_EDGES_JSON__", json.dumps(cy_edges))
    return html


@app.get("/degs")
def list_degs():
    return [
        {"id": d.id, "name": d.name, "optimal_commits": d.optimal_commits,
         "step_budget": d.step_budget, "gate_count": d.gate_count}
        for d in _degs.values()
    ]


@app.post("/session")
def create_session(req: CreateSessionRequest):
    deg = _resolve_deg(req.deg_id)
    if deg is None:
        raise HTTPException(status_code=404, detail=f"DEG {req.deg_id!r} not found.")
    session = new_session(deg)
    session.model = req.model
    if req.fog_radius is not None:
        session.fog_radius = req.fog_radius
    session.show_recall = req.show_recall
    session.show_state = req.show_state
    if req.state_label not in ("", "verified"):
        raise HTTPException(status_code=400, detail=f"Unknown state_label {req.state_label!r}.")
    session.allow_pull = req.allow_pull
    session.state_stub = req.state_stub
    session.state_label = req.state_label
    _sessions[session.session_id] = session
    _session_started[session.session_id] = time.time()
    briefing = session.deg.briefing
    if "{" in briefing:
        briefing = briefing.format(fog_radius=session.fog_radius, max_wrong=session.max_wrong)
    return {
        "session_id": session.session_id,
        "deg_id": req.deg_id,
        "step_budget": session.deg.step_budget,
        "optimal_commits": session.deg.optimal_commits,
        "briefing": briefing,
        "fog_radius": session.fog_radius,
    }


@app.post("/act")
def act(req: ActRequest):
    session = _get_session(req.session_id)
    _session_last_act[req.session_id] = time.time()

    if session.completed and req.action != "score":
        return {"ok": False, "error": "Session already ended.", "completed": True}

    if req.action == "observe":
        result = session.observe()
    elif req.action == "inspect":
        result = session.inspect(req.path_id)
    elif req.action == "commit":
        result = session.commit(req.path_id, req.answer)
    elif req.action == "note":
        result = session.note_action(req.text)
    elif req.action == "pull":
        if not session.allow_pull:
            raise HTTPException(status_code=400, detail="Pull is not enabled for this session.")
        result = session.pull_state()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action {req.action!r}.")

    result["completed"] = session.completed
    result["found_exit"] = session.found_exit
    result["steps_used"] = session.steps_used

    if req.injected_context and session.events:
        session.events[-1].injected_context = req.injected_context

    if session.completed:
        _record_session(session)

    return result


@app.get("/session/{session_id}/state")
def session_state(session_id: str):
    """Return authoritative session state for oracle harness state-block construction."""
    s = _get_session(session_id)
    dead_ends = {
        e.node_id for e in s.events
        if e.outcome in ("dead_end", "wrong", "loop_trapped")
    }
    return {
        "current_node_id": s.current_node_id,
        "gate_results": s.gate_results,
        "confirmed_dead_ends": sorted(dead_ends),
        "steps_used": s.steps_used,
        "step_budget": s.deg.step_budget,
        "note": s.note,
        "pull_count": s.pull_count,
        "var_ledger": s.var_ledger,
        "completed": s.completed,
    }


@app.get("/score/{session_id}")
def score(session_id: str):
    session = _get_session(session_id)
    if not session.completed:
        raise HTTPException(status_code=400, detail="Session not yet completed.")
    _record_session(session)
    return session.score()


@app.get("/eval-status")
def eval_status():
    locks = list(Path("/results").glob(".eval_lock_*"))
    if not locks:
        return {"running": False}
    machines = {}
    any_running = False
    for lock_path in locks:
        try:
            data = json.loads(lock_path.read_text())
            try:
                os.kill(data["pid"], 0)
                machines[data["base_url"]] = {**data, "status": "running"}
                any_running = True
            except ProcessLookupError:
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass
    return {"running": any_running, "machines": machines}


@app.get("/session/{session_id}/graph")
def session_graph(session_id: str):
    """Return Cytoscape-ready graph data for a live session."""
    s = _get_session(session_id)
    nodes, edges = _build_cy_elements(s.deg)
    return {
        "session_id": s.session_id,
        "deg_id": s.deg.id,
        "model": s.model,
        "nodes": nodes,
        "edges": edges,
        "optimal_path": s.deg.optimal_path,
        "budget": s.deg.step_budget,
        "optimal": s.deg.optimal_commits,
        "events": [_event_to_sse(e, s) for e in s.events],
    }


@app.get("/deg/{deg_id}/graph")
def deg_graph(deg_id: str):
    """Return Cytoscape-ready graph data for a DEG (no session needed — for history replay)."""
    deg = _resolve_deg(deg_id)
    if deg is None:
        raise HTTPException(status_code=404, detail=f"DEG {deg_id!r} not found.")
    nodes, edges = _build_cy_elements(deg)
    return {
        "deg_id": deg.id,
        "nodes": nodes,
        "edges": edges,
        "optimal_path": deg.optimal_path,
        "budget": deg.step_budget,
        "optimal": deg.optimal_commits,
    }


@app.get("/results")
def list_results():
    """Return session history for the history page."""
    return list(reversed(_history))


@app.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """SSE stream. Replays full event history on connect, then polls for new events at 0.3s.

    Polling from session.events means a late-joining browser gets the full traversal
    history immediately without waiting for the next action to fire.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    s = _sessions[session_id]

    async def generator():
        seen = 0
        last_hb = time.monotonic()
        while True:
            for evt in s.events[seen:]:
                seen += 1
                yield f"data: {json.dumps(_event_to_sse(evt, s))}\n\n"
            if s.completed and seen >= len(s.events):
                break
            await asyncio.sleep(0.3)
            if time.monotonic() - last_hb > 30:
                yield 'data: {"heartbeat": true}\n\n'
                last_hb = time.monotonic()

    return StreamingResponse(generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# External-validation shell-sandbox: live event relay
# ---------------------------------------------------------------------------
# The sandbox harness (cli/run_sandbox.py) is self-contained — it drives a docker
# target + the LLM directly. These endpoints let it RELAY per-turn events so the
# wali-ui LB tab can stream a run live: the model's CONTEXT next to its thinking,
# chosen action, and the command result. The maze path above is untouched.

def _sandbox_row(sb: dict) -> dict:
    """Shape a sandbox session like a maze session row so the existing UI list renders it."""
    return {
        "session_id": sb["session_id"],
        "kind": "sandbox",
        "deg_id": f"rung{sb['rung']}/{sb['arm']}",
        "model": sb["model"],
        "rung": sb["rung"],
        "arm": sb["arm"],
        "steps_used": len(sb["events"]),
        "step_budget": sb.get("backstop") or 0,
        "current_node": None,
        "completed": sb["completed"],
        "found_exit": bool(sb.get("found_done")),
        "subgoal_depth": sb.get("subgoal_depth"),
        "failure_reason": sb.get("failure_reason"),
        "started_at": sb.get("started_at"),
        "finished_at": sb.get("finished_at"),
    }


@app.post("/sandbox/session")
def sandbox_create(req: SandboxSessionRequest):
    _sandbox_sessions[req.session_id] = {
        "session_id": req.session_id, "kind": "sandbox",
        "model": req.model, "rung": req.rung, "arm": req.arm,
        "num_ctx": req.num_ctx, "label": req.label, "seed": req.seed,
        "backstop": req.backstop,
        "started_at": time.time(), "finished_at": None,
        "completed": False, "found_done": False,
        "subgoal_depth": None, "checks_passed": None, "checks_total": None,
        "failure_reason": None, "events": [],
    }
    return {"session_id": req.session_id}


@app.post("/sandbox/session/{session_id}/turn")
def sandbox_turn(session_id: str, req: SandboxTurnRequest):
    sb = _sandbox_sessions.get(session_id)
    if sb is None:
        raise HTTPException(status_code=404, detail=f"Sandbox session {session_id!r} not found.")
    sb["events"].append(req.model_dump())
    return {"ok": True, "turns": len(sb["events"])}


@app.post("/sandbox/session/{session_id}/complete")
def sandbox_complete(session_id: str, req: SandboxCompleteRequest):
    sb = _sandbox_sessions.get(session_id)
    if sb is None:
        raise HTTPException(status_code=404, detail=f"Sandbox session {session_id!r} not found.")
    sb.update(completed=True, finished_at=time.time(), found_done=req.found_done,
              failure_reason=req.failure_reason, subgoal_depth=req.subgoal_depth,
              checks_passed=req.checks_passed, checks_total=req.checks_total)
    # Persist a history row (kind:sandbox) so it shows on the history page like maze runs.
    if not any(e["session_id"] == session_id for e in _history):
        _history.append({
            "session_id": session_id, "kind": "sandbox", "model": sb["model"],
            "deg_id": f"rung{sb['rung']}/{sb['arm']}", "rung": sb["rung"], "arm": sb["arm"],
            "steps_used": len(sb["events"]), "step_budget": sb.get("backstop") or 0,
            "found_exit": bool(req.found_done), "completed": True,
            "failure_reason": req.failure_reason, "subgoal_depth": req.subgoal_depth,
            "checks_passed": req.checks_passed, "checks_total": req.checks_total,
            "collateral_count": req.collateral_count,
            "started_at": sb.get("started_at"), "finished_at": sb["finished_at"],
        })
        _save_history()
    return {"ok": True}


@app.get("/sandbox/sessions")
def sandbox_sessions():
    return [_sandbox_row(sb) for sb in _sandbox_sessions.values()]


@app.get("/sandbox/stream/{session_id}")
async def sandbox_stream(session_id: str):
    """SSE: a meta frame, then replay the run's turn events, then poll for new ones (mirrors /stream)."""
    if session_id not in _sandbox_sessions:
        raise HTTPException(status_code=404, detail=f"Sandbox session {session_id!r} not found.")
    sb = _sandbox_sessions[session_id]

    async def generator():
        yield f"data: {json.dumps({'meta': _sandbox_row(sb), 'num_ctx': sb.get('num_ctx')})}\n\n"
        seen = 0
        last_hb = time.monotonic()
        while True:
            for evt in sb["events"][seen:]:
                seen += 1
                yield f"data: {json.dumps(evt)}\n\n"
            if sb["completed"] and seen >= len(sb["events"]):
                yield f"data: {json.dumps({'completed': True, 'summary': _sandbox_row(sb)})}\n\n"
                break
            await asyncio.sleep(0.3)
            if time.monotonic() - last_hb > 30:
                yield 'data: {"heartbeat": true}\n\n'
                last_hb = time.monotonic()

    return StreamingResponse(generator(), media_type="text/event-stream")
