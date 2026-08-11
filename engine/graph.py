"""DEG (deterministic evaluation graph) loader and pre-computation utilities."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Gate:
    problem: str
    answer: str
    # wrong_destination: where a wrong answer routes. When None (omitted in YAML) the gate is a
    # LOCK — a wrong answer does not open it; the model stays put and spends a step (nav-1 rung).
    wrong_destination: Optional[str] = None
    # Phase 1: dependent gate chain support (all optional — backward compat with Phase 0 YAML).
    # depends_on may be a single gate-id (str) OR a list of gate-ids (multi-dependency / synthesis
    # gates, e.g. S = A3 + B2). answer_fn references deps by gate-id (and `prev_result` for the
    # single-dep case). __UNRESOLVABLE__ if ANY dependency is missing.
    gate_id: Optional[str] = None
    problem_template: Optional[str] = None
    answer_fn: Optional[str] = None
    depends_on: Optional[object] = None  # str (single) | list[str] (multi-dep / synthesis) | None
    seed: dict = field(default_factory=dict)
    # Belief-revision rung (rev-1): on a CORRECT commit, write this gate's answer into the session's
    # var_ledger under this name, overwriting any prior binding. answer_fn/depends_on of later gates
    # reference variable names directly (resolved from the ledger's CURRENT value). The ledger is the
    # curated "current state" the HUD shows (render_state) — distinct from the raw gate-answer recall.
    sets_var: Optional[str] = None

    _SAFE = {"__builtins__": {}, "int": int, "float": float, "abs": abs, "round": round, "min": min, "max": max}

    @property
    def dep_ids(self) -> list:
        """Dependency gate-ids as a list (depends_on may be str, list, or None)."""
        if not self.depends_on:
            return []
        return [self.depends_on] if isinstance(self.depends_on, str) else list(self.depends_on)

    def _ctx(self, values: dict) -> dict:
        """answer_fn namespace: seed + each dependency's value keyed by gate-id, plus `prev_result`
        for the single-dependency case (back-compat)."""
        ctx = dict(self.seed)
        deps = self.dep_ids
        for d in deps:
            ctx[d] = values.get(d)
        if len(deps) == 1:
            ctx["prev_result"] = values.get(deps[0])
        return ctx

    def resolved_problem(self, gate_results: dict, var_ledger: Optional[dict] = None) -> str:
        """Return the gate problem text, resolving template+seed if present. Variables (current
        ledger bindings) and gate-ids share one flat namespace; variables take precedence."""
        if self.problem_template is None:
            return self.problem
        values = {**(var_ledger or {}), **gate_results}
        ctx = dict(self.seed)
        for d in self.dep_ids:
            ctx[d] = values.get(d, "?")
        if len(self.dep_ids) == 1:
            ctx["prev_result"] = values.get(self.dep_ids[0], "?")
        return self.problem_template.format(**ctx)

    def resolved_answer(self, gate_results: dict, var_ledger: Optional[dict] = None) -> str:
        """Return the expected answer, evaluating answer_fn+seed if present. Dependencies resolve
        from the merged {ledger + gate_results} namespace (a dep may be a variable name or a
        gate-id); __UNRESOLVABLE__ if any is missing."""
        if self.answer_fn is None:
            return self.answer
        values = {**(var_ledger or {}), **gate_results}
        if any(values.get(d) is None for d in self.dep_ids):
            return "__UNRESOLVABLE__"
        return str(eval(self.answer_fn, self._SAFE, self._ctx(values)))

    def answer_given_prev(self, prev) -> Optional[str]:
        """Expected answer derivable from the model's OWN prior submission(s) — not ground truth —
        for ``knowledge_state_consistency``: "did it faithfully execute the program off its own
        answers?" ``prev`` is a str (single-dep, back-compat) or a {gate_id: submission} dict
        (multi-dep). Returns None when not computable (no answer_fn / any dep missing / eval error)."""
        if self.answer_fn is None:
            return self.answer if not self.depends_on else None
        deps = self.dep_ids
        values = prev if isinstance(prev, dict) else ({deps[0]: prev} if deps else {})
        if any(values.get(d) is None for d in deps):
            return None
        try:
            return str(eval(self.answer_fn, self._SAFE, self._ctx(values)))
        except Exception:
            return None


@dataclass
class DEGPath:
    id: str
    label: str
    destination: str
    gate: Optional[Gate] = None

    @property
    def is_gated(self) -> bool:
        return self.gate is not None


@dataclass
class DEGNode:
    id: str
    description: str
    paths: list[DEGPath]
    terminal: bool = False

    def get_path(self, path_id: str) -> Optional[DEGPath]:
        for p in self.paths:
            if p.id == path_id:
                return p
        return None


@dataclass
class DEG:
    id: str
    name: str
    nodes: dict[str, DEGNode]
    start_node_id: str
    optimal_commits: int
    optimal_path: list[str]
    step_budget: int
    gate_count: int = 0
    dead_end_count: int = 0
    loop_count: int = 0
    dead_end_patience: int = 3       # consecutive non-commit actions at a dead end before auto-fail
    dead_end_revisit_limit: int = 2  # visits to the same dead-end node before loop-trap fires
    briefing: str = ""               # optional task-framing prompt prepended to the system message
    fog_radius: int = 0              # 0 = no map (full reveal, legacy); N = fog-of-war, see N corridors out
    max_wrong: int = 0               # ramp mode: wrong-answer budget (0 = disabled); out of lives = done
    dist_to_exit: dict = field(default_factory=dict)  # node_id → min commits to exit; None if unreachable

    def node(self, node_id: str) -> DEGNode:
        return self.nodes[node_id]

    @property
    def start(self) -> DEGNode:
        return self.nodes[self.start_node_id]


def load_deg(path: Path) -> DEG:
    with open(path) as f:
        return load_deg_dict(yaml.safe_load(f))


def load_deg_dict(data: dict) -> DEG:
    """Build a DEG from an already-parsed manifest dict (no file round-trip)."""
    meta = data["meta"]
    nodes: dict[str, DEGNode] = {}

    for nd in data["nodes"]:
        paths = []
        for pd in nd.get("paths", []):
            gate = None
            if "gate" in pd:
                gd = pd["gate"]
                gate = Gate(
                    problem=gd.get("problem", ""),
                    answer=str(gd.get("answer", "")),
                    wrong_destination=gd.get("wrong_destination"),  # None → lock (stay put on wrong)
                    gate_id=gd.get("gate_id"),
                    problem_template=gd.get("problem_template"),
                    answer_fn=gd.get("answer_fn"),
                    depends_on=gd.get("depends_on"),
                    seed=gd.get("seed") or {},
                    sets_var=gd.get("sets_var"),
                )
            # str() coercion: YAML parses bare on/off/yes/no/true/false as booleans, so a path
            # id like `on` would silently become True and never match a commit. Force strings.
            _pid = str(pd["id"])
            paths.append(DEGPath(
                id=_pid,
                label=str(pd.get("label", _pid)),
                destination=str(pd["destination"]),
                gate=gate,
            ))
        nodes[nd["id"]] = DEGNode(
            id=nd["id"],
            description=nd.get("description", "").strip(),
            paths=paths,
            terminal=nd.get("terminal", False),
        )

    deg = DEG(
        id=meta["id"],
        name=meta["name"],
        nodes=nodes,
        start_node_id="start",
        optimal_commits=meta["optimal_commits"],
        optimal_path=meta["optimal_path"],
        step_budget=meta["step_budget"],
        gate_count=meta.get("gate_count", 0),
        dead_end_count=meta.get("dead_end_count", 0),
        loop_count=meta.get("loop_count", 0),
        dead_end_patience=meta.get("dead_end_patience", 3),
        briefing=(meta.get("briefing") or "").strip(),
        fog_radius=int(meta.get("fog_radius", 0) or 0),
        max_wrong=int(meta.get("max_wrong", 0) or 0),
    )
    deg.dist_to_exit = compute_dist_to_exit(deg)
    return deg


def compute_dist_to_exit(deg: DEG) -> dict[str, int]:
    """Reverse BFS from exit over correct-destination edges.

    Returns node_id → min commits to reach exit. Nodes with no path to exit are absent.
    """
    # Build reverse adjacency: dest → list of sources (correct-gate edges only)
    reverse: dict[str, list[str]] = {nid: [] for nid in deg.nodes}
    for node in deg.nodes.values():
        for path in node.paths:
            reverse[path.destination].append(node.id)

    dist: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    # Seed from all terminal nodes
    for node in deg.nodes.values():
        if node.terminal:
            dist[node.id] = 0
            queue.append((node.id, 0))

    while queue:
        node_id, d = queue.popleft()
        for src in reverse[node_id]:
            if src not in dist:
                dist[src] = d + 1
                queue.append((src, d + 1))

    return dist


def bfs_verify(deg: DEG) -> tuple[list[str], int]:
    """BFS over correct-gate edges to find shortest commit path to any terminal node.

    Returns (node_id_path, commit_count). Raises ValueError if no path exists.
    Used at startup to sanity-check the DEG manifest against its declared optimal_commits.
    """
    # BFS state: (current_node_id, path_taken, commit_count)
    queue: deque[tuple[str, list[str], int]] = deque()
    queue.append((deg.start_node_id, [deg.start_node_id], 0))
    visited: set[str] = set()

    while queue:
        node_id, path, commits = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)

        node = deg.node(node_id)
        if node.terminal:
            return path, commits

        for p in node.paths:
            dest = p.destination  # always use correct destination for BFS
            if dest not in visited:
                queue.append((dest, path + [dest], commits + 1))

    raise ValueError(f"DEG {deg.id!r}: no path from start to any terminal node")


def visible_map(deg: DEG, current_id: str, visited, radius: int) -> dict:
    """Fog-of-war visibility for the navigable rung.

    Returns the *topology* the model is allowed to see: explored corridors (every edge out of a
    node it has visited) plus the radius-R neighbourhood of the current node over **correct**
    edges. Reveals node names + edges (open/gated flag + correct destination); **withholds room
    interiors** (descriptions + gate problems) until a node is visited, and **never reveals
    wrong-destinations**. A node at the radius frontier is seen by name but its onward corridors
    stay fogged (it is not "expanded").

    radius 0 → only the current node's own corridors (caller normally skips the map at 0).
    """
    visited = set(visited)
    edges: list[tuple] = []          # (src, path_id, gated, dst)
    expanded: set[str] = set()       # nodes whose corridors are known (visited or within radius)

    def expand(nid: str) -> None:
        if nid in expanded or nid not in deg.nodes:
            return
        expanded.add(nid)
        for p in deg.nodes[nid].paths:
            edges.append((nid, p.id, p.is_gated, p.destination))

    for nid in visited:
        expand(nid)

    queue: deque[tuple[str, int]] = deque([(current_id, 0)])
    seen = {current_id}
    while queue:
        nid, d = queue.popleft()
        if d < radius:
            expand(nid)
            for p in deg.nodes.get(nid, DEGNode(nid, "", [])).paths:
                if p.destination not in seen:
                    seen.add(p.destination)
                    queue.append((p.destination, d + 1))

    nodes = set(visited) | {current_id} | {e[0] for e in edges} | {e[3] for e in edges}
    return {
        "current": current_id,
        "radius": radius,
        "edges": edges,
        "nodes": nodes,
        "visited": visited,
        "expanded": expanded,
    }


def load_all_degs(degs_dir: Path) -> dict[str, DEG]:
    result: dict[str, DEG] = {}
    for f in sorted(degs_dir.glob("*.yaml")):
        deg = load_deg(f)
        result[deg.id] = deg
    return result
