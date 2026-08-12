"""Fixture prep for the renderer rung (instrument-side, disclosed).

Converts one completed run journal + its DEG into the two clean JSON inputs the
render_trace.py task consumes, so the contributor model's artifact can be
stdlib-only (no pyyaml) inside the --network none target.

- journal.json — render-relevant subset of the run record (drops turns_log:
  it is huge and carries prompts, which are not the renderer's business).
- deg.json — the DEG flattened to nodes + a flat edge list. Gate edges keep
  gated=true; a gate's wrong_destination becomes its own edge with wrong=true
  (wrong-answer routing is real maze structure — without it the dz* dead ends
  would be orphan nodes).

Usage (inside the labyrinthbench container, cwd /app):
  python3 scripts/make_renderer_fixtures.py results/trace-120b.jsonl \
      sandbox/rungs/renderer_fixtures/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

JOURNAL_FIELDS = ["session_id", "deg_id", "model", "found_exit", "steps_to_exit",
                  "step_budget", "optimal_commits"]
EVENT_FIELDS = ["action", "node_id", "steps_used", "outcome", "gate_correct"]


def convert_deg(deg_path: Path) -> dict:
    data = yaml.safe_load(deg_path.read_text())
    nodes, edges = [], []
    for n in data["nodes"]:
        nodes.append({"id": n["id"], "terminal": bool(n.get("terminal", False))})
        for p in n.get("paths") or []:
            gate = p.get("gate")
            edges.append({"src": n["id"], "dst": p["destination"],
                          "gated": gate is not None, "wrong": False})
            if gate and gate.get("wrong_destination"):
                edges.append({"src": n["id"], "dst": gate["wrong_destination"],
                              "gated": True, "wrong": True})
    return {"id": data["meta"]["id"], "nodes": nodes, "edges": edges}


def convert_journal(run: dict) -> dict:
    out = {k: run[k] for k in JOURNAL_FIELDS if k in run}
    out["events"] = [{k: e.get(k) for k in EVENT_FIELDS} for e in run["events"]]
    return out


def main() -> None:
    journal_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    # single-run JSONL (one line) — same assumption as scripts/show_trace.py
    run = json.loads(journal_path.read_text().strip().splitlines()[0])
    deg = convert_deg(Path("degs") / f"{run['deg_id']}.yaml")

    node_ids = {n["id"] for n in deg["nodes"]}
    for e in deg["edges"]:
        assert e["src"] in node_ids and e["dst"] in node_ids, e
    for ev in run["events"]:
        assert ev["node_id"] in node_ids, ev

    (out_dir / "journal.json").write_text(json.dumps(convert_journal(run), indent=2) + "\n")
    (out_dir / "deg.json").write_text(json.dumps(deg, indent=2) + "\n")
    print(f"journal.json: {len(run['events'])} events  deg.json: "
          f"{len(deg['nodes'])} nodes / {len(deg['edges'])} edges  -> {out_dir}")


if __name__ == "__main__":
    main()
