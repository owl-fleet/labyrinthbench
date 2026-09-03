"""Session state and action dispatcher for LabyrinthBench."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .gate_bank import score_gate
from .graph import DEG, DEGNode, visible_map
from .renderer import (
    render_budget_exhausted,
    render_commit_result,
    render_impossible,
    render_inspect,
    render_loop_trapped,
    render_map,
    render_note_stored,
    render_observe,
    render_recall,
    render_state,
    render_state_stub,
)


@dataclass
class GateAttempt:
    node_id: str
    path_id: str
    correct: bool
    answer_given: str
    gate_id: Optional[str] = None     # for chain-metric attribution
    depends_on: Optional[str] = None  # chain dependency (None = independent gate)


@dataclass
class SessionEvent:
    """Emitted after every action — streamed to WaliUI minimap via SSE."""
    action: str                    # observe | inspect | commit | note | pull
    node_id: str
    steps_used: int
    found_exit: bool
    budget_exhausted: bool
    gate_attempt: Optional[GateAttempt] = None
    outcome: Optional[str] = None  # ok | wrong | dead_end | back | budget_exhausted
    injected_context: Optional[str] = None  # context shown to model this turn (set by /act caller)


@dataclass
class Session:
    session_id: str
    deg: DEG
    current_node_id: str
    model: str = ""
    fog_radius: int = 0  # active fog-of-war radius (0 = no map); set from DEG or overridden per run
    show_recall: bool = False  # externalize gate answers into the overlay (HUD-as-working-memory arm)
    show_state: bool = False  # externalize the CURRENT variable ledger into the overlay (revision arm)
    allow_pull: bool = False  # pull action enabled (pull-HUD arms; API 400s the action when False)
    state_stub: bool = False  # push tracked-variable NAMES into the overlay, values via pull (hybrid arm)
    state_label: str = ""     # "" | "verified" — [STATE] header authority label (epistemic-label arm)
    pull_count: int = 0       # pulls issued (each costs a step; steps_used includes them)
    tracked_vars: list = field(default_factory=list)  # all sets_var names the DEG declares (stub source)
    var_ledger: dict = field(default_factory=dict)  # variable name → current value (latest sets_var wins)
    visited: set = field(default_factory=set)  # every node ever entered (fog-of-war memory)
    traversal_stack: list[str] = field(default_factory=list)
    steps_used: int = 0
    note: str = ""
    found_exit: bool = False
    budget_exhausted: bool = False
    dead_end_trapped: bool = False
    loop_trapped: bool = False
    impossible: bool = False
    out_of_lives: bool = False     # ramp mode: exhausted the wrong-answer budget (max_wrong)
    max_wrong: int = 0             # wrong-answer budget (0 = disabled; ramp DEGs set 3-5)
    wrong_count: int = 0           # wrong gate answers so far
    dead_end_turn_count: int = 0  # consecutive non-commit actions at a dead-end node
    dead_end_visits: dict = field(default_factory=dict)  # node_id → visit count
    started_at: float = field(default_factory=time.monotonic)
    ended_at: Optional[float] = None
    gate_attempts: list[GateAttempt] = field(default_factory=list)
    gate_results: dict[str, str] = field(default_factory=dict)  # gate_id → answer (correct commits only)
    events: list[SessionEvent] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return (self.found_exit or self.budget_exhausted or self.dead_end_trapped
                or self.loop_trapped or self.impossible or self.out_of_lives)

    def _register_wrong(self) -> bool:
        """Count a wrong gate answer against the ramp's life budget. Returns True if exhausted."""
        if not self.max_wrong:
            return False
        self.wrong_count += 1
        if self.wrong_count >= self.max_wrong:
            self.out_of_lives = True
            self.ended_at = time.monotonic()
            return True
        return False

    @property
    def current_node(self) -> DEGNode:
        return self.deg.node(self.current_node_id)

    def _emit(self, event: SessionEvent) -> None:
        self.events.append(event)

    def _at_dead_end(self) -> bool:
        node = self.current_node
        return not node.terminal and not node.paths

    def _overlay_block(self, force_state: bool = False, include_stub: bool = True) -> str:
        """The curated HUD prefix for the node view: fog map + externalized recall.

        - fog map: active radius>0 → radius-R map; ==0 on a fog DEG → explicit 'no map'; legacy → none.
        - recall: when show_recall, the model's recorded gate answers (HUD-as-working-memory).
        - state: pushed every turn when show_state; force_state=True renders it on demand (a pull).
        - stub: when state_stub and the full ledger isn't shown, push variable NAMES only —
          include_stub=False keeps a pull response from telling the model to pull."""
        parts: list[str] = []
        if self.fog_radius:
            parts.append(render_map(visible_map(self.deg, self.current_node_id, self.visited, self.fog_radius)))
        elif self.deg.fog_radius:
            parts.append("[MAP: none — total fog this run; explore to learn the layout.]")
        if self.show_recall:
            parts.append(render_recall(self.gate_results))
        if self.show_state or force_state:
            parts.append(render_state(self.var_ledger, self.state_label))
        elif self.state_stub and include_stub:
            parts.append(render_state_stub(self.tracked_vars))
        return ("\n\n".join(parts) + "\n\n") if parts else ""

    def _tick_dead_end(self) -> Optional[dict]:
        """Increment dead-end counter; return failure response if patience exceeded, else None."""
        if not self._at_dead_end():
            self.dead_end_turn_count = 0
            return None
        self.dead_end_turn_count += 1
        if self.dead_end_turn_count >= self.deg.dead_end_patience:
            self.dead_end_trapped = True
            self.ended_at = time.monotonic()
            self._emit(SessionEvent(
                action="observe",
                node_id=self.current_node_id,
                steps_used=self.steps_used,
                found_exit=False,
                budget_exhausted=False,
                outcome="dead_end_trapped",
            ))
            # Return the same observe text — no hint that the trap fired
            text = render_observe(
                self.current_node,
                self.steps_used,
                self.deg.step_budget,
                self.note,
                traversal_depth=len(self.traversal_stack),
                gate_results=self.gate_results,
                var_ledger=self.var_ledger,
            )
            return {"ok": True, "text": text}
        return None

    def observe(self) -> dict:
        trap = self._tick_dead_end()
        if trap:
            return trap
        text = self._overlay_block() + render_observe(
            self.current_node,
            self.steps_used,
            self.deg.step_budget,
            self.note,
            traversal_depth=len(self.traversal_stack),
            gate_results=self.gate_results,
            var_ledger=self.var_ledger,
        )
        self._emit(SessionEvent(
            action="observe",
            node_id=self.current_node_id,
            steps_used=self.steps_used,
            found_exit=self.found_exit,
            budget_exhausted=self.budget_exhausted,
        ))
        return {"ok": True, "text": text}

    def inspect(self, path_id: str) -> dict:
        trap = self._tick_dead_end()
        if trap:
            return trap
        if path_id == "back":
            if not self.traversal_stack:
                return {"ok": False, "error": "Already at the start — cannot go back."}
            return {"ok": True, "text": "Path 'back' returns to your previous location. No gate."}

        path = self.current_node.get_path(path_id)
        if path is None:
            return {"ok": False, "error": f"No path '{path_id}' from current location."}

        text = render_inspect(path, gate_results=self.gate_results, var_ledger=self.var_ledger)
        self._emit(SessionEvent(
            action="inspect",
            node_id=self.current_node_id,
            steps_used=self.steps_used,
            found_exit=self.found_exit,
            budget_exhausted=self.budget_exhausted,
        ))
        return {"ok": True, "text": text}

    def commit(self, path_id: str, answer: str = "") -> dict:
        if self.completed:
            return {"ok": False, "error": "Session already ended."}
        self.dead_end_turn_count = 0  # any commit attempt resets the dead-end timer

        # Backtrack
        if path_id == "back":
            if not self.traversal_stack:
                return {"ok": False, "error": "Already at the start — cannot go back."}
            prev_node_id = self.traversal_stack.pop()
            self.steps_used += 1
            self.current_node_id = prev_node_id
            text = render_commit_result(
                "back", self.current_node, self.steps_used, self.deg.step_budget, ""
            )
            self._emit(SessionEvent(
                action="commit",
                node_id=self.current_node_id,
                steps_used=self.steps_used,
                found_exit=False,
                budget_exhausted=False,
                outcome="back",
            ))
            return {"ok": True, "outcome": "back", "node_id": self.current_node_id, "text": text}

        path = self.current_node.get_path(path_id)
        if path is None:
            return {"ok": False, "error": f"No path '{path_id}' from current location."}

        # Resolve destination and gate outcome
        gate_attempt: Optional[GateAttempt] = None
        gate_feedback = ""
        outcome = "ok"

        if path.is_gated:
            expected = path.gate.resolved_answer(self.gate_results, self.var_ledger)
            correct = score_gate(answer, expected)
            gate_attempt = GateAttempt(
                node_id=self.current_node_id,
                path_id=path_id,
                correct=correct,
                answer_given=answer,
                gate_id=path.gate.gate_id,
                depends_on=path.gate.depends_on,
            )
            self.gate_attempts.append(gate_attempt)
            if correct:
                destination = path.destination
                gate_feedback = "Gate answer: CORRECT"
                outcome = "ok"
                if path.gate.gate_id:
                    self.gate_results[path.gate.gate_id] = expected
                # Belief-revision: record/overwrite the variable's CURRENT value in the ledger.
                if path.gate.sets_var:
                    self.var_ledger[path.gate.sets_var] = expected
            elif path.gate.wrong_destination is None:
                # LOCK: a wrong answer does not open the gate. Stay put, spend a step. In ramp mode
                # it also costs a life; out of lives ends the run (score = depth reached).
                self.steps_used += 1
                out = self._register_wrong()
                budget_hit = self.steps_used >= self.deg.step_budget and not out
                if budget_hit:
                    self.budget_exhausted = True
                    self.ended_at = time.monotonic()
                end_outcome = "out_of_lives" if out else ("budget_exhausted" if budget_hit else "locked")
                self._emit(SessionEvent(
                    action="commit",
                    node_id=self.current_node_id,
                    steps_used=self.steps_used,
                    found_exit=False,
                    budget_exhausted=budget_hit,
                    gate_attempt=gate_attempt,
                    outcome=end_outcome,
                ))
                if out:
                    text = (f"--- OUT OF MOVES ---\nYou have used all {self.max_wrong} of your "
                            f"wrong-answer allowance. Session ended.")
                    return {"ok": True, "outcome": "out_of_lives", "node_id": self.current_node_id, "text": text}
                if budget_hit:
                    text = render_budget_exhausted(self.steps_used, self.deg.step_budget)
                    return {"ok": True, "outcome": "budget_exhausted", "node_id": self.current_node_id, "text": text}
                text = self._overlay_block() + render_commit_result(
                    "locked", self.current_node, self.steps_used, self.deg.step_budget,
                    "Gate answer: WRONG — the gate does not open.",
                )
                return {"ok": True, "outcome": "locked", "node_id": self.current_node_id, "text": text,
                        "gate_id": gate_attempt.gate_id}
            else:
                destination = path.gate.wrong_destination
                gate_feedback = "Gate answer: WRONG."
                outcome = "wrong"
                self._register_wrong()  # routing gates also cost a life in ramp mode
        else:
            destination = path.destination

        # Move
        self.traversal_stack.append(self.current_node_id)
        self.current_node_id = destination
        self.visited.add(destination)
        self.steps_used += 1

        new_node = self.current_node

        # Check terminal
        if new_node.terminal:
            self.found_exit = True
            self.ended_at = time.monotonic()
            outcome = "exit"
        # Check dead end (after wrong gate or just an open dead end)
        elif not new_node.paths:
            outcome = "dead_end"
            self.dead_end_visits[destination] = self.dead_end_visits.get(destination, 0) + 1
            if self.dead_end_visits[destination] >= self.deg.dead_end_revisit_limit:
                self.loop_trapped = True
                self.ended_at = time.monotonic()
                text = render_loop_trapped(self.steps_used, self.deg.step_budget)
                self._emit(SessionEvent(
                    action="commit",
                    node_id=self.current_node_id,
                    steps_used=self.steps_used,
                    found_exit=False,
                    budget_exhausted=False,
                    gate_attempt=gate_attempt,
                    outcome="loop_trapped",
                ))
                return {"ok": True, "outcome": "loop_trapped", "node_id": self.current_node_id, "text": text}

        # Check mathematical impossibility: exit unreachable within remaining budget
        if not self.found_exit and not self.loop_trapped:
            remaining = self.deg.step_budget - self.steps_used
            dist = self.deg.dist_to_exit.get(self.current_node_id)
            if dist is None and self.traversal_stack:
                # Dead-branch node: walk stack to find nearest spine ancestor + backtrack cost
                for depth, ancestor in enumerate(reversed(self.traversal_stack)):
                    ancestor_dist = self.deg.dist_to_exit.get(ancestor)
                    if ancestor_dist is not None:
                        dist = ancestor_dist + (depth + 1)
                        break
            if dist is not None and dist > remaining:
                self.impossible = True
                self.ended_at = time.monotonic()
                text = render_impossible(self.steps_used, self.deg.step_budget)
                self._emit(SessionEvent(
                    action="commit",
                    node_id=self.current_node_id,
                    steps_used=self.steps_used,
                    found_exit=False,
                    budget_exhausted=False,
                    gate_attempt=gate_attempt,
                    outcome="impossible",
                ))
                return {"ok": True, "outcome": "impossible", "node_id": self.current_node_id, "text": text}

        # Check budget exhaustion
        budget_hit = self.steps_used >= self.deg.step_budget
        if budget_hit and not self.found_exit:
            self.budget_exhausted = True
            self.ended_at = time.monotonic()
            text = render_budget_exhausted(self.steps_used, self.deg.step_budget)
            self._emit(SessionEvent(
                action="commit",
                node_id=self.current_node_id,
                steps_used=self.steps_used,
                found_exit=False,
                budget_exhausted=True,
                gate_attempt=gate_attempt,
                outcome="budget_exhausted",
            ))
            return {"ok": True, "outcome": "budget_exhausted", "node_id": self.current_node_id, "text": text}

        text = render_commit_result(
            outcome, new_node, self.steps_used, self.deg.step_budget, gate_feedback
        )
        self._emit(SessionEvent(
            action="commit",
            node_id=self.current_node_id,
            steps_used=self.steps_used,
            found_exit=self.found_exit,
            budget_exhausted=False,
            gate_attempt=gate_attempt,
            outcome=outcome,
        ))
        # gate_id: the engine's own public label for the gate this commit answered (None for an
        # ungated move) — relayed by the harness into ContextPolicy.turn_end so a ledger policy
        # labels a solve by the gate's name even when the model committed without observing the
        # room (MCV rung ii, 2026-09-02: `#2=12` for c1b). Additive; older clients ignore it.
        return {"ok": True, "outcome": outcome, "node_id": self.current_node_id, "text": text,
                "gate_id": gate_attempt.gate_id if gate_attempt else None}

    def note_action(self, text: str) -> dict:
        trap = self._tick_dead_end()
        if trap:
            return trap
        self.note = text[:500]  # cap note length
        response_text = render_note_stored(self.note)
        self._emit(SessionEvent(
            action="note",
            node_id=self.current_node_id,
            steps_used=self.steps_used,
            found_exit=self.found_exit,
            budget_exhausted=self.budget_exhausted,
        ))
        return {"ok": True, "text": response_text}

    def pull_state(self) -> dict:
        """Pull-HUD: the model requests the full current ledger on demand. Costs one step.

        Returns the [STATE] block plus a fresh node view, so an overlay-only cold prompt built
        from this response is self-sufficient (gate problem + paths + current values)."""
        if self.completed:
            return {"ok": False, "error": "Session already ended."}
        trap = self._tick_dead_end()
        if trap:
            return trap
        self.steps_used += 1
        self.pull_count += 1
        if self.steps_used >= self.deg.step_budget:
            self.budget_exhausted = True
            self.ended_at = time.monotonic()
            self._emit(SessionEvent(
                action="pull",
                node_id=self.current_node_id,
                steps_used=self.steps_used,
                found_exit=False,
                budget_exhausted=True,
                outcome="budget_exhausted",
            ))
            text = render_budget_exhausted(self.steps_used, self.deg.step_budget)
            return {"ok": True, "outcome": "budget_exhausted", "node_id": self.current_node_id, "text": text}
        text = self._overlay_block(force_state=True, include_stub=False) + render_observe(
            self.current_node,
            self.steps_used,
            self.deg.step_budget,
            self.note,
            traversal_depth=len(self.traversal_stack),
            gate_results=self.gate_results,
            var_ledger=self.var_ledger,
        )
        self._emit(SessionEvent(
            action="pull",
            node_id=self.current_node_id,
            steps_used=self.steps_used,
            found_exit=self.found_exit,
            budget_exhausted=False,
            outcome="ok",
        ))
        return {"ok": True, "outcome": "ok", "node_id": self.current_node_id, "text": text}

    def score(self) -> dict:
        total_gates = len(self.gate_attempts)
        correct_gates = sum(1 for g in self.gate_attempts if g.correct)
        gate_accuracy = correct_gates / total_gates if total_gates else None

        # path_correctness: fraction of commit()s where the model took a gate path correctly
        # (committed to a gated path AND got it right) vs total gated paths encountered
        path_correctness = gate_accuracy  # same signal at Phase 0 scale; diverges in Phase 1+

        # recovery_rate: of wrong-gate dead ends, how many did the model backtrack from?
        wrong_gates = [g for g in self.gate_attempts if not g.correct]
        # Count back actions that follow a wrong gate by checking event sequence
        recovery_count = 0
        for i, ev in enumerate(self.events):
            if ev.gate_attempt and not ev.gate_attempt.correct:
                # Look ahead for a "back" commit from the dead end
                for j in range(i + 1, min(i + 10, len(self.events))):
                    if self.events[j].outcome == "back":
                        recovery_count += 1
                        break
        recovery_rate = recovery_count / len(wrong_gates) if wrong_gates else None

        # ── chain-reasoning metrics (Phase 1 re-instrumentation) ──
        # chain_accuracy: of dependent spine-chain gates the model ATTEMPTED, fraction whose
        #   first attempt was correct vs ground truth (distinct from overall gate_accuracy).
        # knowledge_state_consistency: of attempted chain gates whose dependency was also
        #   attempted, fraction whose submitted answer is derivable from the model's OWN prior
        #   submitted answer — i.e. did it *execute the program*? Independent of ground-truth
        #   correctness, so a faithfully-propagated wrong seed scores high here, low on accuracy.
        gates_by_id = {}
        for node in self.deg.nodes.values():
            for p in node.paths:
                if p.gate and p.gate.gate_id:
                    gates_by_id[p.gate.gate_id] = p.gate
        first_submitted: dict[str, str] = {}
        first_correct: dict[str, bool] = {}
        for ga in self.gate_attempts:
            if ga.gate_id and ga.gate_id not in first_submitted:
                first_submitted[ga.gate_id] = ga.answer_given
                first_correct[ga.gate_id] = ga.correct
        chain_gate_ids = [gid for gid, gate in gates_by_id.items() if gate.depends_on]
        attempted_chain = [gid for gid in chain_gate_ids if gid in first_correct]
        chain_accuracy = (
            sum(1 for gid in attempted_chain if first_correct[gid]) / len(attempted_chain)
            if attempted_chain else None
        )
        consistent = assessable = 0
        for gid in attempted_chain:
            gate = gates_by_id[gid]
            # "prev" = the answer(s) the model actually advanced on, per dependency: gate_results
            # holds it when the dep was passed correctly; otherwise (opaque/mis-seeded case where the
            # model proceeded on its own wrong answer) fall back to its first submission. Multi-dep
            # synthesis gates (e.g. S = A3 + B2) need ALL deps present.
            prev_map = {}
            missing = False
            for d in gate.dep_ids:
                v = self.gate_results.get(d)
                if v is None:
                    v = first_submitted.get(d)
                if v is None:
                    missing = True
                    break
                prev_map[d] = v
            if missing:
                continue  # a dependency never attempted — can't assess consistency
            prev_arg = prev_map if len(gate.dep_ids) > 1 else prev_map[gate.dep_ids[0]]
            expected_from_own = gate.answer_given_prev(prev_arg)
            if expected_from_own is None:
                continue  # not computable (e.g. non-numeric submission)
            assessable += 1
            if score_gate(first_submitted[gid], expected_from_own):
                consistent += 1
        knowledge_state_consistency = (consistent / assessable) if assessable else None

        elapsed = (self.ended_at or time.monotonic()) - self.started_at

        failure_reason = (
            "exit" if self.found_exit
            else "out_of_lives" if self.out_of_lives
            else "budget_exhausted" if self.budget_exhausted
            else "dead_end_trapped" if self.dead_end_trapped
            else "loop_trapped" if self.loop_trapped
            else "impossible" if self.impossible
            else None
        )

        # Ramp depth: how far the model climbed = distinct gates passed (correct commits with a gate_id).
        ramp_depth = len(self.gate_results)

        return {
            "session_id": self.session_id,
            "deg_id": self.deg.id,
            "found_exit": self.found_exit,
            "failure_reason": failure_reason,
            "steps_to_exit": self.steps_used if self.found_exit else None,
            "ramp_depth": ramp_depth,
            "lives_used": self.wrong_count,
            "max_wrong": self.max_wrong,
            "out_of_lives": self.out_of_lives,
            "budget_exhausted": self.budget_exhausted,
            "dead_end_trapped": self.dead_end_trapped,
            "loop_trapped": self.loop_trapped,
            "impossible": self.impossible,
            "step_budget": self.deg.step_budget,
            "optimal_commits": self.deg.optimal_commits,
            "normalized_efficiency": (
                self.deg.optimal_commits / self.steps_used
                if self.found_exit and self.steps_used > 0
                else None
            ),
            "gate_accuracy": gate_accuracy,
            "total_gates_encountered": total_gates,
            "correct_gates": correct_gates,
            "path_correctness": path_correctness,
            "recovery_rate": recovery_rate,
            "wrong_gate_count": len(wrong_gates),
            "recovery_count": recovery_count,
            "chain_gate_count": len(chain_gate_ids),
            "chain_gates_attempted": len(attempted_chain),
            "chain_accuracy": chain_accuracy,
            "knowledge_state_consistency": knowledge_state_consistency,
            "consistency_assessable": assessable,
            "note_used": bool(self.note),
            "pull_count": self.pull_count,
            "elapsed_seconds": round(elapsed, 2),
            "events": [
                {
                    "action": e.action,
                    "node_id": e.node_id,
                    "steps_used": e.steps_used,
                    "outcome": e.outcome,
                    "gate_correct": e.gate_attempt.correct if e.gate_attempt else None,
                }
                for e in self.events
            ],
        }


def new_session(deg: DEG) -> Session:
    s = Session(
        session_id=str(uuid.uuid4()),
        deg=deg,
        current_node_id=deg.start_node_id,
        fog_radius=deg.fog_radius,
        max_wrong=deg.max_wrong,
    )
    s.visited.add(deg.start_node_id)
    # All variable names the DEG declares via sets_var — the hybrid stub's source, complete
    # from turn 0 (the ledger itself only holds variables set so far).
    s.tracked_vars = sorted({
        p.gate.sets_var
        for n in deg.nodes.values()
        for p in n.paths
        if p.gate and p.gate.sets_var
    })
    return s
