"""ContextPolicy — pluggable per-arm context management for the LB CLI harness.

Design of record (private lab notebook):
every experiment arm becomes a NAMED POLICY selected by --context-policy, not a fork of
run_eval.py's turn loop. Three hooks per policy:

  seed(engine_text)                         bootstrap, called once before turn 1
  turn_start(snap) -> list[dict]            what enters context THIS turn (the messages sent to the LLM)
  turn_end(snap) -> None                    what survives into next turn (policy owns its own state)
  telemetry(snap, call_messages) -> ...     the per-turn mechanism signal (extends wali/orchestrator.py's
                                             _hud_event pattern to the LB harness)
  task_end() -> dict                        final per-task summary folded into score_data

`wipe-curated` and `accumulate` wrap run_eval.py's pre-existing --overlay-only and default
(no-flags) code paths byte-for-byte — see run_session's context_policy branch, guarded mutually
exclusive with --overlay-only/--stateless/--inject-history/--kos-prompt so the legacy flag
matrix (every already-published historical arm) is untouched. Three more arms are built on top
of those two: `wipe-curated+actions` (wipe plus an action->outcome ledger), `accumulate+ledger`
and `drop-old-engine` (the marginal-context-value rung-ii arms — additive rescue and the
engine-channel falsifier). Everything else here is a STUB: selectable via --context-policy
(present in POLICIES, so config can NAME a future arm without a harness fork) but refuses
construction with NotImplementedError until its owning chunk lands.

The curated overlay's CONTENT (map/recall/state) stays engine-side (Session._overlay_block in
engine/runner.py, driven by --show-recall/--show-state/--fog-radius) — orthogonal to this module,
which owns only the harness-side message-list axis (what turn_start sends, what turn_end keeps).

Smoke (no network, no GPU — pure message-construction + provenance-gate checks):

    python3 cli/context_policy.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Optional


@dataclass
class TurnSnapshot:
    """Everything a ContextPolicy hook needs for one turn. Not every field is populated at every
    call site: turn_start/telemetry only need `engine_text` (this turn's observation, already
    resolved by the time the harness builds it); turn_end additionally carries `model_text` (this
    turn's reply) — both are passed together in one turn_end call, since nothing reads a policy's
    internal state between the assistant-append and the next-observation-append within a turn."""
    turn: int
    sys_prompt: str
    engine_text: str
    model_text: str = ""
    action: Optional[dict] = None
    gate_id: Optional[str] = None   # the engine's gate id for THIS turn's gate commit (run_eval relays
                                    # /act's `gate_id`; None for observes, moves, and older engines)


@dataclass
class ContextTelemetry:
    """Per-turn mechanism signal. injected_chars is broken down BY SOURCE (overlay / history /
    facts / scratchpad) so a mixed policy (e.g. wipe-curated+actions) shows where its extra chars
    come from, not just a total — the Boundary-II lesson: per-turn telemetry was the only true
    loss there, so it must round-trip into the results JSONL, not live only in stdout."""
    turn: int
    policy: str
    injected_chars: dict
    context_size_at_commit: int
    external_reads: int = 0
    external_read_chars: int = 0
    wipe_event: bool = False

    def to_event(self) -> str:
        """SSE-shaped stdout line, same envelope as wali/orchestrator.py's _hud_event — makes the
        wipe (or its absence) verifiable in a live log, not just reconstructable after the fact."""
        return "data: " + json.dumps({"type": "lb_context_event", **asdict(self)}) + "\n\n"


class ContextPolicy:
    """Base class. Subclasses either implement the three hooks (real policies) or inherit
    _StubPolicy's refuse-on-construction behavior (not-yet-built policies)."""

    name: ClassVar[str] = "base"
    generality_class: ClassVar[str] = "task-general"  # or "deg-aware" — leaderboard integrity field (Will, 2026-07-14)
    needs_observe_refresh: ClassVar[bool] = False      # True: a state-changing commit needs a fresh
                                                        # observe() before the next cold turn_start (wipe-style
                                                        # policies only — run_eval.py's shared refresh block)

    def __init__(self, sys_prompt: str):
        self.sys_prompt = sys_prompt

    def seed(self, engine_text: str) -> None:
        """Bootstrap hook, called once with the pre-loop observe(). No-op by default — wipe-style
        policies rebuild from scratch every turn and need no seed."""
        pass

    def turn_start(self, snap: TurnSnapshot) -> list:
        raise NotImplementedError

    def turn_end(self, snap: TurnSnapshot) -> None:
        raise NotImplementedError

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        raise NotImplementedError

    def task_end(self) -> dict:
        return {}


class WipeCuratedPolicy(ContextPolicy):
    """Current champion (E1a 'wiped' arm / run_eval's --overlay-only): context is wiped every
    turn — [system, user=curated overlay] IS the entire context, no accumulation. Trivial by
    construction: always exactly two messages, so turn_start/turn_end can't drift from the
    legacy branch they replace."""

    name = "wipe-curated"
    generality_class = "task-general"
    needs_observe_refresh = True

    def __init__(self, sys_prompt: str):
        super().__init__(sys_prompt)
        self._wipes = 0

    def turn_start(self, snap: TurnSnapshot) -> list:
        self._wipes += 1
        return [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": snap.engine_text},
        ]

    def turn_end(self, snap: TurnSnapshot) -> None:
        pass  # nothing survives the wipe — that IS the policy

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        return ContextTelemetry(
            turn=snap.turn,
            policy=self.name,
            injected_chars={"overlay": len(snap.engine_text), "history": 0, "facts": 0, "scratchpad": 0},
            context_size_at_commit=sum(len(m["content"]) for m in call_messages),
            wipe_event=True,
        )

    def task_end(self) -> dict:
        return {"wipe_events": self._wipes}


class WipeCuratedNoRefreshPolicy(WipeCuratedPolicy):
    """wipe-curated MINUS the harness's free observe-refresh (marginal-context-value chunk-04 arm
    `s6_wipe_no_refresh`: the falsifier of "the wipe is what wins", taken from the wipe side).

    Identical to `wipe-curated` in every respect but `needs_observe_refresh`, so run_eval.py's
    shared refresh block does not fire and the two messages sent each turn are
    [system, THIS turn's RAW /act response]. After a commit that response is bare: the CORRECT
    path renders no overlay at all (engine/runner.py — outcome, verdict, location, room, steps)
    and neither path renders a `Paths:` listing, so no gate problem reaches the model and it must
    spend its own turn on `observe` exactly as the accumulate family does. The arm is therefore
    "wipe-to-nothing" on the turns that matter, and is named for what it removes rather than for
    what the overlay would have carried.

    Coherent, not degenerate-by-construction: observe costs no step (Session.observe never
    increments steps_used), nav-3 budgets 60 steps against 20 gates with the life budget as the
    real constraint, and the harness turn cap is 3x the step budget — a two-turns-per-gate climb
    to depth 20 fits with headroom, and --show-recall restores the recall block on every elective
    observe. Task-general: it removes a mechanism, and a subtraction cannot add DEG knowledge.
    """

    name = "wipe-curated-norefresh"
    generality_class = "task-general"
    needs_observe_refresh = False


class AccumulatePolicy(ContextPolicy):
    """Control arm (run_eval's default, no flags): the full conversation accumulates turn over
    turn, nothing ever wiped. Owns its own running message list — turn_end appends (assistant
    reply, next observation) together, exactly the two messages.append() calls the pre-refactor
    loop made at two separate points in the turn (nothing reads the list in between, so the
    combined call is behavior-identical)."""

    name = "accumulate"
    generality_class = "task-general"
    needs_observe_refresh = False

    def __init__(self, sys_prompt: str):
        super().__init__(sys_prompt)
        self._messages: list = [{"role": "system", "content": sys_prompt}]

    def seed(self, engine_text: str) -> None:
        self._messages.append({"role": "user", "content": engine_text})

    def turn_start(self, snap: TurnSnapshot) -> list:
        return list(self._messages)

    def turn_end(self, snap: TurnSnapshot) -> None:
        self._messages.append({"role": "assistant", "content": snap.model_text})
        self._messages.append({"role": "user", "content": snap.engine_text})

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        # injected_chars is the MARGINAL contribution (last turn's observation, already the
        # newest message in call_messages) — context_size_at_commit is the running total.
        return ContextTelemetry(
            turn=snap.turn,
            policy=self.name,
            injected_chars={"overlay": 0, "history": len(snap.engine_text), "facts": 0, "scratchpad": 0},
            context_size_at_commit=sum(len(m["content"]) for m in call_messages),
            wipe_event=False,
        )

    def task_end(self) -> dict:
        return {"wipe_events": 0, "final_context_chars": sum(len(m["content"]) for m in self._messages)}


_ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_action_text(model_text: str) -> Optional[dict]:
    """Best-effort JSON action out of a model reply — the policy-side twin of run_eval's parser,
    used only when the harness did not hand the parsed action over on the snapshot."""
    m = _ACTION_JSON_RE.search(model_text or "")
    if not m:
        return None
    try:
        a = json.loads(m.group(0))
    except Exception:
        return None
    return a if isinstance(a, dict) else None


# One path line of an observation's "Paths:" listing, e.g. `  forward: gate  [GATE c1a: 3 + 4]`.
_GATE_LABEL_RE = re.compile(r"^\s*(?P<path>\S+):\s+gate\s+\[GATE\s+(?P<gate>[^\s:\]]+)", re.MULTILINE)


class AccumulatePlusLedgerPolicy(AccumulatePolicy):
    """accumulate PLUS a one-line recall ledger of engine-confirmed solves prepended to the
    current observation (marginal-context-value rung-ii arm `s4_ledger_plus_full`: the
    additive-rescue arm — nothing removed, the concentrate added).

    The accumulating history is untouched; the only change is one line on top of the turn's
    observation:

        recall = c1a=7 | c1b=13 | ...        (or `recall = (none yet)` before the first solve)

    Ledger-provenance rule: an entry is recorded only when the model's own commit was followed
    by the engine's `Gate answer: CORRECT` — anchored on the engine's verdict, never on the
    model's self-report. The gate label is the engine's own gate id for the commit when the
    harness relays it (`TurnSnapshot.gate_id`, from `/act`'s response — the same public label the
    room listing shows and the same key `--show-recall` renders); otherwise it is read off the
    observation the model was shown (`[GATE <id>: ...]` on the committed path's line), and only
    when neither exists does it fall back to an ordinal (`#n`). Task-general: no answer-key
    content enters the ledger, only the engine's verdict and the gate's public name.
    """

    name = "accumulate+ledger"
    generality_class = "task-general"
    needs_observe_refresh = False
    LEDGER_EMPTY = "recall = (none yet)"

    def __init__(self, sys_prompt: str):
        super().__init__(sys_prompt)
        self._ledger: list = []   # (gate_label, answer) — confirmed solves in commit order

    def ledger_line(self) -> str:
        if not self._ledger:
            return self.LEDGER_EMPTY
        return "recall = " + " | ".join(f"{g}={a}" for g, a in self._ledger)

    def turn_start(self, snap: TurnSnapshot) -> list:
        msgs = list(self._messages)
        # Prepend to the CURRENT observation only (the newest user message). The stored history
        # never carries a ledger line, so every earlier turn is byte-identical to plain accumulate.
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1] = {"role": "user", "content": self.ledger_line() + "\n\n" + msgs[-1]["content"]}
        return msgs

    def _gate_label_for(self, path_id: str) -> str:
        """Gate id shown on `path_id`'s line of the newest observation listing the model saw. A
        label already in the ledger belongs to an older listing and is never reused; the search
        stops at the newest listing rather than walking back into solved rooms."""
        taken = {g for g, _ in self._ledger}
        for m in reversed(self._messages):
            if m["role"] != "user":
                continue
            for hit in _GATE_LABEL_RE.finditer(m["content"]):
                if hit.group("path") == path_id and hit.group("gate") not in taken:
                    return hit.group("gate")
            if "[GATE " in m["content"]:
                break
        return f"#{len(self._ledger) + 1}"

    def turn_end(self, snap: TurnSnapshot) -> None:
        # The engine's CORRECT verdict is itself proof a commit was dispatched (the harness may
        # have normalised a synonym like "move" into it), so gate on the verdict + an answer
        # field rather than on the literal action name.
        a = snap.action or _parse_action_text(snap.model_text)
        if a and "answer" in a and "Gate answer: CORRECT" in (snap.engine_text or ""):
            # Label precedence: engine-relayed gate id → id on the committed path's line of the
            # newest listing the model saw → ordinal. The engine id closes the blind-commit hole
            # (MCV rung ii, 2026-09-02: a model that committed at n1 without ever observing it had
            # never been shown "c1b", so the listing search could only yield `#2=12` — a label a
            # later "add N to your c1b answer" cannot be mapped onto; 3 of 6 trajectories).
            label = snap.gate_id or self._gate_label_for(str(a.get("path_id", "")))
            self._ledger.append((label, str(a.get("answer", ""))))
        super().turn_end(snap)

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        return ContextTelemetry(
            turn=snap.turn,
            policy=self.name,
            injected_chars={"overlay": 0, "history": len(snap.engine_text),
                            "facts": len(self.ledger_line()), "scratchpad": 0},
            context_size_at_commit=sum(len(m["content"]) for m in call_messages),
            wipe_event=False,
        )

    def task_end(self) -> dict:
        d = super().task_end()
        d["ledger_entries"] = len(self._ledger)
        d["ledger_final"] = self.ledger_line()
        return d


class AccumulatePlusRefreshPolicy(AccumulatePolicy):
    """accumulate PLUS the free observe-refresh the wipe arm gets (marginal-context-value chunk-04
    arm `s5_accumulate_plus_refresh`: the falsifier of "the wipe is what wins", taken from the
    accumulate side).

    Declaring `needs_observe_refresh = True` is NECESSARY BUT NOT SUFFICIENT, and that is the whole
    reason this class exists. run_eval.py's refresh block only reassigns `current_engine_text`,
    which reaches a policy as the NEXT turn's TurnSnapshot.engine_text — and AccumulatePolicy's
    turn_start returns its own running list and never reads that field. A bare flag flip would fire
    a free engine observe whose text is discarded: an arm behaviourally identical to `accumulate`
    plus a wasted HTTP call, yielding valid rows for an invalid cell. This class closes that hole by
    CONSUMING the refresh in turn_start. `_smoke()` carries a regression asserting plain accumulate
    still ignores the snapshot, so the reason this class exists cannot silently move.

    Shape: the refreshed observation is APPENDED to the newest user message, never substituted for
    it —

        <this turn's raw /act response>      the CORRECT/WRONG verdict, kept
        <blank line>
        <the refreshed observation>          map, recall if enabled, Paths, gate problem

    Append, not replace, is what makes this a ONE-VARIABLE edit against the `accumulate` control:
    replacing would also delete the gate verdict, which is exactly the defect wipe-curated+actions
    exists to patch (the wipe arm never sees its own WRONG), and would bundle two changes into the
    arm built to unbundle two changes.

    Like accumulate+ledger's recall line, the fold rides the SENT copy only: self._messages never
    stores a refresh block, so every earlier turn stays byte-identical to plain accumulate and the
    arm cannot drift into a context-volume manipulation. Task-general: the refresh is "ask the
    engine for the current view" — no DEG structure, no answer-key content. It IS a budget subsidy,
    which is why refresh_folds/refresh_chars_total make it a counted column rather than an
    assumption.
    """

    name = "accumulate+refresh"
    generality_class = "task-general"
    needs_observe_refresh = True

    def __init__(self, sys_prompt: str):
        super().__init__(sys_prompt)
        self._fold_chars = 0        # chars folded onto the newest user message THIS turn
        self._folds = 0
        self._fold_chars_total = 0

    def turn_start(self, snap: TurnSnapshot) -> list:
        msgs = list(self._messages)
        fresh = snap.engine_text or ""
        self._fold_chars = 0
        # Text inequality is the exact — and only available — signal that the harness refreshed:
        # run_eval never populates TurnSnapshot.action, so the dispatched action cannot be keyed
        # off here (the same reason the ledger policy falls back to _parse_action_text). It is
        # sound because all three non-refresh cases already compare equal: on turn 1 and after any
        # elective observe/pull, snap.engine_text IS the newest stored message, and a failed
        # refresh POST leaves current_engine_text unchanged.
        if msgs and msgs[-1]["role"] == "user" and fresh and fresh != msgs[-1]["content"]:
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + fresh}
            self._fold_chars = len(fresh)
            self._folds += 1
            self._fold_chars_total += self._fold_chars
        return msgs

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        # Exact partition, never double-counted: this turn's marginal engine text arrived EITHER as
        # the free refresh (overlay) or as the model's own elected observation (history).
        return ContextTelemetry(
            turn=snap.turn,
            policy=self.name,
            injected_chars={"overlay": self._fold_chars,
                            "history": 0 if self._fold_chars else len(snap.engine_text),
                            "facts": 0, "scratchpad": 0},
            context_size_at_commit=sum(len(m["content"]) for m in call_messages),
            wipe_event=False,
        )

    def task_end(self) -> dict:
        d = super().task_end()
        d["refresh_folds"] = self._folds
        d["refresh_chars_total"] = self._fold_chars_total
        return d


class DropOldEnginePolicy(AccumulatePolicy):
    """accumulate MINUS every engine turn but the latest (marginal-context-value rung-ii arm
    `s2_drop_old_engine`: the pre-registered falsifier of "the engine channel carries the memory").

    The running history is kept in full; what is SENT each turn is
    [system, every assistant reply so far, the current observation] — the model's own actions
    survive, the engine's earlier observations and verdicts do not. If observation repetition
    were the poison this arm would beat plain accumulate; the h=1 screen said the opposite
    (solved-gate confirmations live in engine text). Task-general: it reads only message roles.
    """

    name = "drop-old-engine"
    generality_class = "task-general"
    needs_observe_refresh = False

    def __init__(self, sys_prompt: str):
        super().__init__(sys_prompt)
        self._dropped_chars = 0     # engine chars withheld on the most recent turn_start
        self._dropped_total = 0

    def turn_start(self, snap: TurnSnapshot) -> list:
        msgs = self._messages
        if len(msgs) <= 2:
            self._dropped_chars = 0
            return list(msgs)
        middle = msgs[1:-1]
        self._dropped_chars = sum(len(m["content"]) for m in middle if m["role"] == "user")
        self._dropped_total += self._dropped_chars
        return [msgs[0]] + [m for m in middle if m["role"] == "assistant"] + [msgs[-1]]

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        return ContextTelemetry(
            turn=snap.turn,
            policy=self.name,
            injected_chars={"overlay": 0, "history": len(snap.engine_text), "facts": 0, "scratchpad": 0},
            context_size_at_commit=sum(len(m["content"]) for m in call_messages),
            wipe_event=False,
        )

    def task_end(self) -> dict:
        d = super().task_end()
        d["dropped_engine_chars_total"] = self._dropped_total
        return d


class _StubPolicy(ContextPolicy):
    """Not-yet-built policy. Present in POLICIES (so --context-policy can NAME it — config, not a
    harness fork) but refuses construction with NotImplementedError pointing at the chunk that
    owns the real implementation."""

    owning_chunk: ClassVar[str] = ""

    def __init__(self, sys_prompt: str):
        raise NotImplementedError(
            f"context policy {self.name!r} is a stub (lb-post-release chunk 02) — "
            f"implementation lands in {self.owning_chunk}."
        )


class WipeEveryKPolicy(_StubPolicy):
    name = "wipe-every-k"
    generality_class = "task-general"
    owning_chunk = "01-axes-and-prereg.md / 06-cells-and-paper2.md (cadence cell: wipe every k turns, not every turn)"


class CompactPolicy(_StubPolicy):
    name = "compact"
    generality_class = "task-general"
    owning_chunk = "05-compactor-baseline.md (honest traditional-compaction arm)"


class ScratchpadPolicy(_StubPolicy):
    name = "scratchpad"
    generality_class = "task-general"
    owning_chunk = "04-scratchpad-and-log.md (external scratchpad + append-only log)"


class WipeCuratedPlusActionsPolicy(ContextPolicy):
    """wipe-curated PLUS an action->outcome ledger that survives the wipe (E1a turnlog-pass
    follow-up: the perseveration reversal — every life burned re-submitting one number at the
    same gate).

    The curated overlay (map + recall of correct commits) carries no memory of FAILED attempts:
    after a wrong commit the harness's refresh-observe replaces the LOCKED message, so the next
    cold turn sees the same room and recomputes the same wrong answer. This policy appends a
    compact, task-general ledger of the run's commits and their outcomes under the overlay — the
    action->outcome feedback, nothing else (no task structure, no hints).
    """

    name = "wipe-curated+actions"
    generality_class = "task-general"
    needs_observe_refresh = True
    LEDGER_HEADER = "[ACTION LOG — your commits this run and what the gate did]"
    MAX_ENTRIES = 40   # ledger stays bounded; a 20-gate run has ~20-30 commits

    def __init__(self, sys_prompt: str):
        super().__init__(sys_prompt)
        self._wipes = 0
        self._ledger: list = []      # rendered lines, oldest first
        self._wrong = 0

    def _ledger_block(self) -> str:
        if not self._ledger:
            return ""
        return self.LEDGER_HEADER + "\n" + "\n".join(self._ledger[-self.MAX_ENTRIES:])

    def turn_start(self, snap: TurnSnapshot) -> list:
        self._wipes += 1
        block = self._ledger_block()
        user = snap.engine_text + ("\n\n" + block if block else "")
        return [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _outcome(engine_text: str) -> str:
        t = engine_text or ""
        if "CORRECT" in t:
            return "CORRECT — gate opened"
        if "WRONG" in t or "LOCKED" in t:
            return "WRONG — gate stayed locked"
        first = t.strip().splitlines()[0] if t.strip() else "(no response)"
        return first[:80]

    def turn_end(self, snap: TurnSnapshot) -> None:
        a = snap.action or _parse_action_text(snap.model_text)
        if not a or a.get("action") != "commit":
            return  # only commits carry an outcome worth remembering; observes are re-derivable
        loc = re.search(r"Location:\s*(\S+)", snap.engine_text or "")
        gate = re.search(r"Gate\s+(\d+)", snap.engine_text or "")
        outcome = self._outcome(snap.engine_text)
        if outcome.startswith("WRONG"):
            self._wrong += 1
        where = ""
        if loc:
            where = f" at {loc.group(1)}"
            if gate:
                where += f" (gate {gate.group(1)})"
        self._ledger.append(
            f"  turn {snap.turn}: commit path={a.get('path_id', '')!s} answer={a.get('answer', '')!s}{where} -> {outcome}"
        )

    def telemetry(self, snap: TurnSnapshot, call_messages: list) -> ContextTelemetry:
        block = self._ledger_block()
        return ContextTelemetry(
            turn=snap.turn,
            policy=self.name,
            injected_chars={"overlay": len(snap.engine_text), "history": len(block), "facts": 0, "scratchpad": 0},
            context_size_at_commit=sum(len(m["content"]) for m in call_messages),
            wipe_event=True,
        )

    def task_end(self) -> dict:
        return {"wipe_events": self._wipes, "ledger_entries": len(self._ledger), "ledger_wrong_commits": self._wrong}


class WipeCuratedPlusPointerPolicy(_StubPolicy):
    """Cell 3 target: wipe-curated plus a 'previous gate' pointer label — the off-by-one the
    turnlog pass identified. DEG-aware (encodes nav-3's chain structure), not task-general —
    the leaderboard integrity rule (Will, 2026-07-14) requires this class be declared, not
    inferred, before the arm scores against the leaderboard."""

    name = "wipe-curated+pointer"
    generality_class = "deg-aware"
    owning_chunk = "cell 3 game-the-system demo (index.md priority cell 3)"


POLICIES: dict = {
    cls.name: cls
    for cls in (
        WipeCuratedPolicy,
        WipeCuratedNoRefreshPolicy,      # chunk-04 disambiguation: the wipe-side falsifier
        AccumulatePolicy,
        AccumulatePlusLedgerPolicy,
        AccumulatePlusRefreshPolicy,     # chunk-04 disambiguation: the accumulate-side falsifier
        DropOldEnginePolicy,
        WipeEveryKPolicy,
        CompactPolicy,
        ScratchpadPolicy,
        WipeCuratedPlusActionsPolicy,
        WipeCuratedPlusPointerPolicy,
    )
}


def make_policy(name: str, sys_prompt: str) -> ContextPolicy:
    """Construct a named policy. Unbuilt arms are still selectable (POLICIES contains them) —
    they raise immediately (see _StubPolicy) instead of silently no-op'ing into a bad run."""
    cls = POLICIES.get(name)
    if cls is None:
        raise ValueError(f"unknown context policy {name!r} — choices: {sorted(POLICIES)}")
    return cls(sys_prompt)


def _repo_commit() -> str:
    """Best-effort short git SHA of this checkout — the auto-derived provenance ref for built-in
    policies (they ship in the repo, so the commit already links the exact code)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except Exception:
        return "unknown"


def policy_provenance(name: str, code_ref: Optional[str] = None) -> dict:
    """Leaderboard integrity plumbing (Will, 2026-07-14): a run's policy code must be inspectable,
    not honor-based. A built-in policy auto-derives a repo-commit ref (this file ships in the LB
    repo — the commit already IS the linked code); an unrecognized name (a future custom/
    leaderboard-submitted policy) has no such guarantee and MUST supply an explicit code_ref, or
    this raises — 'no answer-key smuggling' rides on this check, not on trust.

    NOTE (scope): this is the declaration/validation plumbing only. Dynamically LOADING and
    running a custom policy module is not built in chunk 02 — POLICIES is a closed registry of
    the named arms above; a future chunk wires a loader through this same function.
    """
    cls = POLICIES.get(name)
    if cls is not None:
        return {
            "policy": name,
            "generality_class": cls.generality_class,
            "source": "builtin",
            "code_ref": code_ref or f"labyrinthbench/cli/context_policy.py@{_repo_commit()}",
        }
    if not code_ref:
        raise ValueError(
            f"custom context policy {name!r} requires an explicit code_ref "
            f"(leaderboard integrity rule — repo-link the exact policy code)"
        )
    return {"policy": name, "generality_class": None, "source": "custom", "code_ref": code_ref}


# ---------------------------------------------------------------------------
# Smoke — the cheapest thing that can break: pure message-construction +
# provenance-gate checks, no network, no GPU, no engine/session dependency.
# ---------------------------------------------------------------------------
def _smoke() -> int:
    fails: list = []

    # wipe-curated: exactly [system, user] every turn, no growth across turns.
    wc = make_policy("wipe-curated", "SYS")
    for t in (1, 2, 3):
        snap = TurnSnapshot(turn=t, sys_prompt="SYS", engine_text=f"obs{t}")
        msgs = wc.turn_start(snap)
        expected = [{"role": "system", "content": "SYS"}, {"role": "user", "content": f"obs{t}"}]
        if msgs != expected:
            fails.append(f"wipe-curated turn {t}: {msgs} != {expected}")
        wc.turn_end(TurnSnapshot(turn=t, sys_prompt="SYS", engine_text=f"obs{t}", model_text="m"))
        telem = wc.telemetry(snap, msgs)
        if not telem.wipe_event or telem.injected_chars["overlay"] != len(f"obs{t}"):
            fails.append(f"wipe-curated turn {t} telemetry malformed: {telem}")
    if wc.task_end() != {"wipe_events": 3}:
        fails.append(f"wipe-curated task_end: {wc.task_end()}")

    # accumulate: turn_start returns the FULL running list; turn_end grows it by (assistant, user).
    ac = make_policy("accumulate", "SYS")
    ac.seed("obs0")
    expected = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "obs0"}]
    for t in (1, 2):
        msgs = ac.turn_start(TurnSnapshot(turn=t, sys_prompt="SYS", engine_text=""))
        if msgs != expected:
            fails.append(f"accumulate turn {t} turn_start: {msgs} != {expected}")
        ac.turn_end(TurnSnapshot(turn=t, sys_prompt="SYS", engine_text=f"obs{t}", model_text=f"m{t}"))
        expected = expected + [
            {"role": "assistant", "content": f"m{t}"},
            {"role": "user", "content": f"obs{t}"},
        ]
    msgs = ac.turn_start(TurnSnapshot(turn=3, sys_prompt="SYS", engine_text=""))
    if msgs != expected:
        fails.append(f"accumulate final turn_start: {msgs} != {expected}")
    if ac.task_end()["wipe_events"] != 0:
        fails.append(f"accumulate task_end reports a wipe: {ac.task_end()}")

    # wipe-curated+actions: wiped every turn like wipe-curated, PLUS a ledger of commit outcomes
    # that survives the wipe (the one thing the overlay never carried: failed attempts).
    wa = make_policy("wipe-curated+actions", "SYS")
    m1 = wa.turn_start(TurnSnapshot(turn=1, sys_prompt="SYS", engine_text="obs1"))
    if m1 != [{"role": "system", "content": "SYS"}, {"role": "user", "content": "obs1"}]:
        fails.append(f"wipe-curated+actions turn 1 should be bare overlay: {m1}")
    wa.turn_end(TurnSnapshot(turn=1, sys_prompt="SYS", model_text='{"action": "observe"}', engine_text="--- OBSERVE ---"))
    wa.turn_end(TurnSnapshot(turn=2, sys_prompt="SYS", model_text='{"action": "commit", "path_id": "forward", "answer": "59"}',
                             engine_text="--- LOCKED ---\nGate answer: WRONG — the gate does not open.\nLocation: n8\nGate 9.\n"))
    wa.turn_end(TurnSnapshot(turn=3, sys_prompt="SYS", model_text='{"action": "commit", "path_id": "forward", "answer": "61"}',
                             engine_text="--- OK ---\nGate answer: CORRECT\nLocation: n9\nGate 10.\n"))
    m4 = wa.turn_start(TurnSnapshot(turn=4, sys_prompt="SYS", engine_text="obs4"))
    u = m4[1]["content"]
    if len(m4) != 2 or not u.startswith("obs4") or WipeCuratedPlusActionsPolicy.LEDGER_HEADER not in u:
        fails.append(f"wipe-curated+actions turn 4 missing ledger: {m4}")
    if "answer=59 at n8 (gate 9) -> WRONG" not in u or "answer=61 at n9 (gate 10) -> CORRECT" not in u:
        fails.append(f"wipe-curated+actions ledger lines malformed:\n{u}")
    if "turn 1:" in u:
        fails.append("wipe-curated+actions ledger recorded an observe (should be commits only)")
    te = wa.task_end()
    if te != {"wipe_events": 2, "ledger_entries": 2, "ledger_wrong_commits": 1}:
        fails.append(f"wipe-curated+actions task_end: {te}")
    telem = wa.telemetry(TurnSnapshot(turn=4, sys_prompt="SYS", engine_text="obs4"), m4)
    if not telem.wipe_event or telem.injected_chars["history"] <= 0 or telem.injected_chars["overlay"] != 4:
        fails.append(f"wipe-curated+actions telemetry malformed: {telem}")

    # accumulate+ledger (MCV s4): plain accumulate history + ONE ledger line on the current
    # observation; entries appear only after the engine says CORRECT, labelled from what the
    # model saw; the stored history never carries the line.
    obs0 = "--- OBSERVE ---\nLocation: start\nPaths:\n  forward: gate  [GATE c1a: 3 + 4]\n"
    ok1 = "--- OK ---\nGate answer: CORRECT\nLocation: n1\nGate 2.\n"
    obs1 = "--- OBSERVE ---\nLocation: n1\nPaths:\n  forward: gate  [GATE c1b: 2 * 6]\n  back: open\n"
    wrong = "--- LOCKED ---\nGate answer: WRONG — the gate does not open.\nLocation: n1\nGate 2.\n"
    al = make_policy("accumulate+ledger", "SYS")
    al.seed(obs0)
    m = al.turn_start(TurnSnapshot(turn=1, sys_prompt="SYS", engine_text=obs0))
    if len(m) != 2 or m[1]["content"] != AccumulatePlusLedgerPolicy.LEDGER_EMPTY + "\n\n" + obs0:
        fails.append(f"accumulate+ledger turn 1 should carry the empty ledger on the observation: {m}")
    if al._messages[1]["content"] != obs0:
        fails.append("accumulate+ledger wrote the ledger line into its stored history")
    al.turn_end(TurnSnapshot(turn=1, sys_prompt="SYS",
                             model_text='{"action": "commit", "path_id": "forward", "answer": "7"}', engine_text=ok1))
    al.turn_end(TurnSnapshot(turn=2, sys_prompt="SYS", model_text='{"action": "observe"}', engine_text=obs1))
    al.turn_end(TurnSnapshot(turn=3, sys_prompt="SYS",
                             model_text='{"action": "commit", "path_id": "forward", "answer": "13"}', engine_text=wrong))
    m = al.turn_start(TurnSnapshot(turn=4, sys_prompt="SYS", engine_text=wrong))
    if al.ledger_line() != "recall = c1a=7":
        fails.append(f"accumulate+ledger ledger after CORRECT/observe/WRONG: {al.ledger_line()!r}")
    if len(m) != 8 or not m[-1]["content"].startswith("recall = c1a=7\n\n--- LOCKED ---"):
        fails.append(f"accumulate+ledger turn 4 message list malformed: {[x['role'] for x in m]} / {m[-1]['content'][:40]!r}")
    if any("recall =" in x["content"] for x in m[:-1]):
        fails.append("accumulate+ledger leaked a ledger line into an earlier message")
    al.turn_end(TurnSnapshot(turn=4, sys_prompt="SYS",
                             model_text='{"action": "commit", "path_id": "forward", "answer": "12"}',
                             engine_text="--- OK ---\nGate answer: CORRECT\nLocation: n2\nGate 3.\n"))
    al.turn_end(TurnSnapshot(turn=5, sys_prompt="SYS",
                             model_text='{"action": "commit", "path_id": "forward", "answer": "18"}',
                             engine_text="--- OK ---\nGate answer: CORRECT\nLocation: n3\nGate 4.\n"))
    if al.ledger_line() != "recall = c1a=7 | c1b=12 | #3=18":
        fails.append(f"accumulate+ledger labelling (fresh listing / stale listing -> ordinal): {al.ledger_line()!r}")
    # Engine-relayed gate id wins over the listing search — the blind-commit case: the model commits
    # at a room it never observed, so no listing it saw names the gate (MCV rung ii `#2=12`).
    # Precedence chain stays intact: relayed / relayed / absent -> ordinal.
    al2 = make_policy("accumulate+ledger", "SYS")
    al2.seed(obs0)
    al2.turn_end(TurnSnapshot(turn=1, sys_prompt="SYS",
                              model_text='{"action": "commit", "path_id": "forward", "answer": "7"}',
                              engine_text=ok1, gate_id="c1a"))
    al2.turn_end(TurnSnapshot(turn=2, sys_prompt="SYS",
                              model_text='{"action": "commit", "path_id": "forward", "answer": "12"}',
                              engine_text="--- OK ---\nGate answer: CORRECT\nLocation: n2\nGate 3.\n", gate_id="c1b"))
    al2.turn_end(TurnSnapshot(turn=3, sys_prompt="SYS",
                              model_text='{"action": "commit", "path_id": "forward", "answer": "18"}',
                              engine_text="--- OK ---\nGate answer: CORRECT\nLocation: n3\nGate 4.\n"))
    if al2.ledger_line() != "recall = c1a=7 | c1b=12 | #3=18":
        fails.append(f"accumulate+ledger engine gate_id labelling (relayed / relayed blind / absent -> ordinal): {al2.ledger_line()!r}")
    al2.turn_end(TurnSnapshot(turn=4, sys_prompt="SYS", model_text='{"action": "observe"}', engine_text=obs1, gate_id=None))
    al2.turn_end(TurnSnapshot(turn=5, sys_prompt="SYS",
                              model_text='{"action": "commit", "path_id": "forward", "answer": "9"}',
                              engine_text=wrong, gate_id="c1b"))
    if al2.ledger_line() != "recall = c1a=7 | c1b=12 | #3=18":
        fails.append(f"accumulate+ledger recorded a non-CORRECT turn despite a relayed gate_id: {al2.ledger_line()!r}")
    telem = al.telemetry(TurnSnapshot(turn=6, sys_prompt="SYS", engine_text="x"), al.turn_start(TurnSnapshot(turn=6, sys_prompt="SYS", engine_text="x")))
    if telem.wipe_event or telem.injected_chars["facts"] != len(al.ledger_line()):
        fails.append(f"accumulate+ledger telemetry malformed: {telem}")
    te = al.task_end()
    if te.get("wipe_events") != 0 or te.get("ledger_entries") != 3:
        fails.append(f"accumulate+ledger task_end: {te}")

    # accumulate+refresh (MCV chunk 04): plain accumulate history PLUS the harness's free
    # observe-refresh, APPENDED to the newest user message. The trap covered here is the NULL ARM —
    # needs_observe_refresh on its own changes nothing, because AccumulatePolicy.turn_start ignores
    # snap.engine_text, so the FOLD (not the flag) is what makes this cell real.
    ok_bare = "--- OK ---\nGate answer: CORRECT\nLocation: n1\nGate 2.\nSteps: 1 / 60"
    fresh_obs = ("[MAP — fog of war: corridors shown ~2 hops out]\n  n1 (here): forward [gated] -> n2\n\n"
                 "--- OBSERVE ---\nLocation: n1\n\nPaths:\n  forward: gate  [GATE c1b: Add 5 to your c1a answer]\n")
    ar = make_policy("accumulate+refresh", "SYS")
    if not ar.needs_observe_refresh:
        fails.append("accumulate+refresh does not request the harness refresh")
    ar.seed(obs0)
    m = ar.turn_start(TurnSnapshot(turn=1, sys_prompt="SYS", engine_text=obs0))
    if m != [{"role": "system", "content": "SYS"}, {"role": "user", "content": obs0}]:
        fails.append(f"accumulate+refresh turn 1 should be plain accumulate (nothing to refresh): {m}")
    ar.turn_end(TurnSnapshot(turn=1, sys_prompt="SYS",
                             model_text='{"action": "commit", "path_id": "forward", "answer": "7"}',
                             engine_text=ok_bare))
    m = ar.turn_start(TurnSnapshot(turn=2, sys_prompt="SYS", engine_text=fresh_obs))
    u = m[-1]["content"]
    if len(m) != 4 or [x["role"] for x in m] != ["system", "user", "assistant", "user"]:
        fails.append(f"accumulate+refresh turn 2 message shape drifted from accumulate: {[x['role'] for x in m]}")
    if not u.startswith(ok_bare) or "--- OBSERVE ---" not in u or "[GATE c1b" not in u:
        fails.append(f"accumulate+refresh turn 2 did not consume the refresh (NULL ARM): {u[:120]!r}")
    if "Gate answer: CORRECT" not in u:
        fails.append("accumulate+refresh dropped the gate verdict — the fold appends, it never replaces")
    if ar._messages[-1]["content"] != ok_bare:
        fails.append("accumulate+refresh wrote a refresh block into its stored history")
    telem = ar.telemetry(TurnSnapshot(turn=2, sys_prompt="SYS", engine_text=fresh_obs), m)
    if telem.wipe_event or telem.injected_chars["overlay"] != len(fresh_obs) or telem.injected_chars["history"] != 0:
        fails.append(f"accumulate+refresh telemetry malformed on a refreshed turn: {telem}")
    # An ELECTIVE observe is not a refresh: engine_text is already the newest message -> no fold.
    ar.turn_end(TurnSnapshot(turn=2, sys_prompt="SYS", model_text='{"action": "observe"}', engine_text=obs1))
    m = ar.turn_start(TurnSnapshot(turn=3, sys_prompt="SYS", engine_text=obs1))
    if m[-1]["content"] != obs1:
        fails.append(f"accumulate+refresh folded a turn the harness never refreshed: {m[-1]['content'][:80]!r}")
    telem = ar.telemetry(TurnSnapshot(turn=3, sys_prompt="SYS", engine_text=obs1), m)
    if telem.injected_chars["overlay"] != 0 or telem.injected_chars["history"] != len(obs1):
        fails.append(f"accumulate+refresh mis-attributed an elective observe: {telem}")
    te = ar.task_end()
    if te.get("wipe_events") != 0 or te.get("refresh_folds") != 1 or te.get("refresh_chars_total") != len(fresh_obs):
        fails.append(f"accumulate+refresh task_end: {te}")

    # The null-arm premise, asserted directly: plain accumulate handed the SAME refreshed snapshot
    # is UNCHANGED. This is why the flag alone is not an arm — if this check ever fails, the
    # subclass's reason to exist has moved and the arm must be re-derived before it runs again.
    ac2 = make_policy("accumulate", "SYS")
    ac2.seed(obs0)
    ac2.turn_end(TurnSnapshot(turn=1, sys_prompt="SYS", model_text="m1", engine_text=ok_bare))
    if ac2.turn_start(TurnSnapshot(turn=2, sys_prompt="SYS", engine_text=fresh_obs))[-1]["content"] != ok_bare:
        fails.append("accumulate consumed snap.engine_text — the null-arm premise moved")

    # wipe-curated-norefresh (MCV chunk 04): byte-identical to wipe-curated except it declines the
    # harness refresh, so what it sends is [system, THIS turn's RAW /act response] — after a commit,
    # the bare commit result with no Paths listing and no gate problem.
    wn = make_policy("wipe-curated-norefresh", "SYS")
    if wn.needs_observe_refresh:
        fails.append("wipe-curated-norefresh still requests the harness refresh")
    m = wn.turn_start(TurnSnapshot(turn=1, sys_prompt="SYS", engine_text=obs0))
    if m != [{"role": "system", "content": "SYS"}, {"role": "user", "content": obs0}]:
        fails.append(f"wipe-curated-norefresh turn 1: {m}")
    m = wn.turn_start(TurnSnapshot(turn=2, sys_prompt="SYS", engine_text=ok_bare))
    if m != [{"role": "system", "content": "SYS"}, {"role": "user", "content": ok_bare}]:
        fails.append(f"wipe-curated-norefresh turn 2 must send the RAW commit result verbatim: {m}")
    if "Paths:" in m[1]["content"] or "[GATE" in m[1]["content"]:
        fails.append("wipe-curated-norefresh fixture is not a bare commit result — the cell is mis-specified")
    if wn.task_end() != {"wipe_events": 2}:
        fails.append(f"wipe-curated-norefresh task_end: {wn.task_end()}")

    # provenance: both chunk-04 arms declare task-general and auto-derive a builtin code_ref.
    for _nm in ("accumulate+refresh", "wipe-curated-norefresh"):
        _p = policy_provenance(_nm)
        if _p["source"] != "builtin" or _p["generality_class"] != "task-general":
            fails.append(f"{_nm} provenance malformed: {_p}")

    # drop-old-engine (MCV s2): full history kept, but what is SENT is
    # [system, every assistant reply, the CURRENT observation] — older engine turns withheld.
    de = make_policy("drop-old-engine", "SYS")
    de.seed("obs0")
    m = de.turn_start(TurnSnapshot(turn=1, sys_prompt="SYS", engine_text="obs0"))
    if m != [{"role": "system", "content": "SYS"}, {"role": "user", "content": "obs0"}]:
        fails.append(f"drop-old-engine turn 1 should be plain [system, obs0]: {m}")
    de.turn_end(TurnSnapshot(turn=1, sys_prompt="SYS", model_text="m1", engine_text="obs1"))
    de.turn_end(TurnSnapshot(turn=2, sys_prompt="SYS", model_text="m2", engine_text="obs2"))
    m = de.turn_start(TurnSnapshot(turn=3, sys_prompt="SYS", engine_text="obs2"))
    expected = [{"role": "system", "content": "SYS"}, {"role": "assistant", "content": "m1"},
                {"role": "assistant", "content": "m2"}, {"role": "user", "content": "obs2"}]
    if m != expected:
        fails.append(f"drop-old-engine turn 3: {m} != {expected}")
    if len(de._messages) != 6:
        fails.append(f"drop-old-engine did not keep its full history: {len(de._messages)} messages")
    te = de.task_end()
    if te.get("wipe_events") != 0 or te.get("dropped_engine_chars_total") != len("obs0") + len("obs1"):
        fails.append(f"drop-old-engine task_end: {te}")

    # stubs: selectable (present in the registry) but refuse construction.
    for stub_name in (
        "wipe-every-k", "compact", "scratchpad", "wipe-curated+pointer",
    ):
        if stub_name not in POLICIES:
            fails.append(f"stub {stub_name!r} missing from POLICIES registry")
            continue
        try:
            make_policy(stub_name, "SYS")
            fails.append(f"stub {stub_name!r} constructed without raising")
        except NotImplementedError:
            pass

    # provenance: built-in auto-derives a code_ref; unknown name without one is refused.
    prov = policy_provenance("wipe-curated")
    if prov["source"] != "builtin" or not prov["code_ref"]:
        fails.append(f"wipe-curated provenance malformed: {prov}")
    try:
        policy_provenance("some-leaderboard-submission")
        fails.append("policy_provenance did not refuse a code_ref-less unknown policy")
    except ValueError:
        pass
    prov2 = policy_provenance("some-leaderboard-submission", code_ref="https://example/repo@abc123")
    if prov2["source"] != "custom" or prov2["code_ref"] != "https://example/repo@abc123":
        fails.append(f"custom provenance malformed: {prov2}")

    print("=" * 60)
    if fails:
        print("[context_policy smoke] FAIL — first breaks found (this is the point):")
        for f in fails:
            print(f"  ✗ {f}")
        return 1
    print("[context_policy smoke] PASS — wipe-curated/accumulate/wipe-curated+actions/accumulate+ledger/"
          "drop-old-engine/accumulate+refresh/wipe-curated-norefresh message construction verified "
          "(including the null-arm regression), stubs refuse construction, the provenance gate holds.")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
