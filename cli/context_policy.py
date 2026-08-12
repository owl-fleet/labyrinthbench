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
matrix (every already-published historical arm) is untouched. Everything else here is a STUB:
selectable via --context-policy (present in POLICIES, so config can NAME a future arm without a
harness fork) but refuses construction with NotImplementedError until its owning chunk lands.

The curated overlay's CONTENT (map/recall/state) stays engine-side (Session._overlay_block in
engine/runner.py, driven by --show-recall/--show-state/--fog-radius) — orthogonal to this module,
which owns only the harness-side message-list axis (what turn_start sends, what turn_end keeps).

Smoke (no network, no GPU — pure message-construction + provenance-gate checks):

    python3 cli/context_policy.py
"""
from __future__ import annotations

import json
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


class WipeCuratedPlusActionsPolicy(_StubPolicy):
    """Cell 3 target (E1a turnlog pass, 2026-07-14, e1a_turnlog_pass.py): wipe-curated plus the
    action->outcome ledger — the missing feedback that produced llama3.3:70b's perseveration
    reversal (4 wrong commits, same gate, same answer, every wiped run)."""

    name = "wipe-curated+actions"
    generality_class = "task-general"
    owning_chunk = "cell 3 game-the-system demo (index.md priority cell 3)"


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
        AccumulatePolicy,
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
    the seven named arms above; a future chunk wires a loader through this same function.
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

    # stubs: selectable (present in the registry) but refuse construction.
    for stub_name in (
        "wipe-every-k", "compact", "scratchpad", "wipe-curated+actions", "wipe-curated+pointer",
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
    print("[context_policy smoke] PASS — wipe-curated/accumulate message construction verified, "
          "stubs refuse construction, the provenance gate holds.")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
