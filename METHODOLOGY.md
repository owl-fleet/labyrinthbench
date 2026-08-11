# LabyrinthBench Leaderboard — Methodology

LabyrinthBench is a deterministic multi-turn benchmark: a model navigates a maze of **gates** (checkpoints with verifiable answers) toward a single objective, the **exit**. Scoring is mechanical — a run's result is the deepest gate it cleared, re-derivable from its trace by anyone. No LLM sits anywhere in the judging loop.

## 1. Two lanes

The board runs two competition lanes, each pinning one half of the system so the other half is what's being measured.

**Model lane** — the harness is pinned (the standard accumulating-context harness in this repo); models compete. An entry is a model plus its run distribution under that fixed harness.

**Harness lane** — the model is pinned per division (launch division: qwen3:14b at **Q4_K_M**, manifest digest recorded in the entry); context strategies compete. Open harness code is mandatory in this lane: an entry is code, and the board links it. Additional pinned-model divisions open based on demand; a division is a data change, not a redesign.

The quantization pin is evidence-based, not default-by-omission: a pre-registered screening cell (2026-07-21) found Q4_K_M, Q8_0, and F16 indistinguishable on this instrument (median gates 3/3/4, N=3, falsifier threshold Δ≥5 unmet), so the lane pins the community-default tag. Entries carry nullable `quantization`/`digest` provenance fields; digests are recorded as-run from the serving registry — model tags are mutable, so the digest, not the tag, is the datum.

## 2. Integrity ladder

Trust is a property of the screening chain, not the producer. Every entry sits on one rung, and the rung is displayed:

1. **Replay-consistent** — the entry requirement, not a badge: the verification flow re-walks the submitted trace against the instance manifest (deterministic answers, transitions, depth). A trace that fails replay doesn't exist for the board.
2. **Open** — harness code plus container pinned by hash plus model artifact hash public. Mandatory in the harness lane; model-lane entries without it appear greyed as "reported," unranked.
3. **Board-reproduced** — the board reruns top entries from their pinned container on its own fleet (n=3) against sealed instances; a compatible distribution earns the badge. **Declared verification envelope:** the board can re-execute up to ~120B-class Q4 models on current hardware. Beyond-envelope entries are labeled as such and can be confirmed only by community reproduction (or an on-demand rented GPU for high-stakes disputes).
4. **Contested** — reproduction was attempted and failed: the entry is downgraded with the receipts attached (the failed attempts listed on the entry), never deleted. Standing is decided by the public reproduction record.

## 3. Scoring

- **Every dealt instance counts.** No run caps, no self-selected subsets — the deal ledger is the registry, so selection is not restricted, it is undefined. Large-n batches are welcome: more runs mean a tighter estimate, and evidence is a contribution.
- **Cohorts are declared before they run.** An entry carries its planned run count (`declared.planned_n`); the board badges any shortfall — a partial cohort is visible, never silent.
- **Rank = a conservative bound**: the one-sided 95% bootstrap lower confidence bound on median depth (B=10,000, seeded — the computation is deterministic and re-runnable from the entry's data file). The board displays median, n, the bound, and the full run distribution. A lucky small-n entry self-limits: wide interval, low bound, modest rank until more evidence arrives. Compute buys precision, never inflation.
- **Aborts are data**: an abandoned instance scores its depth at last commit into the entry's distribution.
- **Within-run vs between-run**: max depth is the defining metric of a single run; comparison between entries is meaningless without the statistics. The board header says so.
- **Efficiency columns** (turns, pulls — on-demand full-state-ledger requests in pull-enabled arms, costing one step each, lives) are metrics, never gates — exit/depth remains the only objective.
- **Highlights** (deepest single run, cleanest exit) are celebrated per season, explicitly labeled as highlights, never rankings.

## 4. Verification limits

Replay verification proves a trace is consistent with the maze — not that a model produced it.
Board reproduction proves the submitted code reproduces the claimed results — the strongest guarantee replication offers anyone, including science.
We make no cryptographic claim about model provenance; hashes authenticate artifacts (manifests, containers, model files), never processes.
The runner and dealer make honest running the cheapest path; the statistics make luck unrankable; reproduction settles the rest.
Once the dealer is live, selection is structurally impossible: every dealt episode is on the ledger and every ledger entry counts.
Until then (season 0), that guarantee does not exist for manual submissions — replay-checking proves each submitted trace is internally real against its instance, not that unfavorable runs were also submitted.
Board-seed entries publish their full pre-registered cohorts for exactly this reason, and early manual submissions are asked to declare theirs.

## 5. How to submit

Target flow (both lanes): the runner CLI fetches a dealt instance from the dealer, hashes the model file and harness container, streams the per-turn trace as it runs, and auto-opens an entry pull request; the verification flow replays the trace and computes the score into the entry file.

**Current status:** the dealer service and runner CLI ship after the initial release. Until they do, the board carries the seed entries (the board's own runs, marked `board-seed`) and accepts early submissions as manual pull requests carrying the results JSONL plus, for the harness lane, the harness code — replay-checked by the maintainers through the same engine primitives (`bfs_verify`, `score_gate`) the harness itself uses.

## 6. Season cadence

Instances are minted per season from the deterministic generator and sealed by manifest hash (quarterly cadence). The generator's output doubles as the open practice set, marked non-scoring; season instances stay sealed until the season closes. Pre-season (season 0) is the seed-entry period: board-run entries only, establishing the baselines to beat.
