# LabyrinthBench — Claude Code context

Read this before touching anything. It is the whole project brief for an agent session, local or cloud.

## What this repo is

A deterministic multi-turn benchmark for language-model agents. Each run deals a maze instance minted byte-deterministically from a seed (a directed evaluation graph, "DEG"), the model navigates it turn by turn, and the scoring API records every transition. The board has two lanes (`METHODOLOGY.md` §1): the **model lane** pins the harness and ranks models; the **harness lane** pins the model and ranks context strategies, whose code must be open. Two rules shape every decision here:

- **Exit and depth are the only objective.** Turns, pulls, lives and every other efficiency column are metrics, never gates (`METHODOLOGY.md` §3).
- **Trust is a property of the screening chain, not the producer.** An entry exists for the board only if its trace replays against the instance manifest (`METHODOLOGY.md` §2, rung 1). Every number the repo publishes is re-derived by CI from the committed data.

## Layout

| Path | What |
|---|---|
| `engine/` | Deterministic maze engine + instance mint |
| `cli/` | Evaluation harness (`run_eval.py`), context policies (`context_policy.py`), analysis passes, `verify.py`, `validate_degs.py`, `doctor.py`, and the `test_*.py` scripts |
| `api/` | FastAPI scoring server + the live `/watch` view |
| `degs/` | Maze manifests (`*.yaml`) — the map sources every minted instance derives from |
| `entries/` | Board entries (`*.json`) — each score block re-derives from its own run distribution |
| `results/` | Committed traces and the headline cells (`e1a-table1/`, `renderer-cell/`, `null-baseline/`); `results/*.jsonl` at the top level is the gitignored local run ledger |
| `sandbox/`, `sandbox-harness/`, `sandbox-target*/` | External-validation sandbox: real-shell sysadmin tasks driven by `cli/run_sandbox.py` |
| `companion/` | Companion modules (dispatcher, interceptor) that sit between the harness and the model |
| `e2/` | Frozen MMLU seed list with pinned provenance |
| `site/` | Eleventy static site for labyrinthbench.ai: leaderboard, methodology, `/data` annex |
| `docs/` | Explainers and the `annex/` briefs behind published results |
| `scripts/` | `pre-push-scan.sh` (the secret/identifier gate), trace renderers, fixture makers |

## Commands

What CI runs on every PR and every push to `main` (`.github/workflows/validate.yml`):

```bash
pip install pyyaml
python3 cli/validate_degs.py                                       # manifests load, BFS parity, corridor walk
python3 cli/verify.py results/null-baseline/null_random_walk.jsonl  # committed traces replay
python3 cli/verify.py --entry entries/<one>.json [--entry ...]      # every entry's score block re-derives
bash scripts/pre-push-scan.sh                                      # pattern battery + gitleaks (docker)
```

Tests are plain scripts, not a pytest suite: each `cli/test_*.py` prints PASS/FAIL lines and exits 1 on any failure. Run them directly:

```bash
python3 cli/test_classify_failures.py
python3 cli/test_lock_path.py
python3 cli/test_truncation_guard.py
```

Full dependencies for the API and harness: `pip install -r requirements.txt` (plus `numpy` and `matplotlib` for the figure passes in `cli/e1a_table1.py`).

`cli/run_eval.py`, `cli/run_sandbox.py`, `cli/run_oracle.py` and `cli/doctor.py` need a live OpenAI-compatible model server (`--base-url` or `$LB_BASE_URL`). Nothing else does. Site preview: `cd site && bash dev.sh`.

## Rules

- **Branch and PR, never push `main`.** Main is not locked against direct pushes, so this rule is on you. Create a branch, push it, open a PR; CI is the merge gate.
- **Never create or push a tag.** A `v*` tag builds and publishes the public container image (`.github/workflows/publish-image.yml`).
- **Never hand-edit `entries/`, committed traces under `results/`, or `degs/` manifests.** They are replay-verified; an entry that fails replay does not exist for the board, and a manifest edit changes what every instance means. Data changes go through the tools that produce them, with the verification commands above passing.
- **Run `bash scripts/pre-push-scan.sh` before every push**, and never commit anything in its categories: operator identity, email addresses, private-network addresses, host mount paths, keys, connection strings, internal service names, private-notebook paths. This is a public repository. CI runs the same scan server-side; a hit fails the PR.
- **No secrets exist here, by design.** No `.env`, no accounts, no API keys. Don't add any.
- **Don't invent references.** Some docstrings mention a private lab notebook or pre-registration that lives outside this repo. Treat those as out of reach; do not fabricate their contents or add links to them.
- **Keep docs in the author's voice.** README, METHODOLOGY and `docs/` are first-person and written by the author; propose wording in the PR, don't rewrite the voice.

## Conventions

- Commit subject: `<area>: <what changed>`, in plain words; the body says why and what was measured. Look at `git log` for the style.
- New tests follow the existing `cli/test_*.py` shape: a docstring that says what it proves and "no live server", stubbed transport, `sys.exit(1 if FAILS else 0)`.
- One change per PR. Say in the PR body which of the commands above you ran and their results.

## Task briefs

Work handed to a session arrives as a GitHub issue in this repo labelled `cloud` (Goal / Where / Acceptance / Constraints / Out of scope / Report). The issue is the whole brief: there is no other context, and nothing outside the repo is reachable. Read it with `gh issue view <n> --comments` before touching anything.

- Branch `claude/issue-<n>-<short-slug>`; implement exactly the brief's scope.
- Run the brief's acceptance commands, then the CI commands above, then the scan.
- The PR body starts with `Closes #<n>`, lists every command run with its exit status and result, states whether gitleaks ran, and covers the brief's Report section.
- If the brief is ambiguous or an acceptance command cannot pass after a real attempt, still push and open the PR as a draft with a `## Blocked` section naming exactly what is missing. An unpushed result is lost when a hosted VM ends.

`.claude/settings.json` in this repo denies the commands the Rules forbid (pushing main, force-pushes, tags, merges, `git add -A`); it applies to local and hosted sessions alike.

## Cloud sessions

When `CLAUDE_CODE_REMOTE` is `true` (the documented marker is `CLAUDE_CODE_REMOTE_SESSION_ID`, set to the session id) you are on a hosted VM with a fresh clone. Measured on the first session (2026-09-24):

- Python is **3.11**, while CI pins **3.12**. Don't rely on 3.12-only syntax or stdlib additions.
- `gh`, `pip` and PyPI work. There is no model server and no GPU, so scope is code, tests, docs and the site.
- The docker CLI is installed but **dockerd is not running**, so `scripts/pre-push-scan.sh` skips gitleaks and still reports PASS on the pattern battery alone. That is a weaker pass than the local one. Either start the daemon first (`nohup dockerd >/tmp/dockerd.log 2>&1 &`, then wait for `/var/run/docker.sock`) and rerun the scan, or say in the PR body that gitleaks did not run locally. CI's `scan` job runs gitleaks on the PR regardless.

Every rule above applies unchanged.
