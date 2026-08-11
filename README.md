<div align="center">

# LabyrinthBench

**A deterministic multi-turn benchmark for language-model agents — no LLM judge anywhere — with a built-in harness for testing whatever context-management strategy you can imagine.**

[**Leaderboard**](https://labyrinthbench.ai) · [**Data annex**](https://labyrinthbench.ai/data) · [**Board rules**](METHODOLOGY.md) · [**Quickstart**](QUICKSTART.md)

</div>

If you've ever played a text based game, you already have a pretty good idea of how this benchmark works. The model wakes up in a maze it can't see, reads what the current room gives it, types commands, and hunts for the exit. Between it and the exit stand a number of **gates** that are impassable until solved. Gate problems have objective answers that are scored deterministically and range in difficulty from simple arithmetic to complex multistep recall and synthesis. A run's score is the deepest gate it cleared, re-derivable by anyone from the trace — the full recorded log of the run.

That's the whole benchmark: can a model carry what it learned at gate 3 into a decision at gate 19, and act on it when it counts? I went looking for a test that measured that and couldn't find one, so I built it.

Here's what a run looks like, live — a 14B model exploring one of the maze maps, the route drawing itself on the map as the action log fills in below:

![A live run in the watch view: the map above, the action log below](docs/assets/watch.gif)

**Who it's for:** people who run agentic AI models and want to know what one actually does 30 turns into a task.
**What it is:** a deterministically scored multi-turn benchmark.
**When to use it:** before trusting a model in a long-horizon loop; when tuning a context strategy.
**Where it runs:** any OpenAI-compatible endpoint (Ollama, LM Studio, llama.cpp, etc.)
**Why it exists:** it measures what an agent retains — and correctly reuses — as its own context piles up against it, scored without an LLM judge; the built-in harness lets you test the context strategies that are supposed to help.

---

## Why I built it

I think just about everyone who's used AI for any meaningful amount of time has run into this scenario: you make a great plan with the AI and get most of the way through it — then you hit the context limit. You reluctantly hit "compact" or equivalent and keep going, and everything eventually comes off the rails, because the compaction removed something weight-bearing. Compaction has to decide what gets cut and what gets carried forward — and what comes out the other side is effectively a new session. It isn't THAT big of a surprise when the AI starts to do something that is clearly based on a misunderstanding or miscommunication: "It's time for me to come clean. You expressly told me not to do that and I did it anyway". This is an unfortunately common user experience, and personally makes me want to see how far I can throw my keyboard in the aftermath.

I want to know if we can objectively measure what breaks when context gets cut — and whether deterministic tools can stop it breaking. I didn't feel confident that it could be done with any of the current benchmarks, and I have felt pretty strongly that we can and **should** be benchmarking AI models with deterministic tests that can be objectively scored without the involvement of any type of AI judge. So, I built a new benchmark that:

* Is scored objectively and deterministically — no LLM judge anywhere.
* Includes a harness to manage context accumulation across turns.
* Makes sure a single lucky run can't top the leaderboard — rank is a conservative statistical bound, so evidence moves you up, not variance.
* Allows for cross model comparison of performance and failure pathology.

My name is John William Deaver — everyone calls me Will. I don't have a degree in computer science, but I do have a PhD in Exercise Physiology. If there is a way to measure something objectively and deterministically, then why would we ever settle for measuring it subjectively and probabilistically? So I lean into what I know, and that just happens to be objective testing of probabilistic output generators. They used to be mice, rats, and cells. Now I toss AI models into a maze and see which ones can stay on task and recall previously pertinent information until the very end.

The maze on its own isn't novel, but couple the navigation with simple, deterministically scored questions and different context management strategies, and some really interesting things start to shake out.

TL;DR: using an LLM judge to score benchmarks makes my head spin with all of the different flaws and biases that are possible. Ask the right questions in the right way, and you don't need a judge.

## Run it (~10 minutes)

What you need:

* **Docker** (Desktop or engine) — the benchmark is a couple of small containers.
* **Any OpenAI-compatible endpoint** serving a model — the only hardware requirement is whatever your model needs.
* **git — only if you plan to enter the harness lane** with code-linked runs; running the benchmark itself needs none.

Two commands. Point `--base-url` at whatever endpoint you already run (Ollama shown; LM Studio serves on port 1234), and swap `qwen3:14b` for any model tag your server actually has.

```bash
docker compose -f docker-compose.standalone.yml up -d

docker compose -f docker-compose.standalone.yml exec labyrinth-bench \
  python cli/run_eval.py --model qwen3:14b --no-think \
  --base-url http://host.docker.internal:11434/v1
```

Then open **<http://localhost:8090/watch>** and watch your run live, turn by turn — the same view as the GIF up top.

The three things most likely to bite on a first run:

* **Linux:** `host.docker.internal` works — the compose file maps it to your host gateway. If the connection refuses, your endpoint isn't listening on a host interface (bind it, or swap in your machine's LAN IP).
* **Qwen3-family thinking models** want `--no-think` for comparable runs (that's why it's in the example — drop it for non-thinking models). The flag is **Ollama-only** (it uses Ollama's native API); on LM Studio or any other server, run without it — thinking models still work, they just spend turns thinking.
* **Context window:** some models need more than your server's default — 8k+ recommended.

No accounts, no API keys, no `.env` — Docker plus the model server you already run is the whole setup. No git or Python either: [QUICKSTART.md](QUICKSTART.md) walks the whole thing step by step, LM Studio included, with a fuller troubleshooting list.

## How it works

A run is a sequence of turns. The model sees only what the current room shows (fog — a couple of steps ahead), issues a command, and the engine answers. Gates are locks with objective answers: a wrong answer doesn't open the gate — you stay put and it costs one of a small budget of lives (four, on the 20-gate corridor; each map sets its own).

The first map family is a real maze — dead ends, a loop trap, one exit. Here's an actual run through one of them (`alpha-2`), drawn from its trace: a 120B-class model, detours and backtracks included, exit at step 15. (The renderer that drew it was written by qwen3:14b — one of the benchmarked models, working under the wipe policy described below. It draws pictures; it scores nothing. Receipts in `results/renderer-cell/`.)

![A rendered run trace through the alpha-2 maze](site/src/assets/figures/run-trace-alpha2-gptoss120b.svg)

The corridor family removes navigation entirely, so there is no excuse left but memory. `nav-3` is 20 gates in a straight line: gate 1 is simple arithmetic, every later gate builds on the running total, and from gate 8 on each gate also asks for one specific earlier answer — the reach-backs rotate, so the model must be able to recall any prior answer on demand. Trivial with clean notes; brutal from a chat history full of its own noise.

The third family attacks the notes themselves. In `rev-2` (34 gates), eight variables change value as you climb — and change *back*, so a variable may return to a value it held earlier and several variables may share a value. Every gate asks about the **current** value: the most recent time you saw a value is not necessarily current, and the same question can recur with a change in between — recompute, don't reuse. Gate 1 is `"Variable A is initialized to 3. What is the value of A?"`; later gates ask things like `"Add the current value of C to the current value of D."` — same arithmetic, but only if your record of C and D is current rather than merely familiar.

| Family | Shape | What it isolates |
|---|---|---|
| `alpha` | branching maze — dead ends, a loop trap | navigation and recall together, the agent case |
| `nav-3` | corridor, 20 gates | retention: every gate reaches back to a named earlier answer |
| `rev-2` | corridor, 34 gates, values change mid-run | currency under interference: is the note current, or just familiar? |

A map (a **DEG** — deterministic evaluation graph) is structure plus gate content; the mint (`engine/mint.py`) deals *instances* — same structure, different seeded values, byte-deterministic from the seed. The board can deal you an instance no one has seen and still re-derive your entire run from the trace.

These three families are a launch set, not a finished taxonomy — maps and map types will keep evolving, and each season's competition instances are minted fresh and stay sealed until the season closes (season cadence and rules: [METHODOLOGY.md](METHODOLOGY.md)).

## The harness

The other half of the benchmark: every run goes through a **context policy** — a swappable layer that decides what the model sees each turn. Two ship with the repo. The default is what every chat framework does: the full conversation accumulates. The alternative wipes the model's context every turn and re-injects only its own recorded gate answers — the policy behind the headline experiment further down this page. Both are small classes in one file (`cli/context_policy.py`), selected with `--context-policy`; testing your own strategy is subclassing that file, not forking the harness.

Two things make a strategy measurable rather than anecdotal:

* **Per-turn telemetry.** Every character injected into the model's context is counted by source and emitted in the live log — so what a policy actually did is verifiable from the run record, not inferred from its description.
* **Code-linked runs.** Every run records the exact commit of the policy code that produced it (`--policy-code-ref`, auto-derived when you run from a git clone; ZIP users pass it explicitly). The leaderboard has a lane where context strategies compete under a pinned model — the harness lane, rules in [The board](#the-board) — and entries there link to their code; a strategy you can't read doesn't rank.

Enforcement flags stack on top of any policy. `--look-gate` is the shipped example, and it exists because instruction demonstrably wasn't enough:

### Why `--look-gate` is a flag, not a prompt

Information alone kept failing: inject the full authoritative game state every turn and every scored model answers every gate correctly — and the models that couldn't find the exit still can't. Same story at the instruction level: on the 34-gate map, qwen3:14b answers before looking — median depth 4.5, every wrong answer an unobserved guess. A one-line instruction to observe first — with the pass bar locked before it ran, same as every experiment here — moved the median to 5.5, inside noise, guesses intact. A five-line interceptor that converts any answer at an unobserved gate into an observation took the same model to median 21+, zero guesses across all 12 forced runs. That's why the harness ships `--look-gate` as a flag and not as a suggestion in the system prompt. Full brief in the [annex](https://labyrinthbench.ai/data).

## What the first experiment found

It surprised me in both directions. On the 20-gate corridor, I ran 13 local models twice over: once keeping the full chat history, once with context wiped every turn and only the model's own recorded gate answers re-injected — six runs per model per condition. The pass bar — median depth up 5 or more gates — was locked before any wiped run existed.

![Paired control vs wiped runs, per model](results/e1a-table1/e1a_table1_paired_stripplot.png)

**Wiping won in 7 of the 9 models that had room to show a gain — and backfired in the other 2.** (The remaining four — gemma4:31b, gpt-oss:120b, qwen3.5:122b, qwen3.6:27b — already ran at the map's ceiling, a control median of 20, so a 5-gate gain is unreachable by construction. Those four got their own wiped arms in a registered follow-up; that brief lands in the [data annex](https://labyrinthbench.ai/data).) deepseek-r1:70b went from a median of 1 to 20: five of its six control runs cleared exactly one gate; wiped, it exited all six. glm-4.7-flash and qwen3:14b both gained 15.5 gates of median, and four of the seven winners went from exiting 0–33% of their runs to 83–100%. Wiping lifted qwen3:14b to a 20/20 median (83% exits) — the same ceiling the 120B-class models occupy unaided on this map. That sentence is the whole claim: where the re-injected record doesn't carry what a task needs, the same lever inverts.

**The two reversals:** llama3.3:70b fell from a median of 15 to 9; llama4:scout from 10 to 7. Parameter count doesn't predict the direction — the biggest gainer and the biggest loser are both 70B models; what the two reversals share is the highest unaided starting scores of the nine. The traces say why for one of them: the re-injected answers carry no record of what already *failed*, and llama3.3:70b burns all four lives re-submitting the identical wrong answer in every wiped run. With history intact it never does that.

The costs are measured too: models that cleared the bar used 0.11–0.42× the control's turns per gate — but qwen3.5:9b (one of the seven winners) pays for its depth per turn, re-deriving its whole reasoning chain from the notes every single turn: 21× the output tokens, 10.5 → 150 seconds per turn.

Full brief, run tables, the pre-registration with its lock dates visible, and raw run logs: [results/e1a-table1/](results/e1a-table1/) and the [data annex](https://labyrinthbench.ai/data).

### Reproduce it on your rig

Six runs where the model keeps its full chat history, six where the harness wipes it every turn — same model, same map.

```bash
# control: the model keeps its full chat history
docker compose -f docker-compose.standalone.yml exec labyrinth-bench \
  python cli/run_eval.py --model <your-model> --deg nav-3 --runs 6 \
  --base-url http://host.docker.internal:11434/v1 --output /results/control.jsonl

# wiped: context cleared every turn; only the model's own recorded gate answers are re-injected
docker compose -f docker-compose.standalone.yml exec labyrinth-bench \
  python cli/run_eval.py --model <your-model> --deg nav-3 --runs 6 \
  --base-url http://host.docker.internal:11434/v1 --overlay-only --show-recall --output /results/wiped.jsonl
```

(`--overlay-only --show-recall` is the exact flag pair the 13-model comparison above ran, kept so your runs match the published record; it's equivalent to `--context-policy wipe-curated`, which is the interface to reach for in your own experiments.)

Compare median depths. A gain of 5 or more gates is the same bar the comparison above was scored against — set and published before any wiped run existed. Whichever direction your model moves, that's a datum — the leaderboard wants both.

## The board

The leaderboard at [labyrinthbench.ai](https://labyrinthbench.ai) — the board, from here on — runs two lanes, per [METHODOLOGY.md](METHODOLOGY.md):

* **Model lane** — harness pinned, models compete.
* **Harness lane** — model pinned, context strategies compete. At launch the pinned model is qwen3:14b at Q4_K_M, the exact weights recorded by digest so a mutable tag can't drift. Open policy code is mandatory: an entry is code, and the board links it. Want a different pinned model? Adding one is a data change, not a redesign — open an issue.

The pin is Q4_K_M because I checked what the pin costs: Q8_0 landed on the same median, and FP16 bought one gate — inside noise at n=3 — for several times the wall-clock (a field measurement made alongside the pre-registered screening cell, [METHODOLOGY.md §1](METHODOLOGY.md), but not itself part of the published data).

Rank is a conservative bound — the one-sided 95% bootstrap lower confidence bound on median depth — so large-n evidence tightens rank and a lucky small-n entry self-limits. You also get eyeball outlier detection for free: after enough runs, a point that sits far outside an entry's distribution and never reproduces is visibly luck, whether anyone labels it or not — and the bound has already priced it in. Entries never pool: every entry ranks alone on its own recorded runs, so nobody else's runs — however bad — can touch yours. Seed entries ran every model at its shipped default sampling settings; the harness itself passes none. Every dealt instance counts, aborts included (an abandoned run scores at the depth it reached). Efficiency columns (turns taken, lives spent) are metrics, never gates: exit is the only objective. Every entry sits on a displayed rung of the integrity ladder: replay-consistent (the submitted trace re-walks cleanly against the map) → open (policy code and artifact hashes public) → board-reproduced (the board reran it and got a compatible result) → contested (a reproduction failed; the receipts stay attached).

**Getting on it:** automated submission tooling ships after launch. For now, the board accepts entries as pull requests carrying the results file — plus, for the harness lane, your policy code. Submission flow and verification: [METHODOLOGY.md §5](METHODOLOGY.md).

The wiping policy I ship demonstrably doesn't win everywhere — two of nine cohort models got worse under it. I'm looking forward to someone beating my attempt with their own harness. I have my own ideas for a few improvements here and there.

## What's in the repo

| Path | What |
|---|---|
| `engine/` | Deterministic maze engine + instance mint (byte-deterministic from seed) |
| `cli/` | Evaluation harness (`run_eval.py`), analysis passes |
| `api/` | Scoring API + the live `/watch` view |
| `degs/` | Maze manifests |
| `results/e1a-table1/` | The headline cell: tables, figure, raw pass outputs |
| `results/renderer-cell/` | The maze image above, sourced: `scripts/render_trace.py` was written by qwen3:14b — all three attempts, full transcripts, the automated checker |
| `METHODOLOGY.md` | Board rules: lanes, integrity ladder, scoring, verification limits |

## How this was built

I'm the solo human contributor, and nearly every commit in this project's development was AI-assisted — drafted by AI under my direction, reviewed line by line, corrected constantly. I directed the work in excruciating detail; every design decision, every experimental call, and every error is mine. The public repo's history starts at a fresh extraction (the development history stays private for credential-hygiene reasons), so commits from release day onward carry explicit AI co-author trailers. The case where the AI's authorship *is* the point — a local 14B writing this repo's trace renderer under a frozen 11-check gate — is documented with full transcripts in `results/renderer-cell/`. Fittingly, almost every failure along the way was a context-management failure — which is what this benchmark measures.

## FAQ

**My first run immediately shows DNF with barely any steps.**
The model was never reached — a connection failure or an unknown model tag records a DNF row rather than crashing. Check that your server is running, the port in `--base-url` matches, and the tag exists on your server (`ollama list`); on Linux, see the networking note under Run it. Or catch all of it before spending a token: `python3 cli/doctor.py --model <tag> --base-url <url>` checks the endpoint, the tag, and the context-size trap in one shot.

**I hit Ctrl-C but it says a session is still running.**
The run keeps going server-side, and a lock blocks a new run until it finishes. Watch it wind down at `/watch`, or wait it out. Either way it ends up scored — an abandoned run counts at the depth it reached.

**My model degrades or dies late in long runs.**
Runs that keep full chat history grow the prompt every turn (the wiped policy stays flat). Give the model 8k+ of context — and set it **server-side** (Modelfile, app settings): Ollama's OpenAI-compatible endpoint silently ignores a per-request context-size option, so a client flag can't be trusted.

**Can I set temperature / top_k?**
The harness passes no sampling parameters — your server's settings apply, whatever they are. Seed entries ran every model at its shipped defaults. If you tune settings server-side, declare it in your entry: replay verification checks the trace, not your sampler.

**Do I need git?**
Not to run anything — the ZIP download works. A harness-lane board entry links its policy code, and `--policy-code-ref` fills itself in from a git checkout; from a ZIP, pass it explicitly.

**Can I run a 4B model? A 200B? A cloud model?**
Any size, against any endpoint that speaks the OpenAI chat API without an auth key — local servers, LAN boxes, proxies. Key-authed cloud APIs need a small proxy in front (LiteLLM-class) for now. Board divisions for other pinned models open on demand.

**Could someone flood bad runs to drag a model down the board?**
No — entries never pool. Every entry ranks alone on its own recorded runs, so a bad-faith submission can only create its own low, attributed row, wearing its integrity rung. It can't touch anyone else's.

**Why should I believe any number on the board?**
Don't. Re-derive it. `python3 cli/verify.py <results.jsonl>` re-walks any trace against the map: every transition checked as a legal edge, every score field recomputed, and where the full turn log is present, the whole session re-driven through the engine until the regenerated event stream matches the submitted one exactly. `python3 cli/verify.py --entry <entry.json>` recomputes an entry's median and bootstrap bound from its own run distribution. The seed is recorded in the file. CI runs both on every pull request. What this does *not* prove: that the claimed model produced the trace, or that unfavorable runs were also submitted. That part falls to the artifact hashes and board-reproduction rungs ([METHODOLOGY.md §4](METHODOLOGY.md)).

**Could a model just guess its way through the gates?**
Measured, not assumed: the board carries a null control. 200 seeded random-walk runs (random path, random answer, no model) reached depth 0 in all 200. It sits on the board unranked, and its trace file is replay-verified in CI like any other entry. Regenerate it yourself with `python3 cli/run_null_baseline.py`.

License: MIT.
