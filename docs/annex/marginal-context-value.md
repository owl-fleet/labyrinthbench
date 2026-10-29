# What in an agent's context is load-bearing? Wipe, refresh and recall on a 20-gate maze

*DRAFT 1 for the author's red pen — not yet published. Substrate: nav-3 (the 20-gate recall/synthesis ramp; manifest public in `degs/`). Each experiment's design and falsifiers were written down before its first row was read. Model names literal. Every number is a point estimate at the n stated beside it.*

## Abstract

An agent that keeps its whole history in context usually gets worse as the history grows. I wanted to know which parts of that history carry the weight, so I took the context apart by channel instead of by length, on a maze where depth reached is an exact score. Subject: qwen3:14b on nav-3, where the full-history control dies at a median of gate 3 of 20.

**Subtraction worked where addition never did.** Wiping the context to the latest engine message, with a fresh view of the maze after every action and a one-line record of the gate answers already passed, reached the exit in **20 of 24 runs across three nights (median depth 20)**; the full-history control reached it in **0 of 23**. The wiped arm also used fewer prompt tokens per turn (887 against 1,260 on the same night). Appending that same answer record to the full history did not rescue it (0/6, median 3), even in runs where the record made the model's chain arithmetic perfect.

**The win is an interaction of three parts, not one trick.** In a same-night 2×2 (n = 12 per arm): the fresh view alone, given to the full-history model, moved nothing (0/12, median 4); the wipe alone moved depth but reached no exit (0/11, median 7); both together reached the exit 10 of 12 times. Removing the record of passed answers from the winning arm sent it below the full-history control (0/6, median 1.5 against 3).

**Why, as far as the commit logs show.** Handed the same fresh view, the model answered a gate correctly on its first attempt 0.95 of the time without its history and 0.72 with it; after a wrong answer, it recovered 0.53 of the time without history and 0.11 with it. Across 118 wrong answers it chose to look again 6 times, though looking was free. The wipe removes the reasoning the model re-derives its wrong answer from; the fresh view supplies the look it does not take. Forcing a look after every wrong answer recovered about a third of the fresh view's value (median 5.5 → 10.5), so the fresh view does more than patch that one habit.

**On a bigger model the trade changes.** llama3.3:70b with its full history reaches depth 15; the wiped arm costs it six gates (median 9), every run stuck re-submitting the same wrong answer at gate 9. Showing it its wrong answers at that gate as exclusions cut the repeat rate from 67% to 4% and moved depth not at all (9). Showing it the ordered record of every commit and its outcome (successes included) took it to median 17.5. On this cell the record of successes is what carries the model; the repetition was a symptom.

Scope: one maze, two models, one seed of the maze, sampled decoding, n = 6–12 per arm. Whether any of this transfers to other tasks is a conjecture this brief does not test.

## 1. Question

What is the usable concentrated form of an agent's context? On nav-3, qwen3:14b with its full history stops at a median depth of 4.5 (the E1a control table), while earlier wiped-context runs on the same map reached 20. That contrast names a policy, not a cause. A wiped context differs from a full one in several ways at once, and an agent builder needs to know which of them to keep.

I decompose the context into channels instead of trimming it by length: the engine's observations (the map and gate text), the model's own replies and actions, the record of what worked and what failed, and raw length. Then I ask, one channel at a time, what removing or adding it does to depth.

## 2. Method

**Substrate.** nav-3 is a 20-gate deterministic ramp under a fog-of-war map (radius 2, present in every arm). Gates are arithmetic over earlier answers ("add 5 to your c1a answer"), so a run needs the values it has already earned. Four wrong answers end a run; the score is ramp depth, the number of gates passed. Steps are effectively unlimited (60, never binding). No LLM judge.

**Arms, named by component.** Every arm uses the same system prompt and the same maze.

| Letter | Component | What the model sees |
|---|---|---|
| — | accumulate (control) | the whole conversation: every engine reply and every one of its own replies |
| **W** | wipe | only the system prompt and the engine's latest message each turn; no earlier turns, no earlier reasoning |
| **R** | free refresh | after any action that is not itself a look, the harness issues a free `observe`, and its output replaces the action's reply as the next input |
| **M** | recall | one line of passed gates and their correct values: `[RECALL — gate answers you have recorded] c1a = 7 \| c1b = 12 …` |
| **L** | forced look | the turn after any wrong answer is overridden to `observe` |
| **K** | commit ledger | every commit this run and its outcome, in order, always shown |
| **P** | failure pad | the current gate's wrong answers, shown as exclusions, only once that gate has at least one wrong answer |

The winning arm is W+R+M. Earlier drafts of this work called it "wipe-curated"; that name survives only as the code identifier `--context-policy wipe-curated --show-recall`.

**Runs.** Arms ran interleaved within a night so that each comparison is same-night. qwen3:14b (Q4_K_M, the same weights blob throughout) at the harness's default sampling (temperature 0.6, top_k 20), on a consumer CUDA GPU for the first two campaigns and an AMD Strix Halo box for the third; the third campaign re-ran the winning arm as a drift check and it reproduced. llama3.3:70b at the server's default sampling. Every death in every arm is `out_of_lives`.

**Pre-registration.** Each campaign named its falsifiers and the band each control had to land in before any row was read; a night whose control left its band would have been discarded. Both drift checks passed on every night reported here.

## 3. Results

### 3.1 Addition does not rescue the full history (n = 6 per arm, one night)

| Arm | Exits | Depths | Median | Prompt tokens/turn, mean (max) |
|---|---|---|---|---|
| accumulate | 0/6 | 2 2 8 6 3 2 | 2.5 | 1,396 (3,140) |
| accumulate + answer record | 0/6 | 3 3 3 5 2 3 | 3 | 1,089 (1,738) |
| accumulate, old engine text dropped | 0/6 | 4 4 5 4 4 4 | 4 | 746 (939) |
| W+R+M | **4/6** | 7 20 20 8 20 20 | **20** | 872 (1,124) |

In three of the six "accumulate + answer record" runs the record drove chain accuracy to 1.00, and those runs still died at gate 3 with all four lives spent. The record did its job and did not save the run.

### 3.2 The fresh view and the wipe are jointly necessary (n = 12 per arm, one night)

| Arm | Exits | Median depth (range) | Gate accuracy | Prompt tokens/turn |
|---|---|---|---|---|
| accumulate | 0/11 | 3 (2–9) | 0.44 | 1,260 |
| accumulate + R | 0/12 | 4 (4–8) | 0.54 | 1,188 |
| W+M (no refresh) | 0/11 | 7 (3–16) | 0.65 | 695 |
| W+R+M | **10/12** | **20** (7–20) | 0.90 | 887 |

Two rows were lost to a GPU fault on the host (one accumulate, one W+M); both arms are reported at n = 11.

What the model had in front of it when it committed, across all 46 runs that night:

| Arm | History | First attempt on a fresh view | Retry after a wrong answer, no new look | Looked again after a wrong answer |
|---|---|---|---|---|
| accumulate | yes | 1.00 when it chose to look (32); 0.12 when it did not (17) | 0.03 (30) | 3 of 33 |
| accumulate + R | yes | 0.72 (58) | 0.11 (36) | 0 of 36 |
| W+R+M | no | 0.95 (207) | 0.53 (15) | 1 of 16 |
| W+M | no | 0.92 (97) | 0.00 (31) | 2 of 33 |

Accuracy shown as share correct (number of commits). The 1.00 for elective looks under history is mostly early gates with short arithmetic and is not evidence that a look neutralizes history at depth.

### 3.3 Recall is necessary on this maze; a forced look closes a third of the gap (n = 6 per arm, one night)

| Arm | Exits | Depths | Median | Gate accuracy |
|---|---|---|---|---|
| accumulate | 0/6 | 5 8 3 3 3 2 | 3 | 0.47 |
| W+R+M | **6/6** | 20 20 20 20 20 20 | **20** | 0.95 |
| W+M | 1/6 | 2 4 7 4 20 7 | 5.5 | 0.60 |
| W+R (no recall) | 0/6 | 1 2 2 2 1 1 | 1.5 | 0.56 |
| W+M+L | 2/6 | 9 20 20 12 7 8 | 10.5 | 0.76 |

Without recall the wiped model spent about twenty turns per run walking back to earlier gates to re-read answers it had already given, and passed one or two gates. The forced look lifted W+M from median 5.5 to 10.5: (10.5 − 5.5) / (20 − 5.5) = 0.34 of the gap to W+R+M.

### 3.4 llama3.3:70b: the record of successes, not the failures (n = 6 per arm, one night)

| Arm | Exits | Depths | Median | Repeat-wrong rate | Prompt tokens/turn (median) |
|---|---|---|---|---|---|
| accumulate | 0/6 | 15 15 15 16 16 15 | 15 | 21% | 2,882 |
| W+R+M | 0/6 | 9 9 9 9 9 9 | 9 | 67% | 809 |
| W+R+M+K | 0/6 | 19 9 18 17 19 9 | 17.5 | 25% | 1,127 |
| W+R+M+P | 0/6 | 9 9 9 9 9 9 | 9 | 4% | 809 |

A repeat-wrong is a wrong answer the model had already been told was wrong at the same gate in the same run. Every W+R+M run committed `35` at gate 9 and re-submitted it until out of lives. With the failure pad, the model stopped repeating (one repeat in 24 wrong answers) and cycled through new wrong answers instead (`35, 34, 55, 29, 41`), dying at the same gate. With the ledger, four of six runs passed gate 9 and reached 17–19; the two that did not died the same way with the ledger in view, and three of the deep runs repeated a wrong answer at gate 19 while the ledger showed it. No arm exited on this model.

## 4. Discussion

**What I think this says, at the n above.** On nav-3, for qwen3:14b, the history the model accumulates is a liability it cannot use well: given a fresh view, it does worse with its history than without, and it re-derives a rejected answer from its own earlier reasoning. Taking the history away is not enough on its own, because the model does not look again after failing; the harness's free refresh does that looking for it. And taking the history away removes the values the next gate needs, so a one-line record of passed answers has to come back in. Wipe, refresh and recall are each necessary here; none is sufficient.

On llama3.3:70b the same wipe costs depth instead of buying it, and the cost is not paid back by preventing repetition. It is paid back by the ordered record of what the model did and what happened. I read that as: a larger model can use more of its past than the 14B can, and the useful part is the record of outcomes, not the replay of reasoning.

**What this does not show.**
- That any of it holds off this maze. nav-3's gates refer to earlier answers by name, so recall's necessity may be a property of the maze rather than of curated context in general. A second maze is the test.
- Significance. These are point estimates at n = 6–12 with one seed of the maze; the larger separations (0 of 23 against 20 of 24) are hard to attribute to noise, the smaller ones (W+M+L against W+M) are not settled.
- That the 70B result generalizes to large models. It is one model on one cell.
- Anything about reasoning traces as a channel. Through this harness the model's reasoning never re-enters its context, so it was never ablated.

**Open.** The 70B cell asks what the ledger carries that recall does not: the order of successes, a pointer to the last passed gate, or the outcomes themselves. A three-arm follow-up (K, a pointer-only arm, recall plus pointer) is designed and not run.

## Data

Per-run JSONL rows with full provenance (model weights hash, engine version, context size, policy code ref), and per-turn context dumps for the 3.1 campaign. *[Author: choose which trace sets ship with this annex and where — see the PR.]*
