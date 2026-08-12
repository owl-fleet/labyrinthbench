# The `--look-gate` result — instruction vs. enforcement of observe-before-answer

*The origin of the harness's `--look-gate` flag. Substrate: rev-2 (the 34-gate belief-revision corridor; manifest public in `degs/`). Design and go/no-go locked before any run; the registered design is summarized in §2 and its one disclosed slip is noted there. Model names literal.*

*Rev (2026-07-16): 120B reference replaced by the journal-verified rerun (`n_ctx_slot` 32768, n=9) — median 34 held; the spread and the P = 0.07 overlap did not (all nine runs exit at 34; complete separation from every local arm). Stats re-derived below.*

## Abstract

On a 34-gate belief-revision corridor whose gate answers are printed in the gate text, a 14B model (qwen3:14b) answers before looking: median depth 5.5 of 34, with every wrong answer an unobserved guess. Three interventions targeted the same variable — does the model observe a gate before answering it. **Telling it to** (a registered one-line imperative in the system rules: "Before you answer any gate, observe it — each gate's problem is unique") moved the median from the pilot's fresh control at 4.5 to 5.5, inside noise (P(rec > none) = 0.72, n = 6 vs. 6, fresh interleaved runs; the older control of record sits at 5.5 itself); the trace shows why: the instruction is ignored at the commit point, not half-followed — the instructed arm observed 0.8 more times per run, and every one of its 30 wrong answers was still an unobserved guess, identical to the uninstructed arm. **Forcing it** — a five-line deterministic interceptor that converts any answer-attempt at an unobserved gate into an observation — lifted the same model to median 21 (registered arm) and 28 (fresh pilot), zero guesses, failing only at depths 18–33 on genuine interference. **Agent-tuned models** of the same class arrive with the discipline built in: Devstral-24B 23.5 and qwen3-coder-30B 28.5, zero guesses, no interceptor. Every looking-fixed arm separates completely from every unfixed arm (P = 0.00 pairwise; no fixed-arm run is as shallow as any unfixed-arm run), while ordering *within* the fixed band is unresolved at n = 6 (P = 0.29–0.46). The unaided 120B reference separates completely from every local arm, instrumented included (median 34 vs. 21, all 54 run-pairings; its nine runs all exit at depth 34 with zero variance at journal-verified 32k context). The interceptor fired 13–31 times per run to the end of every run — enforcement contains the guessing impulse without extinguishing it, which is consistent with the instruction result: a disposition the model cannot hold is not a disposition a sentence can install.

## 1. Question

The prior look-gate cell established that *forcing* qwen3:14b to observe before answering lifts it from median 5.5 to 21+. No arm had ever been *told* to observe. The missing condition decides what the enforcement result means: if instruction matches enforcement, the "instrument" is scaffolding for a prompt; if instruction does nothing, the look-discipline is structural — a loop the harness must close because the model will not.

## 2. Method

Substrate: rev-2, a 34-gate deterministic corridor (fixed manifest, seed 2027; no LLM judge) built on variable-revision interference — values oscillate, several variables share values, and byte-identical questions recur with revisions in between. Ramp depth = gates passed; a budget of 5 wrong answers ends a run.

Three observe-policies, differing from the control config by exactly one element (registered design, locked before any run):

| Policy | Delta vs. base | Mechanism |
|---|---|---|
| none | nothing | control config verbatim, re-run fresh |
| recommended | + one observe-first rule in the system `Rules:` block | instruction, no enforcement |
| forced | + look-gate interceptor (`--look-gate`) | enforcement, no instruction |

An observe-cap (5 consecutive observations end the run, scored at current depth) applies uniformly to all three arms, so termination rules are identical and the policy is the only moving part. The pilot ran n = 6 per policy, fresh and interleaved, qwen3:14b, thinking-on; prior look-gate/control data contributed no evidence to the contrast (the registered motivation-not-evidence rule). A pre-committed go/no-go gated a full 3-policy × 3-model grid on `recommended` emerging as a distinct middle level. Two of the go/no-go's P-value clauses were committed with inverted direction; the median-distance clause carried the verdict unchanged, and the slip is disclosed in place in the registered design. Mechanism metric: `cli/classify_failures.py`, which classes every wrong commit as unobserved-guess / stale-value / other-var-value / other-wrong and locates it on the gate ladder.

Reference points of record: raw control 5.5 (n = 6), ReAct-orchestrated 14b 7 (n = 6; a local agent frame external to this repo), look-gate capped arm 21 (n = 6), Devstral-24b 23.5 (n = 6), qwen3-coder-30b 28.5 (n = 6), gpt-oss:120b unaided 34 (n = 9, rerun 2026-07-16 at journal-verified `n_ctx_slot` 32768).

## 3. Results

**Pilot (fresh, interleaved, n = 6 per arm):**

| Policy | depths | median | 90% CI | obs/commit | unobserved guesses |
|---|---|---|---|---|---|
| none | 3,4,4,5,6,6 | 4.5 | [3.5, 6.0] | 0.32 | 30 of 30 wrongs |
| recommended | 4,5,5,6,7,7 | 5.5 | [4.5, 7.0] | 0.38 | 30 of 30 wrongs |
| forced | 22,25,28,28,29,32 | 28.0 | [25, 30] | 1.20 | 0 |

*(Wrong-answer budget: 5 per run. The registered classifier reads the four non-terminal wrongs per out-of-lives run — 24 per arm here; the six terminal wrongs per arm were classed by a direct trace pass and were also unobserved.)*

P(rec > none) = 0.72; P(forced > rec) = 1.00. The go/no-go verdict: `recommended` collapses toward `none` (1.0 gate off the nearer pole against a ≥ 3 requirement) — the middle condition does not exist, and the grid did not run. A clear imperative bought ~1 gate of median; enforcement bought ~23.

**The full five-condition picture** (one axis: how the looking happens):

| Condition | mechanism | median | sd | guesses |
|---|---|---|---|---|
| raw qwen3:14b | — | 5.5 | 1.4 | 24 |
| ReAct-orchestrated 14b | agent frame | 7 | 1.3 | ~3 |
| recommended (pilot) | instruction | 5.5 | 1.2 | 24 |
| look-gate 14b | enforcement | 21 (capped) / 28 (pilot) | 3.6 | 0 |
| Devstral-24b | agent-tuning, dense | 23.5 | 6.5 | 0 |
| qwen3-coder-30b | agent-tuning, MoE | 28.5 | 8.1 | 0 |
| gpt-oss:120b (ref) | scale, unaided | 34 | 0.0 | 0 |

*(Guess counts are the registered classifier's non-terminal wrongs, per the pilot-table note.)*

Every looking-fixed arm separates completely from every unfixed arm — pairwise P = 0.00; the deepest unfixed run (8) is shallower than the shallowest fixed run (coder's 10). Ordering within the fixed band does not resolve at n = 6 (pairwise P = 0.29–0.46, overlapping CIs); no tuning-vs-instrument ranking is claimed. The unaided 120B separates completely from every local arm, the instrumented ones included: its nine journal-verified runs all exit at depth 34, taking all 54 run-pairings against each fixed arm. An earlier 120B reference with unverified context carried spread (sd 4.2) and a P = 0.07 overlap with the look-gate arm; at verified 32k the spread is gone, and the overlap with it. Against the raw 14b's median gap to the 120B (5.5 → 34), the registered enforcement arm recovers 54%, and the fresh pilot's forced arm 79% — from five lines of harness code and no weight changes.

## 4. Mechanism (trace analysis)

**Instruction fails by non-deployment, not half-deployment.** The pre-registered candidate mechanisms for `recommended` were "observes more but still guesses" and "ignores the instruction outright." The traces show the second, cleanly: all 24 wrong answers at never-observed gates on the trivial init ladder (gates 2–8), exactly like `none`, with zero observed-wrongs in either arm — when this model looks at a trivial gate, it reads it correctly. The instruction's entire measured effect is +0.8 observations per run that never landed on the gate about to be failed.

**Enforcement contains the impulse; nothing extinguishes it.** Across all 12 forced runs, the interceptor fired 13–31 times per run (median ≈ 20), and ~45% of interceptions fall after each run's midpoint — 40+ turns of enforced look-then-answer rhythm leave the commit-without-looking impulse intact at the frontier of every run. The instrument does not teach; it catches.

**What the deepest run says about the wall.** The record forced run (depth 32 — two gates from the first 14B exit on record) stalled at a conditional gate through a chain worth reporting exactly. Its honest re-derivation concluded "B is not initialized in any previous gate — thus B = 0," while gate 2's text "Variable B is initialized to 6" sat ~4.8k tokens into a 16k window — in-context reconstruction failure, not truncation. It answered wrong, then *self-corrected to the right answer* one turn later — and the commit was intercepted (the arm requires re-observation after any commit), after which the model exited the JSON action protocol into analysis prose for four turns and the observe-cap ended the run. The wrong answer is interference; the death is a protocol lapse the instrument declined to bail out. The instrument's strictness has a measurable cost at the frontier, and this run paid it while holding the correct answer.

**What tuning installs that the instrument doesn't: revision.** At the same interference wall, the failure signatures diverge. The instrumented 14b re-committed the *identical* wrong answer in 5 of its 12 classified registered-arm wrongs (one gate retried 3,3,3,3); across the two tuned arms' 44 classified wrongs there is not one identical retry — every retry was a changed answer. The interceptor installs look-before-answer; it does not install revise-after-wrong. The failure *locations* also differ: the instrumented 14b dies almost exclusively on arithmetic-combination gates (11 of 12), qwen3-coder on conditionals (14 of 24) — the tuned model clears the arithmetic class and fails one class later.

**Why the instrument is the consistency king.** Look-gate sd 3.6 vs. tuning's 6.5/8.1. The interceptor makes the trivial ladder unlosable — a guess attempt becomes a look, and looked-at trivial gates are always answered correctly — so instrument deaths start only where interference starts (terminal gates 19–30, a tight band). The tuned arms reach deeper (runs of 33, dying one and two gates from exit) but keep an early-collapse tail (runs of 10, 12, 15, including two Devstral deaths from answer-less commits while backtracking). Tuning buys the ceiling and keeps the flubs; the instrument trades the ceiling for a floor.

## 5. Boundaries

n = 6 per cell throughout; one substrate family (rev-2, one seed); the instrument tested on one base model (qwen3:14b); one instruction wording at one registered strength — a mid-strength imperative chosen so that neither a vague hint's failure nor a heavy-handed win would carry the result. The 120B reference ran unaided (n = 9, journal-verified `n_ctx_slot` 32768, all nine runs exiting at depth 34); instrumenting it too was never tested, so the recovery fractions bound what a five-line rule recovers, not what scale is worth. Effective context was journal-verified for no other arm — the practice post-dates them. On record: the overnight arms requested a 16k window (`--num-ctx 16384`) and the depth-32 trace shows ~16k in operation for the instrumented arm; ollama's `/v1` endpoint drops the request for models that don't take the native thinking path, so the two tuned arms' effective windows are unconfirmed. The superseded 120B reference (2026-06-03, context unverified) carried sd 4.2; the verified rerun's zero variance is consistent with that spread having been a context artifact, though this is a plausibility, not a demonstrated mechanism. Ordering within the fixed band awaits a larger-n escalation. The observe-cap is crude — the record run shows it (with the re-observe rule) ending a run that held the correct answer; a cap-aware retry rule is untested.

## Provenance

Design + go/no-go locked pre-run; outcome tables, separation statistics, and the trace-mechanism pass are from the lab records those runs produced. Classifier: `cli/classify_failures.py` (`--label`, `--detail`) — public in this repo. Context: `--num-ctx 16384` requested on the overnight arms; ~16k confirmed in operation for the instrumented record run; the 120B reference is the one arm with journal-verified context (32768, the 2026-07-16 rerun, superseding the 2026-06-03 reference, which is kept as the record of the superseded number). Pairwise separation re-derived 2026-07-16 — P(row > col) over all run-pairs, 20k-bootstrap median CIs; new 120B CI [34, 34]. Raw per-run JSONLs for every arm in this brief's tables are public in the rev-2 annex bundle ([`lb-rev2-raw-jsonl.zip`](https://labyrinthbench.ai/assets/data/lb-rev2-raw-jsonl.zip), added 2026-08-12 alongside the nav-3 bundle); its manifest maps each file to its table cell, and each file's per-run depths reproduce the corresponding row above.
