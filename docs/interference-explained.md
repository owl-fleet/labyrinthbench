# Interference — How the Traps Are Built

*Design notes for the `rev-2` corridor. What each choice is for, and what it defeats. Results from this substrate live in the [look-gate brief](annex/look-gate-result.md) and the [data annex](https://labyrinthbench.ai/data) — this document is about the map, not the scores.*

## Two different pressures

"Long context" gets used as one word for two problems that fail differently, and the corridor maps split them on purpose.

**Retention** is `nav-3`: 20 gates, each one combining your running total with the answer to a specific *named* earlier gate ("your `c1a` answer"). The reach-backs rotate, so any prior answer can be asked for at any depth. The information you need was true when you wrote it down and is still true now. The only question is whether you can still find it in a context full of your own noise.

**Currency** is `rev-2`: 34 gates, eight variables, values that change as you climb. Here your note exists, you can find it, and it is *wrong*. The failure mode isn't losing the information — it's successfully retrieving a superseded version of it. That's a harder problem, and it's the one this document is about.

A model can be excellent at the first and hopeless at the second, which is why they are separate maps.

## The primitive: your own correct answer becomes the trap

Everything in `rev-2` is built from one repeating three-gate motif:

| position | gate | text |
|---|---|---|
| ask | `use_N` | `Add the current value of C to the current value of D.` |
| mutate | `rev_c_1` | `C changes: C is now 6. Answer 6 to set it.` |
| ask again | `use_N+1` | `Add the current value of C to the current value of D.` |

The third gate is **byte-identical** to the first and has a different answer.

This is what "self-pollution" means in the map's name (`Rev-2 — interference + self-pollution (break the frontier)`). The distractor is not noise the harness injected; it is the model's own correct work, sitting in its notes, matching the current question exactly. A retrieval system that finds the most relevant prior answer will find that one, and it will be wrong.

All fourteen `use_` gates in `rev-2` exist as one half of such a pair. There are no filler questions — every one of them is either the setup for a trap or the trap itself.

## Four choices that keep the trap shut

The motif alone is easy to beat with one heuristic ("recompute everything, always"). These four properties of the map exist to close the cheap escapes. All of them are readable in [`degs/rev-2.yaml`](../degs/rev-2.yaml).

**Values collide on purpose.** The eight variables initialize to A=3, B=6, C=3, D=3, E=8, F=5, G=9, H=6 — A, C and D all hold 3, and B and H both hold 6. So "I remember seeing a 3" does not identify which variable you saw it on. The collisions get worse as you climb, not better: at the final gate the state is A=6, B=6, C=3, D=3, E=4, F=4, G=9, H=6 — eight variables carrying four distinct values.

**Half the variables never move.** Only **A, C, E and F** are ever revised. **B, D, G and H** hold their opening values to the top. You cannot learn "this map changes things" and blanket-recompute your way through, because half of what you'd recompute was never at risk — and you have no way to know which half until you've tracked it.

**One chain is deep.** F revises three times: 5 → 6 → 7 → 4. One refresh is not enough; a model that updates a variable once and trusts it thereafter fails F specifically.

**One variable comes back.** C runs 3 → 6 → **3**. It returns to the value it started with. This is the sharpest tool on the map, because it decouples *correct* from *current*: a model that never updated C at all answers the last C gate correctly, for entirely the wrong reason. Any scoring that only counts right answers will read that as retention. It isn't.

That return is not an accident of authoring, and it isn't hardcoded either. The mint classifies it by simulation — a revision whose value equals an earlier value of the same variable is a **recall** revision rather than a **fresh** one, detected by walking the value ledger, not by matching gate ids. `rev-2` has exactly one: `rev_c_2 → set_c`.

## Where currency actually gets stressed

Beyond the re-ask pairs, four `rsn_` gates ask for a comparison across several currently-valid values at once:

```
Answer the larger of (current C + current G) and (current D + current H).
If the current value of D is greater than the current value of B,
  answer the current value of A; otherwise answer the current value of F.
```

And the last gate before the exit, `syn_final`, asks for the sum of all eight variables. There is no partial credit on a gate: one stale variable out of eight fails it. A model whose ledger is 7/8 current scores exactly the same as one that tracked nothing.

These gates also make failure *locate* itself. In the look-gate cell, the instrumented 14B died almost exclusively on arithmetic-combination gates (11 of 12 classified wrongs) while qwen3-coder-30B died on conditionals (14 of 24) — the tuned model clears one class of gate and fails the next one up. The gate kinds are what make that statement possible.

## Why it's a mint, not a fixture

A published map that never changes is a map that gets memorized. `engine/mint.py` deals *instances*: `mint_instance` is a pure function of `(template, seed)`, so the board can hand you a corridor nobody has seen and still re-derive your entire run from the trace.

The split between what's frozen and what's dealt is where the design lives:

- **Frozen per season** — topology and node ids, the ordered gate-role sequence, every derived gate's `depends_on` / `answer_fn` / problem text, the revision schedule *including* which revisions are recall vs fresh, and the briefing verbatim.
- **Free per instance** — the establish literals and each fresh-revision literal, drawn from the template's inferred range.

The traps survive minting because they are structural. A derived gate like `Add the current value of C to the current value of D` contains no literals at all — it is never rewritten, and it stays a re-ask trap no matter which numbers get dealt. `syn_final` carries an `answer_fn` evaluated against the live ledger rather than a baked answer, so it recomputes itself for free.

Two invariants are enforced rather than hoped for:

- **A fresh revision must actually change the value.** The mint re-draws while the new literal equals the variable's current value. A revision that quietly changed nothing would silently disarm the pair it sits between.
- **Recall revisions copy forward.** C's return is preserved by construction, never re-rolled into a value that isn't a return.

The mint refuses rather than degrades. If the inferred value range spans a single value, it cannot guarantee the first invariant and raises instead of emitting a map with dud traps. If a template's gate wording doesn't match what the mint would render for that role, it raises rather than silently rewriting the phrasing out from under you. And every instance is validated by `simulate_solve` — which walks the corridor exactly as the runner would — *before* it can be written or ledgered. That check is load-bearing: `bfs_verify` is nearly a no-op on a pure lock corridor, where every edge is traversable regardless of whether the values are consistent, so structural verification alone would pass a broken instance.

## The part that matters: a wrong answer names its own mechanism

This is the reason the map is shaped the way it is.

`cli/classify_failures.py` takes a results file and the map, rebuilds the per-variable value timeline, and puts every wrong commit in one of four boxes:

| class | test |
|---|---|
| `unobserved-guess` | answered at a gate the model never observed since arriving |
| `stale-value` | the answer equals an **earlier, now-superseded** value of the variable being asked about |
| `other-var-value` | the answer equals the **current** value of a **different** variable |
| `other-wrong` | observed, wrong, and neither of the above |

No judge, no rubric, no interpretation. `stale-value` is interference caught in the act — the model reached for its notes and got a version that has since been overwritten. `other-var-value` is the deliberate value collisions paying off — the model tracked a number correctly and attached it to the wrong variable. The distinction between those two is a distinction between different memory failures, and it costs one pass over the trace to compute.

That is what the design is *for*. The point of the collisions, the stable decoys, the deep chain and the return isn't only to make the map harder. It's to arrange the state space so that each characteristic way of getting it wrong lands on a different, mechanically detectable answer. Interference here isn't just made difficult — it's made diagnosable.

It reads failures with the same directness the benchmark scores with:

```bash
python cli/classify_failures.py --deg rev-2 --detail /results/rev2.jsonl
```

## What this map deliberately doesn't test

`rev-2` is a corridor: `fog_radius: 0`, no dead ends, no loops, one path. Navigation is *removed*, not merely easy. There is no decomposition to do, no tools to call, and the arithmetic is trivial on purpose — addition and comparison, so that arithmetic ability can't be the thing being measured. The map is what's left when everything except retention and currency is taken away, and that subtraction is exactly what buys the mechanism attribution above. It is not a general agent benchmark. `alpha` is where navigation and recall are tested together.

One known limit, recorded rather than fixed: a re-ask pair can neutralize itself in a given instance. `use_7`/`use_8` ask for the larger of F and G on either side of F's 6 → 7 revision — and with G at 9, both resolve to 9, so the trap doesn't bite. The original `rev-2` has this, and minted instances can produce their own. Balancing branch outcomes across instances is a deferred dealer-service option, not something v1 constrains, so a given instance's effective trap count can be slightly lower than its gate census suggests.

## Run it

```bash
docker exec labyrinthbench python cli/run_eval.py --model <tag> --deg rev-2 --runs 6 \
  --base-url http://host.docker.internal:11434/v1 --output /results/rev2.jsonl

python cli/classify_failures.py --deg rev-2 --detail /results/rev2.jsonl
```

Budget is five wrong answers per run; the score is the deepest gate cleared. If your model dies shallow with every wrong answer classed `unobserved-guess`, it isn't failing at interference yet — it's answering before reading, and `--look-gate` is the flag that separates those two questions. The [look-gate brief](annex/look-gate-result.md) is that experiment.

## Related work

GVS5H ([arXiv:2608.26480](https://arxiv.org/abs/2608.26480v1)) reports that a manager–worker scaffold's gains over single-pass baselines trace in part to context management — short worker calls and shared notes organizing state — established by transcript analysis across nine models on LiveCodeBench Hard. Different task, and a different way of getting at the mechanism: that analysis reads the mechanism out of transcripts, where the design above computes it against a known value timeline.
