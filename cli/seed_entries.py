#!/usr/bin/env python3
"""Seed leaderboard entries from the E1a aggregate results.

Converts results/e1a-table1/e1a_table1.json (produced by cli/e1a_table1.py from
the per-run results JSONLs) into one entry file per leaderboard row under
entries/. Board-seed entries predate the dealer/runner flow, so they carry
origin="board-seed" and no artifact hashes.

Lanes:
  model   — standard harness pinned, one entry per model (control arm)
  harness — model pinned (per division's pinned_model), one entry per harness

Rank metric: one-sided 95% bootstrap lower confidence bound on median depth
(seeded RNG, so output is byte-deterministic for a given input file).

Provenance: model entries carry nullable quantization/digest fields. Board-seed
entries predate as-run digest capture (standing practice from the quant-ladder
cell, 2026-07-21), so they are null except where verifiable: the pinned-lane
qwen3:14b digest is registry-unmodified since 2026-05-21 (pre-campaign) and
identical on both run hosts, so today's manifest digest IS the as-run digest.

Usage (aggregate — the E1a campaign path):
  python3 cli/seed_entries.py [--src results/e1a-table1/e1a_table1.json] [--out entries]

Usage (--single — a lone run/instance, no e1a_table1.py aggregate needed):
  python3 cli/seed_entries.py --single results.jsonl --entry-id my-entry --model-id my-model
    [--lane model|harness] [--arm-name control] [--display "My Model"] [--planned-n N]
    [--harness-name standard --harness-summary "..."] [--ceiling-row] [--out entries]

  Until the dealer/runner flow ships (METHODOLOGY.md §5), a lone results JSONL — one dealt
  instance, one or a handful of runs — has no tool-made path to a board entry short of
  hand-editing one to match this script's schema; --single is that path. It reuses
  cli/e1a_table1.py's cell_metrics() for the runs-block aggregation (same field names
  seed_entries already expects) and the identical seeded-bootstrap score_block() below, so a
  --single entry scores exactly like an aggregate one and verifies with
  `cli/verify.py --entry` the same way.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

BOOTSTRAP_B = 10_000
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_SEED = 1337

# aggregate-key -> display name (ollama-style tags where they exist)
DISPLAY = {
    "deepseek-r1-70b": "deepseek-r1:70b",
    "gemma4-12b": "gemma4:12b",
    "gemma4-31b": "gemma4:31b",
    "glm-4-7-flash": "glm-4.7-flash",
    "gpt-oss-120b": "gpt-oss:120b",
    "hf-co-InternScience-Agents-A1-Q4_K_M-GGUF": "InternScience Agents-A1 (Q4_K_M GGUF)",
    "hf-co-empero-ai-Qwythos-9B-Claude-Mythos-5-1M-GGUF-Q4_K_M": "Qwythos-9B (Q4_K_M GGUF)",
    "llama3-3-70b": "llama3.3:70b",
    "llama4-scout": "llama4:scout",
    "ornith-9b": "ornith:9b",
    "qwen3-14b": "qwen3:14b",
    "qwen3-5-122b": "qwen3.5:122b",
    "qwen3-5-27b": "qwen3.5:27b",
    "qwen3-5-9b": "qwen3.5:9b",
    "qwen3-6-27b": "qwen3.6:27b",
}

# The harness lane's launch division. Additional divisions are a data change:
# add a key here (or introduce per-division config) and re-run the seeder.
HARNESS_DIVISIONS = {"qwen3-14b"}

# As-run provenance where verifiable (see docstring); everything else stays null.
# Fixed-model lane pin: qwen3:14b at Q4_K_M (quant-ladder prereg outcome, 2026-07-21).
PROVENANCE = {
    "qwen3-14b": {
        "quantization": "Q4_K_M",
        "digest": "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
    },
    "hf-co-InternScience-Agents-A1-Q4_K_M-GGUF": {"quantization": "Q4_K_M", "digest": None},
    "hf-co-empero-ai-Qwythos-9B-Claude-Mythos-5-1M-GGUF-Q4_K_M": {"quantization": "Q4_K_M", "digest": None},
}


def model_block(key):
    prov = PROVENANCE.get(key, {})
    return {
        "id": key,
        "display": DISPLAY.get(key, key),
        "quantization": prov.get("quantization"),
        "digest": prov.get("digest"),
    }

HARNESS_DESC = {
    "control": {
        "name": "standard",
        "summary": "Accumulating context, no intervention — the pinned baseline harness.",
    },
    "wiped": {
        "name": "wiped-curated",
        "summary": "History wiped each turn; a curated note overlay is the only carried state.",
    },
}


def bootstrap_ci_lower(depths, rng):
    """One-sided lower bound: the (1-level) quantile of bootstrap medians."""
    n = len(depths)
    medians = sorted(
        statistics.median(rng.choices(depths, k=n)) for _ in range(BOOTSTRAP_B)
    )
    idx = int((1 - BOOTSTRAP_LEVEL) * BOOTSTRAP_B)
    return medians[idx]


def runs_block(arm):
    return {
        "n": arm["n_valid"],
        "depths": arm["depths"],
        "n_exit": arm["n_exit"],
        "exit_rate": arm["exit_rate"],
        "turns_mean": arm.get("turns_mean"),
        "turns_per_gate_mean": arm.get("turns_per_gate_mean"),
        "consistency_mean": arm.get("consistency_mean"),
        "elapsed_mean_s": arm.get("elapsed_mean"),
    }


def score_block(arm, seed_string):
    # The recorded seed is the FULL derived string handed to random.Random — not the base
    # constant — so the bound re-derives from the entry file alone (cli/verify.py --entry),
    # without this script's per-entry derivation convention.
    rng = random.Random(seed_string)
    return {
        "depth_median": arm["depth_median"],
        "ci_lower_median": bootstrap_ci_lower(arm["depths"], rng),
        "bootstrap": {"B": BOOTSTRAP_B, "level": BOOTSTRAP_LEVEL, "seed": seed_string},
    }


def base_entry(entry_id, lane, src_rel, arm_name):
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "lane": lane,
        "origin": "board-seed",
        "season": 0,
        "source": {"dataset": src_rel, "arm": arm_name},
        # Completeness declaration: the cohort size fixed BEFORE any run. The board badges
        # an entry whose runs.n falls short of its declaration — shortfall is visible, never
        # silent (fail-closed publication; the E1a campaign pre-registered n=6 per cell).
        "declared": {"planned_n": 6, "source": "docs/annex/prereg-cohort-campaign.md"},
        "artifacts": {"harness_container": None, "model_file": None},
    }


def single_entry(args) -> None:
    """Build ONE board entry directly from a plain results JSONL (one row per run, n can be 1),
    bypassing the e1a_table1.py campaign aggregator entirely — see the --single usage note in the
    module docstring. Reuses cli/e1a_table1.py's cell_metrics() so the runs-block fields line up
    with runs_block() below exactly as they do for the aggregate path."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cli.e1a_table1 import cell_metrics  # noqa: E402 — only needed in --single mode

    if not args.entry_id or not args.model_id:
        raise SystemExit("--single requires --entry-id and --model-id")

    src = Path(args.single)
    valid, errors = [], []
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "error" in row:
                errors.append(str(row["error"]))
            else:
                valid.append(row)
    if not valid:
        raise SystemExit(f"REFUSED: no valid run rows in {src} (n_error={len(errors)})")

    arm = cell_metrics(valid, errors)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    entry = base_entry(args.entry_id, args.lane, str(src), args.arm_name)
    entry["declared"] = {
        "planned_n": args.planned_n if args.planned_n is not None else arm["n_valid"],
        "source": str(src),
    }
    model = model_block(args.model_id)
    if args.display:
        model["display"] = args.display
    entry["pinned_model" if args.lane == "harness" else "model"] = model
    entry["harness"] = {"name": args.harness_name, "summary": args.harness_summary, "open": True}
    entry["ceiling_row"] = bool(args.ceiling_row)
    entry["runs"] = runs_block(arm)
    entry["score"] = score_block(arm, f"{BOOTSTRAP_SEED}:{args.entry_id}:single")

    path = out / f"{entry['entry_id']}.json"
    path.write_text(json.dumps(entry, indent=1) + "\n")
    print(f"wrote 1 entry (n={arm['n_valid']} valid runs, {len(errors)} error rows skipped) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/e1a-table1/e1a_table1.json",
                    help="Aggregate mode (default): results/e1a-table1/e1a_table1.json from cli/e1a_table1.py.")
    ap.add_argument("--out", default="entries")
    ap.add_argument("--single", metavar="JSONL", default=None,
                    help="Single-run/instance mode: a plain results JSONL (one row per run) instead "
                         "of the e1a_table1.py aggregate. Requires --entry-id and --model-id.")
    ap.add_argument("--entry-id", default=None, help="--single mode: entry_id and output filename stem.")
    ap.add_argument("--model-id", default=None,
                    help="--single mode: model key — looked up in DISPLAY/PROVENANCE if it matches "
                         "an existing key, else used verbatim as both id and display.")
    ap.add_argument("--lane", choices=["model", "harness"], default="model", help="--single mode: leaderboard lane.")
    ap.add_argument("--arm-name", default="control", help="--single mode: source.arm label.")
    ap.add_argument("--display", default=None, help="--single mode: override the model display name.")
    ap.add_argument("--harness-name", default="standard", help="--single mode: harness.name field.")
    ap.add_argument("--harness-summary", default="", help="--single mode: harness.summary field.")
    ap.add_argument("--planned-n", type=int, default=None,
                    help="--single mode: declared.planned_n (default: the JSONL's own valid-run count).")
    ap.add_argument("--ceiling-row", action="store_true", help="--single mode: mark ceiling_row true.")
    args = ap.parse_args()

    if args.single:
        single_entry(args)
        return

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    data = json.loads(src.read_text())

    written = []
    for key, row in sorted(data.items()):
        control = row.get("control")
        if not control or not control.get("n_valid"):
            continue
        entry = base_entry(f"seed-model-{key}", "model", str(src), "control")
        entry["model"] = model_block(key)
        entry["harness"] = {**HARNESS_DESC["control"], "open": True}
        entry["ceiling_row"] = bool(row.get("contrast", {}).get("ceiling_row"))
        entry["runs"] = runs_block(control)
        entry["score"] = score_block(control, f"{BOOTSTRAP_SEED}:{key}:control")
        path = out / f"{entry['entry_id']}.json"
        path.write_text(json.dumps(entry, indent=1) + "\n")
        written.append(path.name)

        if key in HARNESS_DIVISIONS:
            for arm_name in ("control", "wiped"):
                arm = row.get(arm_name)
                if not arm or not arm.get("n_valid"):
                    continue
                h = base_entry(
                    f"seed-harness-{key}-{HARNESS_DESC[arm_name]['name']}",
                    "harness",
                    str(src),
                    arm_name,
                )
                h["pinned_model"] = model_block(key)
                h["harness"] = {**HARNESS_DESC[arm_name], "open": True}
                h["runs"] = runs_block(arm)
                h["score"] = score_block(arm, f"{BOOTSTRAP_SEED}:{key}:harness:{arm_name}")
                path = out / f"{h['entry_id']}.json"
                path.write_text(json.dumps(h, indent=1) + "\n")
                written.append(path.name)

    print(f"wrote {len(written)} entries -> {out}/")
    for name in written:
        print(" ", name)


if __name__ == "__main__":
    main()
