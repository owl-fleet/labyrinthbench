#!/usr/bin/env python3
"""E1a Table-1 analysis harness — the paired {control, wiped-overlay} depth sweep (prereg §3).

Reads the per-run ``e1a-<model>-<arm>.jsonl`` files a night/flat-out campaign writes
(``e1a-night-scheduler.sh`` → ``e1a-run-row.sh`` → ``run_eval.py``) and emits the
pre-registered Table 1:

  * PRIMARY readout — depth reached of ``CEILING`` (``ramp_depth``), exact per-cell depths
    published in run (file) order (prereg §3: "no summary-only cells"), with median and
    mean ± SEM summaries.
  * SECONDARY — exit rate (``found_exit``).
  * REPORTED, NEVER GATING — turns, knowledge-state consistency, gate accuracy, wall-clock,
    the observe/commit mechanism ratio (from ``turns_log``), and Table 1c's turns-per-gate
    efficiency contrast. Unobserved-guess count is enriched from ``classify_failures.py``
    when it + the DEG ladder are importable (``--degs-dir``); otherwise left null with a
    note (it is secondary). Error rows are counted and their causes summarized in a
    footnote derived from the JSONLs themselves.

For each model it computes the paired contrast ``wiped_median − control_median`` against the
registered falsifier (``>= FALSIFIER_DELTA`` = 25% of the ceiling) and flags ceiling rows
(control already at the ceiling — the instrument's range, not the lever failing).

Outputs (to ``--out-dir``): ``e1a_table1.md`` (the human table), ``e1a_table1.json`` (machine
readable), ``e1a_table1_stripplot.png`` (control-vs-wiped strip plot, paper-2 palette).

Self-contained: stdlib + numpy + matplotlib. Resume-safe: tolerates partial trailing lines
from a live run and unequal / in-progress cells (reports ``n`` per cell — the campaign is
n=6 per cell when complete).

Usage:
  python3 e1a_table1.py --results-dir <dir of e1a-*.jsonl> --out-dir <dir> [--degs-dir <degs>]
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

CEILING = 20          # nav-3 optimal_commits (gates); ramp_depth is of 20
FALSIFIER_DELTA = 5   # prereg §3 replicate falsifier: wiped median − control median >= 5
ARMS = ("control", "wiped")

_FNAME_RE = re.compile(r"^e1a-(?P<safe>.+)-(?P<arm>control|wiped)\.jsonl$")


def discover(results_dir):
    """Map safe-model-name -> {arm: filepath} for every e1a-*-{control,wiped}.jsonl found."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "e1a-*.jsonl"))):
        m = _FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        cells.setdefault(m.group("safe"), {})[m.group("arm")] = path
    return cells


def load_cell(path):
    """Return (valid_rows, error_causes). Error rows are {model,deg_id,error,run_label};
    their cause strings (first line) are kept so the footnote can be derived from the data."""
    valid, errors = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line from a run still in flight — skip, matches count_valid
            if "error" in row:
                cause = str(row["error"]).splitlines()[0]
                # error strings can embed endpoint URLs; the aggregate ships publicly
                cause = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[lan-host]", cause)
                errors.append(cause[:60] + "…" if len(cause) > 60 else cause)
            else:
                valid.append(row)
    return valid, errors


def _obs_commit(row):
    obs = com = 0
    for turn in row.get("turns_log") or []:
        ap = turn.get("action_parsed") or {}
        act = ap.get("action")
        if act == "observe":
            obs += 1
        elif act == "commit" and str(ap.get("answer") or "").strip():
            com += 1
    return obs, com


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _sem(xs):
    xs = [x for x in xs if x is not None]
    return (statistics.stdev(xs) / len(xs) ** 0.5) if len(xs) >= 2 else None


def cell_metrics(valid, errors):
    # depths kept in run (file) order — sorting them manufactures a warm-up curve that
    # is not in the data (Table-1 review, 2026-07-14)
    depths = [r["ramp_depth"] for r in valid if r.get("ramp_depth") is not None]
    exits = [bool(r.get("found_exit")) for r in valid]
    oc = [_obs_commit(r) for r in valid]
    tot_obs, tot_com = sum(o for o, _ in oc), sum(c for _, c in oc)
    tpg = [r["turns"] / r["ramp_depth"] for r in valid
           if r.get("turns") is not None and r.get("ramp_depth")]
    return {
        "n_valid": len(valid),
        "n_error": len(errors),
        "errors": errors,
        "depths": depths,
        "depth_median": _median(depths),
        "depth_mean": _mean(depths),
        "depth_sem": _sem(depths),
        "depth_min": min(depths) if depths else None,
        "depth_max": max(depths) if depths else None,
        "n_exit": sum(exits),
        "exit_rate": _mean([1.0 if e else 0.0 for e in exits]),
        "turns_per_gate": tpg,
        "turns_per_gate_mean": _mean(tpg),
        "turns_per_gate_sem": _sem(tpg),
        "turns_mean": _mean([r.get("turns") for r in valid]),
        "consistency_mean": _mean([r.get("knowledge_state_consistency") for r in valid]),
        "gate_acc_mean": _mean([r.get("gate_accuracy") for r in valid]),
        "elapsed_mean": _mean([r.get("elapsed_seconds") for r in valid]),
        "obs_commit_ratio": (tot_obs / tot_com) if tot_com else None,
        "total_obs": tot_obs,
        "total_commit": tot_com,
        "unobserved_guesses": None,   # filled by enrich_mechanism() when available
    }


def compute_unobserved_guesses(per_cell_valid, degs_dir):
    """Best-effort unobserved-guess counts via classify_failures.py (prereg mechanism metric).

    Returns {(safe, arm): count}; empty when unavailable. Never raises — the mechanism metric is
    reported, never gating.
    """
    if not degs_dir:
        return {}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import classify_failures as cf
    except Exception as exc:
        print(f"[mechanism] classify_failures unavailable ({exc}); unobserved-guess left null", file=sys.stderr)
        return {}
    out, ladder_cache = {}, {}
    for (safe, arm), valid in per_cell_valid.items():
        try:
            total = 0
            for row in valid:
                deg = row.get("deg_id", "nav-3")
                if deg not in ladder_cache:
                    ladder_cache[deg] = cf.load_ladder(deg, degs_dir)
                gates, var_history = ladder_cache[deg]
                counts = cf.classify_run(row, gates, var_history)
                if isinstance(counts, dict):
                    total += counts.get("unobserved-guess", 0)
            out[(safe, arm)] = total
        except Exception as exc:
            print(f"[mechanism] {safe}/{arm}: {exc}", file=sys.stderr)
    return out


def build(results_dir, degs_dir=None, control_only=False):
    cells = discover(results_dir)
    per_cell_valid = {}
    table = {}
    for safe, arms in cells.items():
        table[safe] = {"safe": safe}
        for arm in ARMS:
            if control_only and arm == "wiped":
                table[safe][arm] = None      # wiped arm suppressed (e.g. aborted/invalid)
                continue
            path = arms.get(arm)
            if not path:
                table[safe][arm] = None
                continue
            valid, errors = load_cell(path)
            per_cell_valid[(safe, arm)] = valid
            table[safe][arm] = cell_metrics(valid, errors)

    # optional mechanism enrichment
    guesses = compute_unobserved_guesses(per_cell_valid, degs_dir)
    for safe in cells:
        for arm in ARMS:
            if table[safe].get(arm) and (safe, arm) in guesses:
                table[safe][arm]["unobserved_guesses"] = guesses[(safe, arm)]

    # paired contrast + verdict per model
    for safe in table:
        c = table[safe].get("control")
        w = table[safe].get("wiped")
        cm = c["depth_median"] if c else None
        wm = w["depth_median"] if w else None
        delta = (wm - cm) if (cm is not None and wm is not None) else None
        ceiling_row = (cm is not None and cm >= CEILING)
        tpg_ratio = (w["turns_per_gate_mean"] / c["turns_per_gate_mean"]
                     if (c and w and c.get("turns_per_gate_mean") and w.get("turns_per_gate_mean"))
                     else None)
        table[safe]["contrast"] = {
            "control_median": cm,
            "wiped_median": wm,
            "delta_wiped_minus_control": delta,
            "meets_falsifier": (delta is not None and delta >= FALSIFIER_DELTA),
            "turns_per_gate_ratio_wiped_over_control": tpg_ratio,
            "ceiling_row": ceiling_row,
            "complete": (bool(c and c["n_valid"] >= 6) if control_only
                         else bool(c and w and c["n_valid"] >= 6 and w["n_valid"] >= 6)),
        }
    return table


# ────────────────────────────────────────────────────────────────── rendering

def _fmt(x, nd=2, pct=False):
    if x is None:
        return "—"
    if pct:
        return f"{x*100:.0f}%"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(table, control_only=False):
    lines = []
    if control_only:
        lines.append("# E1a — Table 1 (control baseline): nav-3 depth sweep\n")
        lines.append(f"*Control arm only (wiped arm aborted — invalid, see the campaign record). "
                     f"Primary readout: depth reached of {CEILING}; exact per-cell depths published "
                     f"(prereg §3). Generated by `labyrinthbench/cli/e1a_table1.py --control-only`.*\n")
    else:
        lines.append("# E1a — Table 1: paired {control, wiped-overlay} depth sweep (nav-3)\n")
        lines.append(f"*Primary readout: depth reached of {CEILING}. Exact per-cell depths published "
                     f"(prereg §3). Falsifier: wiped median − control median ≥ {FALSIFIER_DELTA}. "
                     f"Generated by `labyrinthbench/cli/e1a_table1.py`.*\n")

    # Table 1 — per cell
    lines.append("## Table 1 — per cell\n")
    lines.append("| Model | Arm | n (valid/err) | depths (of 20, run order) | median | mean ± SEM | exit | turns | consist. | obs/com |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    arms = ("control",) if control_only else ARMS
    err_notes = []
    for safe in sorted(table):
        for arm in arms:
            c = table[safe].get(arm)
            name = safe if arm == "control" else ""
            if not c:
                lines.append(f"| {name} | {arm} | — | — | — | — | — | — | — | — |")
                continue
            depths = ",".join(str(d) for d in c["depths"]) or "—"
            mean_sem = (f"{_fmt(c['depth_mean'])} ± {_fmt(c['depth_sem'])}"
                        if c["depth_sem"] is not None else _fmt(c["depth_mean"]))
            lines.append(
                f"| {name} | {arm} | {c['n_valid']}/{c['n_error']} | {depths} | "
                f"{_fmt(c['depth_median'])} | {mean_sem} | {_fmt(c['exit_rate'], pct=True)} | "
                f"{_fmt(c['turns_mean'],1)} | {_fmt(c['consistency_mean'], pct=True)} | "
                f"{_fmt(c['obs_commit_ratio'])} |")
            for cause, k in sorted({e: c["errors"].count(e) for e in c["errors"]}.items()):
                err_notes.append(f"{safe} ({arm}): {cause}" + (f" ×{k}" if k > 1 else ""))
    lines.append("")
    lines.append("*n (valid/err) = completed runs / errored attempts. An errored attempt records "
                 "only its failure cause — no partial depth data enters the table — and the "
                 "campaign retried until n=6 valid.*"
                 + (" Errors in this dataset: " + "; ".join(err_notes) + "." if err_notes else "")
                 + "\n")

    if control_only:
        return "\n".join(lines)

    # Table 1b — paired contrast / verdict
    lines.append("## Table 1b — paired contrast (the headline test)\n")
    lines.append("| Model | control median | wiped median | Δ (w−c) | meets falsifier (≥5) | note |")
    lines.append("|---|---|---|---|---|---|")
    for safe in sorted(table):
        ct = table[safe]["contrast"]
        w = table[safe].get("wiped")
        notes = []
        if ct["ceiling_row"]:
            if w is None:
                note = "ceiling row (control at 20 — instrument range, not lever failure); wiped arm not run"
            else:
                note = "ceiling row — A3 efficiency arm (depth falsifier N/A; readout = Table 1c + exit non-inferiority)"
            notes.append(note)
        elif w is None:
            notes.append("control-only (no wiped arm run)")
        else:
            short = [arm for arm in ARMS
                     if not table[safe].get(arm) or table[safe][arm]["n_valid"] < 6]
            if short:
                notes.append(f"**partial** (n<6 in {', '.join(short)})")
        lines.append(
            f"| {safe} | {_fmt(ct['control_median'])} | {_fmt(ct['wiped_median'])} | "
            f"{_fmt(ct['delta_wiped_minus_control'])} | "
            f"{'✅' if ct['meets_falsifier'] else '—'} | {'; '.join(notes) or ''} |")
    lines.append("")

    # Table 1c — turns-per-gate efficiency (reported, never gating; exit is the only objective)
    lines.append("## Table 1c — turns-per-gate efficiency (reported, never gating)\n")
    lines.append("*Per-run turns ÷ ramp depth (turns spent per gate cleared); cell mean ± SEM. "
                 "Read jointly with depth — the ratio conflates progress rate with post-stall "
                 "flailing, and lives/turn-budget truncation ends runs early. A lower wiped ratio "
                 "at LOWER depth (e.g. a fast shallow stall) is not a win. Elapsed wall-clock is "
                 "uncontrolled across arms (non-interleaved lanes, different hosts, and per-turn "
                 "prefill differs: the wiped overlay changes the prompt prefix every turn) — turns "
                 "is the load-independent readout.*\n")
    lines.append("| Model | depth median (c→w) | turns/gate control | turns/gate wiped | w/c | elapsed mean, min (c→w) |")
    lines.append("|---|---|---|---|---|---|")
    for safe in sorted(table):
        c, w, ct = table[safe].get("control"), table[safe].get("wiped"), table[safe]["contrast"]

        def _ms(cell):
            if not cell or cell["turns_per_gate_mean"] is None:
                return "—"
            sem = cell["turns_per_gate_sem"]
            return (f"{_fmt(cell['turns_per_gate_mean'])} ± {_fmt(sem)}"
                    if sem is not None else _fmt(cell["turns_per_gate_mean"]))

        ratio = (w["turns_per_gate_mean"] / c["turns_per_gate_mean"]
                 if c and w and c.get("turns_per_gate_mean") and w.get("turns_per_gate_mean")
                 else None)
        el = "—"
        if c and c.get("elapsed_mean") is not None:
            el = f"{c['elapsed_mean']/60:.1f}"
            if w and w.get("elapsed_mean") is not None:
                el += f" → {w['elapsed_mean']/60:.1f}"
        depth_cw = _fmt(ct["control_median"]) + (f" → {_fmt(ct['wiped_median'])}"
                                                 if ct["wiped_median"] is not None else "")
        lines.append(f"| {safe} | {depth_cw} | {_ms(c)} | {_ms(w)} | {_fmt(ratio)} | {el} |")
    lines.append("")

    # consistency measurement-validity flag (do not interpret — surface it)
    wiped_cons = [table[s]['wiped']['consistency_mean'] for s in table
                  if table[s].get('wiped') and table[s]['wiped']['consistency_mean'] is not None]
    if wiped_cons and all(v == 0.0 for v in wiped_cons):
        lines.append("> ⚠ **Measurement-validity flag (not interpreted here):** knowledge-state "
                     "consistency reads 0% for *every* wiped cell while control cells read >0%. "
                     "Confirm the metric can see the wiped arm (it scores chain-gate execution vs "
                     "guessing; the wiped overlay may leave nothing for it to assess) before reading "
                     "it as behavior.\n")
    return "\n".join(lines)


def render_stripplot(table, out_png, control_only=False, paired_only=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # paper-2 arm palette (drafts/figures/paper2_figures.py)
    COL = {"control": "#b0413e", "wiped": "#2a7fb8"}
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })

    models = [s for s in sorted(table)
              if table[s].get("control") or table[s].get("wiped")]
    # Paired contrast: drop models with no wiped pair (ceiling / control-only rows) so the plot
    # shows only the {control, wiped} pairs it is claiming. Omission is annotated, never silent.
    excluded = []
    if paired_only and not control_only:
        paired = [s for s in models if table[s].get("control") and table[s].get("wiped")]
        excluded = [s for s in models if s not in paired]
        models = paired
    n = len(models)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * n + 1.5), 4.6))

    def strip(x, vals, color):
        if not vals:
            return
        xs = x + np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else [x]
        ax.scatter(xs, vals, s=64, color=color, zorder=3, edgecolor="white", linewidth=0.7)
        ax.hlines(np.median(vals), x - 0.16, x + 0.16, color=color, linewidth=2.6, zorder=2)

    for i, safe in enumerate(models):
        c = table[safe].get("control")
        if control_only:
            strip(i, c["depths"] if c else [], COL["control"])
        else:
            w = table[safe].get("wiped")
            strip(i - 0.17, c["depths"] if c else [], COL["control"])
            strip(i + 0.17, w["depths"] if w else [], COL["wiped"])

    ax.axhline(CEILING, ls="--", color="#666", lw=1)
    ax.text(n - 0.5, CEILING, f" ceiling = {CEILING}", va="center", ha="left", fontsize=9, color="#666")
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"depth reached (of {CEILING})")
    ax.set_ylim(-0.5, CEILING + 1.5)
    ax.set_title("Control baseline: depth reached per model" if control_only
                 else "Control vs. context-wiped runs, per model",
                 pad=(22 if excluded else 12))
    if excluded:
        ceil_ex = [s for s in excluded if table[s]["contrast"]["ceiling_row"]]
        ax.text(0.5, 1.005,
                f"{len(excluded)} models omitted — no wiped pair "
                f"({len(ceil_ex)} ceiling + {len(excluded) - len(ceil_ex)} control-only); see Table 1b",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color="#888", style="italic")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["control"], markersize=9,
                      label="control (accumulating history)")]
    if not control_only:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["wiped"], markersize=9,
                              label="wiped-overlay (--overlay-only)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="E1a Table-1 analysis harness")
    ap.add_argument("--results-dir", required=True, help="dir containing e1a-*-{control,wiped}.jsonl")
    ap.add_argument("--out-dir", required=True, help="output dir for e1a_table1.{md,json,png}")
    ap.add_argument("--degs-dir", default=None, help="optional DEG manifest dir for the unobserved-guess mechanism metric")
    ap.add_argument("--control-only", action="store_true", help="suppress the wiped arm (e.g. aborted/invalid) — control baseline only")
    ap.add_argument("--stripplot-all", action="store_true", help="plot every model in the strip plot, including ceiling / control-only rows with no wiped pair (default: paired-only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    table = build(args.results_dir, degs_dir=args.degs_dir, control_only=args.control_only)
    if not table:
        print(f"No e1a-*-{{control,wiped}}.jsonl files found in {args.results_dir}", file=sys.stderr)
        raise SystemExit(2)

    md = render_markdown(table, control_only=args.control_only)
    with open(os.path.join(args.out_dir, "e1a_table1.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.out_dir, "e1a_table1.json"), "w") as fh:
        json.dump(table, fh, indent=2, default=str)
    render_stripplot(table, os.path.join(args.out_dir, "e1a_table1_stripplot.png"),
                     control_only=args.control_only, paired_only=not args.stripplot_all)

    # stdout summary for validation
    print(f"models: {len([s for s in table])}")
    for safe in sorted(table):
        ct = table[safe]["contrast"]
        if ct["ceiling_row"]:
            flag = "CEILING"
        elif table[safe].get("wiped") is None:
            flag = "[control-only]"
        elif not ct["complete"]:
            flag = "[partial]"
        else:
            flag = ""
        print(f"  {safe:42s} c_med={_fmt(ct['control_median']):>4} "
              f"w_med={_fmt(ct['wiped_median']):>4} Δ={_fmt(ct['delta_wiped_minus_control']):>5} "
              f"{'FALSIFIER✓' if ct['meets_falsifier'] else '':10} {flag}")
    print(f"\nwrote e1a_table1.{{md,json,png}} to {args.out_dir}")


if __name__ == "__main__":
    main()
