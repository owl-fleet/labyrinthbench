"""Prove the E1a strip plot is optional: with matplotlib/numpy importable, main() writes the
md + json tables and the png exactly as before; with either import stubbed out, it writes the
same tables, prints a one-line warning naming the missing module, skips the png and exits 0.
An ImportError for any other module still propagates. No live server: synthetic JSONL cells.
"""
import sys, os, io, json, tempfile, contextlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import e1a_table1  # noqa: E402

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

def _row(depth):
    return {"ramp_depth": depth, "found_exit": depth >= e1a_table1.CEILING, "turns": 3 * depth + 1,
            "turns_log": [{"action_parsed": {"action": "observe"}},
                          {"action_parsed": {"action": "commit", "answer": "a"}}]}

def _fixture(d):
    for arm, depths in (("control", [4, 6, 5]), ("wiped", [9, 11, 10])):
        with open(os.path.join(d, f"e1a-toy-model-{arm}.jsonl"), "w") as fh:
            for x in depths:
                fh.write(json.dumps(_row(x)) + "\n")

def _run(results_dir, out_dir, block=()):
    """Run main() in-process with the named top-level modules made unimportable.
    Returns (exit_code, stdout, stderr)."""
    saved = {k: v for k, v in sys.modules.items()
             if any(k == m or k.startswith(m + ".") for m in block)}
    for k in saved:
        del sys.modules[k]
    for m in block:
        sys.modules[m] = None   # `import m` now raises ModuleNotFoundError(name=m)
    argv, out, err, code = sys.argv, io.StringIO(), io.StringIO(), 0
    sys.argv = ["e1a_table1.py", "--results-dir", results_dir, "--out-dir", out_dir]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            e1a_table1.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = argv
        for m in block:
            sys.modules.pop(m, None)
        sys.modules.update(saved)
    return code, out.getvalue(), err.getvalue()

def _read(p):
    with open(p, "rb") as fh:
        return fh.read()

with tempfile.TemporaryDirectory() as tmp:
    res = os.path.join(tmp, "res"); os.makedirs(res); _fixture(res)

    # --- plotting libraries present: unchanged behavior, png written, no warning ---
    have = os.path.join(tmp, "have")
    code, out, err = _run(res, have)
    check("present: exits 0", code == 0)
    check("present: md, json and png written",
          all(os.path.exists(os.path.join(have, f))
              for f in ("e1a_table1.md", "e1a_table1.json", "e1a_table1_stripplot.png")))
    check("present: no warning on stderr", err == "")
    check("present: summary still reads e1a_table1.{md,json,png}",
          f"wrote e1a_table1.{{md,json,png}} to {have}" in out)

    # --- matplotlib missing / numpy missing: tables intact, png skipped, one-line warning ---
    for mod in ("matplotlib", "numpy"):
        miss = os.path.join(tmp, f"no-{mod}")
        code, out, err = _run(res, miss, block=(mod,))
        check(f"no {mod}: exits 0", code == 0)
        check(f"no {mod}: md and json byte-identical to the present-path tables",
              all(_read(os.path.join(miss, f)) == _read(os.path.join(have, f))
                  for f in ("e1a_table1.md", "e1a_table1.json")))
        check(f"no {mod}: png not written",
              not os.path.exists(os.path.join(miss, "e1a_table1_stripplot.png")))
        lines = err.strip().splitlines()
        check(f"no {mod}: exactly one warning line, naming {mod}",
              len(lines) == 1 and lines[0].startswith("warning:") and mod in lines[0])
        check(f"no {mod}: summary reads e1a_table1.{{md,json}}",
              f"wrote e1a_table1.{{md,json}} to {miss}" in out)

    # --- an ImportError for some other module is a real bug and still propagates ---
    real = e1a_table1.render_stripplot
    def _boom(*a, **k):
        raise ModuleNotFoundError("No module named 'not_a_plot_dep'", name="not_a_plot_dep")
    e1a_table1.render_stripplot = _boom
    try:
        _run(res, os.path.join(tmp, "other"))
        check("unrelated ImportError propagates", False)
    except ModuleNotFoundError as e:
        check("unrelated ImportError propagates", e.name == "not_a_plot_dep")
    finally:
        e1a_table1.render_stripplot = real

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAIL(S)'}")
sys.exit(1 if FAILS else 0)
