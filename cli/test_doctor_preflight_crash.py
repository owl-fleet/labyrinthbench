"""Runner v2: cli/doctor.py is the standalone, human-run twin of the same "FAIL closed on a dead
upstream" contract (its own docstring: "lb doctor — token-free preflight for a run"). It referenced
os.environ.get(...) for its --base-url default with `os` never imported, so EVERY invocation raised
NameError before check 1 (the endpoint-reachability probe) ever ran — the preflight tool itself was
the thing that was broken, regardless of whether the upstream was actually up or down. This proves
doctor.main() now runs past argument parsing and fails closed (non-zero, a clear printed reason)
against an unreachable endpoint, using a closed local port so no real upstream is needed and the
connection-refused response is near-instant. No live model server; read-only per doctor.py's own
contract (no completion request)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import doctor  # noqa: E402

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

# Port 1 on loopback: nothing listens there, so the OS refuses the connection immediately —
# exercises the SAME "unreachable" code path a truly dead ollama/llama.cpp would, with no timeout
# wait and no dependency on anything actually being up on this host.
sys.argv = ["doctor.py", "--model", "does-not-matter", "--base-url", "http://127.0.0.1:1/v1"]

rc = None
name_error = None
try:
    rc = doctor.main()
except NameError as e:
    name_error = e

check(f"doctor.main() runs past argument parsing without NameError "
      f"(was: `os` referenced in an argparse default with no `import os`)"
      + (f" — got {name_error!r}" if name_error else ""),
      name_error is None)

if name_error is None:
    check("doctor.main() FAILS CLOSED (non-zero exit) against an unreachable endpoint, per its own "
          "documented exit codes (0=ready, 1=at least one check failed)", rc == 1)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
