"""Prove GET /eval-status never mutates the lock directory it reports on.

Before this test, eval_status() unlinked a lock whenever os.kill(pid, 0) raised
ProcessLookupError — correct only when the API shares a PID namespace with the writer.
A run launched from the host, or from a sibling container on the same /results volume,
has a pid the API can't see, so a plain status poll erased a LIVE run's lock and answered
running: false. Liveness now falls back to the lock file's own mtime (rewritten every
turn by cli/run_eval.py's _update_heartbeat) whenever the pid probe says "not in this
namespace". No live server: FastAPI TestClient against api.main:app, lock dir monkeypatched
to a temp dir, no eval ever actually started.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402
import api.main as main  # noqa: E402

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

DEAD_PID = 2**22 + 7  # astronomically unlikely to be a live pid in any namespace

tmpdir = tempfile.TemporaryDirectory()
main.EVAL_LOCK_DIR = Path(tmpdir.name)
client = TestClient(main.app)


def write_lock(name, data=None, raw=None, mtime_offset_s=0):
    p = main.EVAL_LOCK_DIR / name
    p.write_text(raw if raw is not None else json.dumps(data))
    if mtime_offset_s:
        now = time.time()
        os.utime(p, (now - mtime_offset_s, now - mtime_offset_s))
    return p

# --- (d) no locks at all: shape unchanged ---
resp = client.get("/eval-status")
check("no locks -> 200", resp.status_code == 200)
check("no locks -> {'running': False} shape unchanged", resp.json() == {"running": False})

# --- (a) dead pid, fresh mtime -> running (heartbeat fallback saves it) ---
lock_a = write_lock(".eval_lock_hosta", {
    "model": "m", "deg": "d", "runs": 1, "base_url": "http://hosta/v1",
    "pid": DEAD_PID, "started": "x", "last_heartbeat": "x",
})
body = client.get("/eval-status").json()
check("(a) dead pid + fresh mtime -> running: true", body["running"] is True)
check("(a) machine status is 'running'", body["machines"]["http://hosta/v1"]["status"] == "running")
check("(a) lock file untouched by a GET", lock_a.exists())

# --- (b) same lock, mtime pushed 900s back -> stale ---
os.utime(lock_a, (time.time() - 900, time.time() - 900))
body = client.get("/eval-status").json()
check("(b) dead pid + stale mtime -> running: false", body["running"] is False)
check("(b) machine status is 'stale'", body["machines"]["http://hosta/v1"]["status"] == "stale")
check("(b) lock file still exists after the GET", lock_a.exists())
lock_a.unlink()

# --- (c) unparseable lock contents -> unreadable, counts as running (fail-safe) ---
lock_c = write_lock(".eval_lock_hostc", raw="{not json")
body = client.get("/eval-status").json()
check("(c) malformed JSON -> running: true", body["running"] is True)
check("(c) machine status is 'unreadable'", body["machines"][".eval_lock_hostc"]["status"] == "unreadable")
check("(c) lock file untouched", lock_c.exists())
lock_c.unlink()

# --- (e) after every call above, every lock file we created is still on disk ---
# (already asserted inline above via .exists(); re-assert on fresh locks for both paths)
lock_live = write_lock(".eval_lock_live", {
    "model": "m", "deg": "d", "runs": 1, "base_url": "http://live/v1",
    "pid": DEAD_PID, "started": "x", "last_heartbeat": "x",
})
lock_bad = write_lock(".eval_lock_bad", raw="{broken")
for _ in range(3):
    client.get("/eval-status")
check("(e) a running-status lock survives repeated GETs", lock_live.exists())
check("(e) an unreadable lock survives repeated GETs", lock_bad.exists())

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
