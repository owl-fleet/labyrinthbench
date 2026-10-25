"""Prove the watch pages (GET /watch, GET /watch/{session_id}) HTML-escape client-supplied
and history-file text (model, session_id, deg_id, steps, status) instead of interpolating it
raw, closing the stored-XSS path through POST /session's model field. No live server —
fastapi.testclient.TestClient drives api.main:app in-process, with HISTORY_FILE monkeypatched
to a temp path so the repo's real history file is never touched.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import api.main`

import api.main as api_main  # noqa: E402

_tmp_history = Path(tempfile.mkdtemp()) / "watch_escaping_history.json"
api_main.HISTORY_FILE = _tmp_history

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(api_main.app)

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

PAYLOAD = "<script>alert(1)</script>\"'"
ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;&quot;&#x27;"

resp = client.post("/session", json={"deg_id": "alpha-1", "model": PAYLOAD})
check("POST /session with a script-tag model string succeeds", resp.status_code == 200)
session_id = resp.json()["session_id"]
short_id = session_id[:8]

# --- GET /watch (list view) --------------------------------------------------------------
list_resp = client.get("/watch")
check("GET /watch returns 200", list_resp.status_code == 200)
check("GET /watch does not contain the raw payload", PAYLOAD not in list_resp.text)
check("GET /watch contains the escaped payload", ESCAPED in list_resp.text)
check("GET /watch contains the session's short id", short_id in list_resp.text)

# --- GET /watch/{session_id} (detail view) -----------------------------------------------
detail_resp = client.get(f"/watch/{session_id}")
check("GET /watch/{id} returns 200", detail_resp.status_code == 200)
check("GET /watch/{id} does not contain the raw payload", PAYLOAD not in detail_resp.text)
check("GET /watch/{id} contains the escaped payload", ESCAPED in detail_resp.text)
check("GET /watch/{id} contains the session's short id", short_id in detail_resp.text)

# --- A normal model name still renders unescaped (no behaviour change) -------------------
resp2 = client.post("/session", json={"deg_id": "alpha-1", "model": "qwen3:14b"})
sid2 = resp2.json()["session_id"]
list_resp2 = client.get("/watch")
check("a plain model name like 'qwen3:14b' still renders unescaped",
      "qwen3:14b" in list_resp2.text)
detail_resp2 = client.get(f"/watch/{sid2}")
check("the detail page for a plain model name also renders unescaped",
      "qwen3:14b" in detail_resp2.text)

check("the repo's real history file was never touched",
      not Path("/results/.session_history.json").exists())

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
