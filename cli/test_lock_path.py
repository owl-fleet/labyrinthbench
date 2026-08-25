"""Prove the VRAM run-lock key follows --lock-host when given, and falls back to the
--base-url hostname (today's behavior, byte-for-byte) when it isn't — confined-effector-
gateway chunk 08. No live server: pure path/file logic against a real /results dir.
"""
import sys, os, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import run_eval  # noqa: E402

os.makedirs("/results", exist_ok=True)

FAILS = 0
def check(d, ok):
    global FAILS; FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {d}")

def _cleanup(*paths):
    for p in paths:
        p.unlink(missing_ok=True)

# --- _lock_path: no lock_host -> derives from base_url's own hostname (unchanged default) ---
p1 = run_eval._lock_path("http://192.168.0.10:11435/v1")
check("no lock_host: keys on base_url's hostname",
      p1.name == ".eval_lock_192_168_0_10")

# --- _lock_path: lock_host given -> overrides the base_url-derived host entirely ---
p2 = run_eval._lock_path("http://192.168.0.10:11435/v1", lock_host="framework")
check("lock_host='framework' overrides the gateway hostname",
      p2.name == ".eval_lock_framework")

# --- Two models routed through the SAME gateway but different --lock-host get DIFFERENT
#     lock files, i.e. locking no longer over-serializes once the operator disambiguates ---
p_fw = run_eval._lock_path("http://192.168.0.10:11435/v1", lock_host="framework")
p_ws = run_eval._lock_path("http://192.168.0.10:11435/v1", lock_host="workstation")
check("different --lock-host values under the same gateway base_url don't collide",
      p_fw != p_ws)

# --- Without --lock-host, the SAME gateway base_url collides regardless of model (the
#     documented conservative fallback: safe, but over-serializes) ---
p_a = run_eval._lock_path("http://192.168.0.10:11435/v1")
p_b = run_eval._lock_path("http://192.168.0.10:11435/v1")
check("no --lock-host: same gateway base_url always collides (documented, conservative)",
      p_a == p_b)

# --- _acquire_lock / _update_heartbeat / _release_lock round-trip on the lock_host path ---
_cleanup(run_eval._lock_path("http://192.168.0.10:11435/v1", "framework"))
run_eval._acquire_lock("qwen3:32b", "alpha-1", 1, "http://192.168.0.10:11435/v1", "framework")
lock_file = run_eval._lock_path("http://192.168.0.10:11435/v1", "framework")
check("_acquire_lock writes the lock_host-keyed file, not the gateway-hostname one",
      lock_file.exists())
gateway_keyed = run_eval._lock_path("http://192.168.0.10:11435/v1")
check("_acquire_lock with lock_host does NOT also write the gateway-hostname-keyed file",
      not gateway_keyed.exists())

run_eval._update_heartbeat("http://192.168.0.10:11435/v1", "framework")
data = json.loads(lock_file.read_text())
check("_update_heartbeat touched the correct (lock_host-keyed) file",
      "last_heartbeat" in data)

run_eval._release_lock("http://192.168.0.10:11435/v1", "framework")
check("_release_lock removes the lock_host-keyed file", not lock_file.exists())

# --- A direct (non-gateway) base_url with no --lock-host is completely unaffected ---
_cleanup(run_eval._lock_path("http://192.168.0.11:11434/v1"))
run_eval._acquire_lock("qwen3:14b", "alpha-1", 1, "http://192.168.0.11:11434/v1")
direct_lock = run_eval._lock_path("http://192.168.0.11:11434/v1")
check("direct base_url + no --lock-host: unchanged legacy behavior",
      direct_lock.exists() and direct_lock.name == ".eval_lock_192_168_0_11")
run_eval._release_lock("http://192.168.0.11:11434/v1")

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
