"""Prove scripts/pre-push-scan.sh tells a tooling failure (gitleaks image pull refused,
container can't start, gitleaks itself erroring) apart from a real leak finding — a
docker/gitleaks failure must never read as "leaks detected". No live server: the script
runs standalone against a temp (non-git) directory, with a fake `docker` on PATH driven
by env vars to simulate each case, and only real exit codes/output are inspected.
"""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_SRC = REPO_ROOT / "scripts" / "pre-push-scan.sh"

FAILS = 0


def check(desc, ok):
    global FAILS
    FAILS += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")


FAKE_DOCKER = """#!/usr/bin/env bash
# Fake docker for cli/test_pre_push_scan.py. Behavior is driven entirely by env vars
# so one script covers every scenario the test exercises:
#   FAKE_DOCKER_INFO_EXIT     exit code for `docker info`            (default 0)
#   FAKE_DOCKER_INSPECT_EXIT  exit code for `docker image inspect`   (default 0)
#   FAKE_DOCKER_PULL_EXIT     exit code for `docker pull`            (default 0)
#   FAKE_DOCKER_RUN_EXIT      exit code for `docker run`             (default 0)
case "$1" in
    info)  exit "${FAKE_DOCKER_INFO_EXIT:-0}" ;;
    image) exit "${FAKE_DOCKER_INSPECT_EXIT:-0}" ;;
    pull)  exit "${FAKE_DOCKER_PULL_EXIT:-0}" ;;
    run)   exit "${FAKE_DOCKER_RUN_EXIT:-0}" ;;
    *)     exit 0 ;;
esac
"""


def make_workdir(tmp_root):
    """A plain (non-git) tree: scripts/pre-push-scan.sh copied in, plus one clean file
    so the pattern battery always passes and only the gitleaks branch is under test."""
    workdir = Path(tmp_root) / "work"
    (workdir / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT_SRC, workdir / "scripts" / "pre-push-scan.sh")
    os.chmod(workdir / "scripts" / "pre-push-scan.sh", 0o755)
    (workdir / "README.md").write_text("nothing sensitive here\n")
    return workdir


def make_fake_docker_dir(tmp_root):
    bin_dir = Path(tmp_root) / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    docker_path = bin_dir / "docker"
    docker_path.write_text(FAKE_DOCKER)
    st = os.stat(docker_path)
    os.chmod(docker_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def path_without_real_docker(tmp_root):
    """The current PATH with `docker` itself made unreachable, so a scenario can
    exercise "docker truly missing" (not just unusable) without losing every other
    tool (bash/grep/awk/sed/find/mktemp/...) that may share docker's directory: the
    directory holding the real `docker` binary is replaced by a symlink farm of
    everything in it EXCEPT `docker`."""
    real_docker = shutil.which("docker")
    dirs = os.environ.get("PATH", "").split(os.pathsep)
    if not real_docker:
        return os.pathsep.join(dirs)
    docker_dir = Path(real_docker).resolve().parent
    sanitized = Path(tmp_root) / "no-docker-bin"
    sanitized.mkdir(exist_ok=True)
    for entry in docker_dir.iterdir():
        if entry.name == "docker":
            continue
        link = sanitized / entry.name
        if not link.exists():
            try:
                link.symlink_to(entry)
            except OSError:
                pass
    new_dirs = []
    for d in dirs:
        if not d:
            continue
        new_dirs.append(str(sanitized) if Path(d).resolve() == docker_dir else d)
    return os.pathsep.join(new_dirs)


def run_scan(workdir, extra_path_dir=None, base_path=None, args=(), env_overrides=None):
    env = dict(os.environ)
    base = base_path if base_path is not None else os.environ.get("PATH", "")
    if extra_path_dir is not None:
        env["PATH"] = f"{extra_path_dir}{os.pathsep}{base}"
    else:
        env["PATH"] = base
    for k in ("FAKE_DOCKER_INFO_EXIT", "FAKE_DOCKER_INSPECT_EXIT",
              "FAKE_DOCKER_PULL_EXIT", "FAKE_DOCKER_RUN_EXIT"):
        env.pop(k, None)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        ["bash", "scripts/pre-push-scan.sh", *args],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc


with tempfile.TemporaryDirectory(prefix="lb-pre-push-scan-test-") as tmp_root:
    workdir = make_workdir(tmp_root)
    fake_bin = make_fake_docker_dir(tmp_root)
    no_docker_path = path_without_real_docker(tmp_root)

    # --- image pull failure: docker up, image absent locally, pull refused ---
    p = run_scan(workdir, extra_path_dir=fake_bin, env_overrides={
        "FAKE_DOCKER_INFO_EXIT": "0",
        "FAKE_DOCKER_INSPECT_EXIT": "1",
        "FAKE_DOCKER_PULL_EXIT": "1",
    })
    check("image pull failure -> exit 2 (ERROR, not a leak verdict)", p.returncode == 2)
    check("image pull failure -> GITLEAKS_STATUS: ERROR reported", "gitleaks: ERROR" in p.stdout)
    check("image pull failure -> never printed as a leak finding",
          "leaks detected" not in p.stdout)

    # --- gitleaks finds a real leak (container's own exit-code-1 contract) ---
    p = run_scan(workdir, extra_path_dir=fake_bin, env_overrides={
        "FAKE_DOCKER_INFO_EXIT": "0",
        "FAKE_DOCKER_INSPECT_EXIT": "0",
        "FAKE_DOCKER_RUN_EXIT": "1",
    })
    check("gitleaks finding -> exit 1", p.returncode == 1)
    check("gitleaks finding -> [FAIL] leaks detected reported", "leaks detected" in p.stdout)

    # --- gitleaks clean ---
    p = run_scan(workdir, extra_path_dir=fake_bin, env_overrides={
        "FAKE_DOCKER_INFO_EXIT": "0",
        "FAKE_DOCKER_INSPECT_EXIT": "0",
        "FAKE_DOCKER_RUN_EXIT": "0",
    })
    check("gitleaks clean -> exit 0", p.returncode == 0)
    check("gitleaks clean -> [PASS] reported", "gitleaks: PASS" in p.stdout)

    # --- docker missing entirely (no docker on PATH at all) ---
    p = run_scan(workdir, base_path=no_docker_path)
    check("docker missing, no --require-gitleaks -> exit 0 (warned, not failed)",
          p.returncode == 0)
    check("docker missing -> warning banner printed", "SKIPPED" in p.stdout)

    p = run_scan(workdir, base_path=no_docker_path, args=["--require-gitleaks"])
    check("docker missing, --require-gitleaks -> exit 2", p.returncode == 2)
    check("docker missing, --require-gitleaks -> required-but-unavailable message",
          "required but unavailable" in p.stdout)

    # --- gitleaks itself exits with an internal-error code (neither 0 nor 1) ---
    p = run_scan(workdir, extra_path_dir=fake_bin, env_overrides={
        "FAKE_DOCKER_INFO_EXIT": "0",
        "FAKE_DOCKER_INSPECT_EXIT": "0",
        "FAKE_DOCKER_RUN_EXIT": "2",
    })
    check("gitleaks exits 2 (internal error) -> exit 2, not read as a finding",
          p.returncode == 2)
    check("gitleaks exits 2 -> never printed as a leak finding",
          "leaks detected" not in p.stdout)

    # --- design note: a real pattern-battery hit never echoes the matched value ---
    # Built by concatenation, not a contiguous literal, so THIS source file doesn't
    # itself trip the repo's own api-key-shapes pattern when pre-push-scan.sh scans
    # the tree (the same reason the target script bracket-tricks its own patterns).
    fake_secret = "sk-" + "abcdefghijklmnopqrstuvwx"
    secret_file = workdir / "config.txt"
    secret_file.write_text(f"token = {fake_secret}\n")
    p = run_scan(workdir, extra_path_dir=fake_bin, env_overrides={
        "FAKE_DOCKER_INFO_EXIT": "0",
        "FAKE_DOCKER_INSPECT_EXIT": "0",
        "FAKE_DOCKER_RUN_EXIT": "0",
    })
    secret_file.unlink()
    check("pattern-battery hit -> category reported", "api-key-shapes" in p.stdout)
    check("pattern-battery hit -> matched secret value never printed",
          fake_secret not in p.stdout)
    check("pattern-battery hit alone (gitleaks clean) -> exit 1", p.returncode == 1)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
