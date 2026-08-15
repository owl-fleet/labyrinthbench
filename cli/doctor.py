#!/usr/bin/env python3
"""lb doctor — token-free preflight for a run: endpoint, model tag, digest, context length.

Checks everything that silently DNFs or mis-provenances a run, before any tokens are spent:

  1. The OpenAI-compatible endpoint answers GET /models (Ollama, LM Studio, llama.cpp).
  2. The model tag you're about to pass exists there — an unknown tag runs the whole session
     as instant DNFs (README FAQ), so catch it here.
  3. Ollama only: the native API's digest for the tag — record THIS in your entry; tags are
     mutable, the digest is the datum (METHODOLOGY §1) — and the model's trained context
     length from /api/show.
  4. The num_ctx trap: ollama's /v1 endpoint silently DROPS the num_ctx option — a run can
     believe it has 32k of context while the serving slot has 4k. Doctor can't read the live
     slot (only the server journal shows it); it prints the verification recipe instead.
  5. The serving stack that will actually answer: engine, build, weights, quantization and the
     CHAT TEMPLATE HASH — the tuple run_eval.py stamps onto every row (see provenance.py). An
     unexpected value here means the run you are about to pay for is not comparable to the last one.
  6. The maze API, if one is up: /eval-status — refuse to preflight into a host mid-eval.

Read-only by design: no completion request, no state change, zero tokens.

Usage:
  python3 cli/doctor.py --model qwen3:14b [--base-url http://localhost:11434/v1]
                        [--maze-url http://localhost:8090]
Exit codes: 0 = ready; 1 = at least one check failed; 2 = usage error.
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import urlparse

import httpx

try:
    from cli import provenance
except ImportError:
    import provenance

OK, WARN, FAIL = "  ✓", "  !", "  ✗"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model tag you intend to pass to run_eval.py")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--maze-url", default="http://localhost:8090")
    args = ap.parse_args()

    failed = False
    client = httpx.Client(timeout=10)

    # 1. OpenAI-compatible endpoint
    models: list[str] = []
    try:
        r = client.get(f"{args.base_url.rstrip('/')}/models")
        r.raise_for_status()
        models = [m.get("id", "") for m in r.json().get("data", [])]
        print(f"{OK} endpoint up: {args.base_url} ({len(models)} models served)")
    except Exception as e:
        print(f"{FAIL} endpoint unreachable: {args.base_url} — {e}")
        print("      is the server running? Ollama :11434, LM Studio :1234, llama.cpp :8080")
        return 1

    # 2. Model tag present
    if args.model in models:
        print(f"{OK} model tag found: {args.model}")
    else:
        # LM Studio serves path-style ids; be helpful about near-misses before failing.
        near = [m for m in models if args.model.lower() in m.lower()]
        print(f"{FAIL} model tag NOT served: {args.model!r} — a run with it would be instant DNFs")
        if near:
            print(f"      close matches: {', '.join(near[:5])}")
        elif models:
            print(f"      served tags include: {', '.join(models[:5])}")
        failed = True

    # 3./4. Ollama native API: digest provenance + context length + the num_ctx trap
    parsed = urlparse(args.base_url)
    native_root = f"{parsed.scheme}://{parsed.netloc}"
    is_ollama = False
    try:
        r = client.get(f"{native_root}/api/tags")
        r.raise_for_status()
        is_ollama = True
        digests = {m["name"]: m.get("digest", "") for m in r.json().get("models", [])}
        digest = digests.get(args.model) or digests.get(f"{args.model}:latest")
        if digest:
            print(f"{OK} ollama digest for {args.model}: {digest}")
            print("      record this in your entry — tags are mutable, the digest is the datum")
        elif args.model in models:
            print(f"{WARN} tag served on /v1 but absent from /api/tags — digest not resolvable")
    except Exception:
        print(f"{OK} not an ollama native API (LM Studio / llama.cpp) — no tag digest here;"
              " the serving-stack line below carries this server's weights identity")

    if is_ollama:
        try:
            r = client.post(f"{native_root}/api/show", json={"model": args.model})
            r.raise_for_status()
            info = r.json().get("model_info", {})
            ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
            if ctx:
                print(f"{OK} trained context length: {ctx}")
        except Exception:
            pass
        print(f"{WARN} num_ctx trap: ollama's /v1 endpoint silently DROPS the num_ctx option.")
        print("      The only ground truth is the serving journal — after your first turn, run:")
        print("        journalctl -u ollama | grep n_ctx_slot   # on the serving host")
        print("      and pass the value to run_eval.py --n-ctx-slot for the run's provenance row.")

    # 4b. Serving-stack provenance — the tuple run_eval.py will stamp onto every row. Printed here
    # because preflight is when you can still act on it: a template_sha or engine build you did not
    # expect means the run you are about to spend tokens on is not comparable to the last one.
    try:
        prov = provenance.capture(args.base_url, args.model, timeout=8.0)
        print(f"{OK} serving stack: {provenance.summary(prov)[len('[prov] '):]}")
        if not prov.get("template_sha"):
            print(f"{WARN} chat template not readable here — the run's prompt bytes are unrecorded")
    except Exception as e:
        print(f"{WARN} provenance probe failed (non-fatal): {e}")

    # 5. Maze API busy-check
    try:
        r = client.get(f"{args.maze_url.rstrip('/')}/eval-status", timeout=3)
        r.raise_for_status()
        status = r.json()
        if status.get("running"):
            print(f"{FAIL} maze API at {args.maze_url} has an eval RUNNING — do not start another")
            failed = True
        else:
            print(f"{OK} maze API idle: {args.maze_url}")
    except Exception:
        print(f"{WARN} maze API not reachable at {args.maze_url} — fine if you haven't started it yet"
              " (docker compose -f docker-compose.standalone.yml up)")

    print("\nready to run" if not failed else "\nNOT ready — fix the ✗ items above")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
