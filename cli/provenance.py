"""Record WHAT SERVED a run, not just which model name was asked for.

A model tag is nowhere near enough to call two runs comparable. At minimum these vary
independently, and every one of them moves output:

    node · engine · engine build · weights digest · quantization · CHAT TEMPLATE ·
    sampling params · effective context · thinking regime

The chat template is the one people miss, and the one that moves results hardest. Identical
weights served by two runtimes receive DIFFERENT PROMPT BYTES: ollama compiles its own renderers
into the binary and substitutes them for the publisher's embedded GGUF template, so the formatting
is versioned with the ENGINE, not with the model. Upgrading the engine can therefore change results
with no change to any model — measured on the authors' hardware 2026-08-15, where a benchmark score
moved 1.000 -> 0.188 -> 1.000 across an engine upgrade and rollback on a FIXED model digest.
Diagnosing it took hours, because the run records stored `model` and little else.

This module is PURELY ADDITIVE. It issues read-only requests and returns a dict; it never changes
what is sent to the model, so it cannot alter results and is safe to add mid-campaign. Every field
is best-effort: a probe that fails records None rather than raising. **Failing to describe a run
must never fail the run.**

Supports the three servers LabyrinthBench is run against, and degrades gracefully on anything else:

  llama.cpp   /props        build, GGUF path, quant, the exact chat template, sampling defaults
  ollama      /api/show     digest, quantization, params, and the RENDERER/PARSER override
  other       /v1/models    served id only (LM Studio, vLLM, cloud gateways)
"""
from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import httpx

_TIMEOUT = 8.0

# The sampler fields that actually change output. llama.cpp's /props returns ~30 keys, several
# constant per build; recording all of them on every row is bloat that hides the moving parts.
_SAMPLING_KEYS = (
    "seed", "temperature", "top_k", "top_p", "min_p", "typical_p", "top_n_sigma",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty", "samplers",
)


def _sha(text) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    return hashlib.sha256(text.encode("utf8", "replace")).hexdigest()[:16]


def _sampling(params) -> dict | None:
    if not isinstance(params, dict):
        return None
    return {k: params[k] for k in _SAMPLING_KEYS if k in params} or None


def _gguf_id(model_path, models_doc) -> str | None:
    """Weights identity for llama.cpp: GGUF filename + EXACT byte size.

    Not a content hash — a byte-exact sha256 needs the file itself, and this probe only has HTTP.
    The pair is decisive in practice: a re-quantized or re-downloaded GGUF landing at the same path
    at a byte-identical size is not a case that occurs. The value is namespaced (`gguf:`) so it can
    never be confused with the `sha256:` digest recorded for ollama — a run record has to say WHICH
    kind of identity it carries, or comparing two of them is guesswork.
    """
    if not isinstance(model_path, str) or not model_path:
        return None
    name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
    size = None
    if isinstance(models_doc, dict):
        for m in models_doc.get("data") or []:
            meta = m.get("meta") if isinstance(m, dict) else None
            if isinstance(meta, dict) and meta.get("size"):
                size = meta["size"]
                break
    return f"gguf:{name}:{size}" if size else f"gguf:{name}"


def capture(base_url: str, model: str, api_key: str | None = None,
            timeout: float = _TIMEOUT) -> dict:
    """Best-effort identity tuple for `model` as served at `base_url` (an OpenAI /v1 URL).

    Unlike a gateway-fronted setup, the endpoint being probed is exactly the one the run talks to,
    so there is no guessing about which host owns the model.
    """
    base = (base_url or "").rstrip("/")
    parsed = urlparse(base)
    native_root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme else base

    prov: dict = {
        "base_url": base,
        "model": model,
        "node": parsed.hostname,
        "engine": None,
        "engine_build": None,
        "weights_digest": None,
        "quantization": None,
        "template_source": None,   # "gguf-jinja" | "ollama-renderer" | "ollama-template"
        "template_sha": None,
        "renderer": None,          # ollama only: the compiled substitutes, versioned with the binary
        "parser": None,
        "sampling": None,
        "n_ctx": None,
    }

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    client = httpx.Client(timeout=timeout, headers=headers)
    try:
        # --- llama.cpp first: it also serves /v1/models, but only it answers /props ---
        props = _json(client, "GET", f"{native_root}/props")
        if isinstance(props, dict) and ("build_info" in props or "model_path" in props):
            prov["engine"] = "llama.cpp"
            prov["engine_build"] = props.get("build_info") or props.get("system_fingerprint")
            gen = props.get("default_generation_settings") or {}
            prov["n_ctx"] = gen.get("n_ctx") or props.get("n_ctx")
            prov["quantization"] = props.get("model_ftype")
            prov["sampling"] = _sampling(gen.get("params"))
            prov["weights_digest"] = _gguf_id(
                props.get("model_path"), _json(client, "GET", f"{base}/models")
            )
            tmpl = props.get("chat_template")
            if tmpl:
                # llama-server reports the template it will ACTUALLY render with: the GGUF's
                # embedded Jinja under --jinja, or the file passed to --chat-template-file.
                prov["template_source"] = "gguf-jinja"
                prov["template_sha"] = _sha(tmpl)
            return prov

        # --- ollama: /api/show carries digest, quant, params, and the renderer override ---
        tags = _json(client, "GET", f"{native_root}/api/tags")
        if isinstance(tags, dict) and "models" in tags:
            prov["engine"] = "ollama"
            ver = _json(client, "GET", f"{native_root}/api/version")
            if isinstance(ver, dict):
                prov["engine_build"] = ver.get("version")
            for m in tags.get("models") or []:
                if m.get("name") in (model, f"{model}:latest") or m.get("model") == model:
                    d = (m.get("digest") or "")[:12]
                    prov["weights_digest"] = f"sha256:{d}" if d else None
                    break
            show = _json(client, "POST", f"{native_root}/api/show", json={"model": model})
            if isinstance(show, dict):
                prov["quantization"] = (show.get("details") or {}).get("quantization_level")
                prov["sampling"] = show.get("parameters")
                prov["template_sha"] = _sha(show.get("template"))
                # A Go template of just "{{ .Prompt }}" is a PASSTHROUGH STUB: the real formatting
                # is done by a compiled renderer named in the modelfile and versioned with the
                # ollama BINARY, not the model. That is the invisible variable this file exists for.
                for line in (show.get("modelfile") or "").splitlines():
                    if line.startswith("RENDERER "):
                        prov["renderer"] = line.split(None, 1)[1].strip()
                    elif line.startswith("PARSER "):
                        prov["parser"] = line.split(None, 1)[1].strip()
                prov["template_source"] = (
                    "ollama-renderer" if prov["renderer"] else "ollama-template"
                )
            return prov

        # --- anything else that speaks OpenAI: record what little is on offer ---
        served = _json(client, "GET", f"{base}/models")
        if isinstance(served, dict) and served.get("data"):
            prov["engine"] = "openai-compatible"
        return prov
    finally:
        client.close()


def _json(client: httpx.Client, method: str, url: str, **kw):
    """Read-only probe that swallows everything. A probe failure must never fail a run."""
    try:
        r = client.request(method, url, **kw)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def summary(prov: dict) -> str:
    """One-line human summary for the harness log."""
    bits = [
        f"node={prov.get('node')}",
        f"engine={prov.get('engine')}/{prov.get('engine_build')}",
        f"weights={prov.get('weights_digest')}",
        f"quant={prov.get('quantization')}",
        f"tmpl={prov.get('template_source')}:{prov.get('template_sha')}",
    ]
    if prov.get("renderer"):
        bits.append(f"renderer={prov['renderer']}")
    if prov.get("n_ctx"):
        bits.append(f"n_ctx={prov['n_ctx']}")
    return "[prov] " + " ".join(str(b) for b in bits)
