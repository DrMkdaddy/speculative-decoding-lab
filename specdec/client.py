"""HTTP client for llama-server's raw completion endpoint."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ClientError(RuntimeError):
    pass


def completion(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int,
    seed: int,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], float]:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": seed,
            "cache_prompt": False,
            "stream": False,
            "ignore_eos": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ClientError(f"completion failed: {exc}") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClientError(f"completion returned non-JSON: {raw[:200]}") from exc
    if "error" in payload:
        raise ClientError(f"completion error: {payload['error']}")
    return payload, wall_ms


def timing_row(payload: dict[str, Any], wall_ms: float) -> dict[str, Any]:
    timings = payload.get("timings") or {}
    prompt_n = timings.get("prompt_n") or payload.get("tokens_evaluated") or 0
    predicted_n = timings.get("predicted_n") or 0
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    first_token_ms = None
    if prompt_ms is not None and predicted_ms is not None and predicted_n:
        first_token_ms = prompt_ms + predicted_ms / predicted_n
    return {
        "wall_ms": wall_ms,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "first_token_ms": first_token_ms,
        "draft_n": timings.get("draft_n", 0),
        "draft_n_accepted": timings.get("draft_n_accepted", 0),
    }


def generated_text(payload: dict[str, Any]) -> str:
    return str(payload.get("content", ""))
