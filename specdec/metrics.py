"""Parse llama-server's Prometheus metrics.

The speculative counters are cumulative. The harness snapshots them before and
after each request and subtracts, which gives per-request acceptance instead of
a run-wide average that hides prompt-level variance.
"""

from __future__ import annotations

import re
from typing import Any


_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_POSITION = re.compile(r'position="(\d+)"')


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        metrics[f"{name}{{{labels}}}" if labels else name] = float(match.group("value"))
    return metrics


def metric(metrics: dict[str, float], name: str) -> float:
    return metrics.get(name, 0.0)


def accepted_by_position(metrics: dict[str, float]) -> dict[int, float]:
    result: dict[int, float] = {}
    prefix = "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        match = _POSITION.search(key)
        if match:
            result[int(match.group(1))] = value
    return result


def diff(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Subtract two cumulative snapshots.

    Counter resets (a server restart) would produce negatives; clamp them to
    zero so a bad snapshot is visible as a zero rather than corrupting sums.
    """
    keys = set(before) | set(after)
    return {
        key: max(0.0, after.get(key, 0.0) - before.get(key, 0.0))
        for key in keys
    }


def request_spec_metrics(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, Any]:
    delta = diff(before, after)
    accepted = accepted_by_position(delta)
    drafts = metric(delta, "llamacpp:spec_decode_num_drafts_total")
    draft_tokens = metric(delta, "llamacpp:spec_decode_num_draft_tokens_total")
    accepted_tokens = metric(delta, "llamacpp:spec_decode_num_accepted_tokens_total")
    per_position = {
        position: (count / drafts if drafts else None)
        for position, count in sorted(accepted.items())
    }
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance": (accepted_tokens / draft_tokens if draft_tokens else None),
        "mean_accepted_length": (1.0 + accepted_tokens / drafts if drafts else None),
        "accepted_by_position": {str(k): v for k, v in accepted.items()},
        "acceptance_by_position": {
            str(k): v for k, v in per_position.items() if v is not None
        },
    }
