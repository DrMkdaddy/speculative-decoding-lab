"""Turn per-request rows into acceptance, break-even, and router tables."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _baseline_index(data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    index = {}
    for row in data["rows"]:
        if not row["speculative"]:
            index[(row["prompt_id"], row["run"])] = row
    return index


def _configs(data: dict[str, Any]) -> list[str]:
    seen = []
    for row in data["rows"]:
        if row["config"] not in seen:
            seen.append(row["config"])
    return seen


def _spec_configs(data: dict[str, Any]) -> list[str]:
    return [c for c in _configs(data) if any(
        r["config"] == c and r["speculative"] for r in data["rows"]
    )]


def _speedup(row: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    base_tps = baseline["timings"].get("predicted_per_second")
    spec_tps = row["timings"].get("predicted_per_second")
    if not base_tps or not spec_tps:
        return None
    return spec_tps / base_tps


def _acceptance(row: dict[str, Any]) -> float | None:
    value = row["spec"].get("acceptance")
    if value is not None:
        return value
    draft_n = row["timings"].get("draft_n") or 0
    accepted = row["timings"].get("draft_n_accepted") or 0
    return accepted / draft_n if draft_n else None


def summarize_config(data: dict[str, Any], config: str) -> dict[str, Any]:
    baseline = _baseline_index(data)
    rows = [r for r in data["rows"] if r["config"] == config]
    spec_rows = [r for r in rows if r["speculative"]]
    speedups = []
    acceptances = []
    first_token = []
    decode_tps = []
    exact = 0
    compared = 0
    prefix = []
    for row in rows:
        b = baseline.get((row["prompt_id"], row["run"]))
        if b is not None:
            s = _speedup(row, b)
            if s is not None:
                speedups.append(s)
            if row["speculative"]:
                compared += 1
                text = row.get("text", "")
                base_text = b.get("text", "")
                if text == base_text:
                    exact += 1
                n = min(len(text), len(base_text))
                i = 0
                while i < n and text[i] == base_text[i]:
                    i += 1
                prefix.append(i / max(1, n))
        if row["speculative"]:
            a = _acceptance(row)
            if a is not None:
                acceptances.append(a)
        ft = row["timings"].get("first_token_ms")
        if ft is not None:
            first_token.append(ft)
        tps = row["timings"].get("predicted_per_second")
        if tps is not None:
            decode_tps.append(tps)
    drafts = sum((r["spec"].get("drafts") or 0) for r in spec_rows)
    draft_tokens = sum((r["spec"].get("draft_tokens") or 0) for r in spec_rows)
    accepted = sum((r["spec"].get("accepted_tokens") or 0) for r in spec_rows)
    server = data["servers"].get(config, {})
    return {
        "config": config,
        "requests": len(rows),
        "speculative": bool(spec_rows),
        "decode_tps_mean": _mean(decode_tps),
        "speedup_mean": _mean(speedups),
        "speedup_median": _median(speedups),
        "acceptance_mean": _mean(acceptances),
        "acceptance_pooled": accepted / draft_tokens if draft_tokens else None,
        "mean_accepted_length": 1.0 + accepted / drafts if drafts else None,
        "first_token_ms_median": _median(first_token),
        "vram_idle_bytes": server.get("vram_idle_bytes"),
        "vram_peak_bytes": server.get("vram_peak_bytes"),
        "vram_delta_bytes": server.get("vram_delta_bytes"),
        "outputs_compared": compared,
        "outputs_exact": exact,
        "output_exact_rate": exact / compared if compared else None,
        "prefix_agreement_mean": _mean(prefix),
        "output_mismatches": sum(
            1 for r in rows if r.get("matches_baseline") is False
        ),
    }


def acceptance_by_position(data: dict[str, Any], config: str) -> dict[int, float | None]:
    rows = [
        r for r in data["rows"] if r["config"] == config and r["speculative"]
    ]
    drafts = sum((r["spec"].get("drafts") or 0) for r in rows)
    accepted: dict[int, float] = defaultdict(float)
    for row in rows:
        for pos, value in row["spec"].get("accepted_by_position", {}).items():
            accepted[int(pos)] += value
    return {
        pos: (value / drafts if drafts else None)
        for pos, value in sorted(accepted.items())
    }


def per_class(data: dict[str, Any], config: str) -> dict[str, dict[str, Any]]:
    baseline = _baseline_index(data)
    rows = [r for r in data["rows"] if r["config"] == config]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_class"]].append(row)
    result = {}
    for cls, cls_rows in grouped.items():
        speedups = []
        acceptances = []
        for row in cls_rows:
            b = baseline.get((row["prompt_id"], row["run"]))
            if b is not None:
                s = _speedup(row, b)
                if s is not None:
                    speedups.append(s)
            a = _acceptance(row)
            if a is not None:
                acceptances.append(a)
        result[cls] = {
            "speedup_mean": _mean(speedups),
            "acceptance_mean": _mean(acceptances),
            "requests": len(cls_rows),
        }
    return result


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float] | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    crossing = (1.0 - intercept) / slope if slope != 0 else None
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "acceptance_at_break_even": crossing,
        "n": n,
    }


def break_even(data: dict[str, Any]) -> dict[str, dict[str, float] | None]:
    baseline = _baseline_index(data)
    result = {}
    for config in _spec_configs(data):
        xs = []
        ys = []
        for row in data["rows"]:
            if row["config"] != config:
                continue
            a = _acceptance(row)
            b = baseline.get((row["prompt_id"], row["run"]))
            if a is None or b is None:
                continue
            s = _speedup(row, b)
            if s is None:
                continue
            xs.append(a)
            ys.append(s)
        result[config] = linear_fit(xs, ys)
    return result


def _predicted_ms(row: dict[str, Any]) -> float:
    value = row["timings"].get("predicted_ms")
    if value is None:
        return row["wall_ms"]
    return float(value)


def _best_fixed_spec(data: dict[str, Any]) -> str:
    baseline = _baseline_index(data)
    totals = {}
    for config in _spec_configs(data):
        total = 0.0
        for row in data["rows"]:
            if row["config"] != config:
                continue
            b = baseline.get((row["prompt_id"], row["run"]))
            if b is not None:
                total += _predicted_ms(row)
        totals[config] = total
    return min(totals, key=totals.get)


def router_replay(data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate per-class config selection and an online EWMA policy.

    The per-class policy is leave-one-prompt-out: when scoring a prompt, the
    class mean is computed from the other prompt ids in that class, so the
    prompt being scored never sets its own threshold, and the class can choose
    the draft depth with the best measured speedup. The EWMA policy walks a
    fixed shuffled order and updates after each request.
    """
    baseline = _baseline_index(data)
    spec_configs = _spec_configs(data)
    spec_rows = {
        config: {
            (r["prompt_id"], r["run"]): r
            for r in data["rows"] if r["config"] == config
        }
        for config in spec_configs
    }
    baseline_rows = {
        (r["prompt_id"], r["run"]): r for r in data["rows"] if not r["speculative"]
    }
    keys = sorted(baseline_rows)
    classes = sorted({r["prompt_class"] for r in data["rows"]})
    prompt_class = {r["prompt_id"]: r["prompt_class"] for r in data["rows"]}

    def total(rows: list[dict[str, Any]]) -> float:
        return sum(_predicted_ms(r) for r in rows)

    def speedup(config: str, key: tuple[str, int]) -> float | None:
        row = spec_rows[config].get(key)
        if row is None:
            return None
        return _speedup(row, baseline_rows[key])

    always_base = total(list(baseline_rows.values()))
    best_fixed = _best_fixed_spec(data)
    always_spec = total(list(spec_rows[best_fixed].values()))
    oracle = 0.0
    oracle_spec = 0
    for key in keys:
        choices = [_predicted_ms(baseline_rows[key])]
        choices += [_predicted_ms(spec_rows[c][key]) for c in spec_configs]
        best = min(choices)
        oracle += best
        if best < choices[0]:
            oracle_spec += 1

    class_policy_rows = []
    class_policy_spec = 0
    class_choices: dict[str, int] = {}
    for key in keys:
        cls = prompt_class[key[0]]
        best_config = None
        best_mean = 1.0
        for config in spec_configs:
            others = [
                speedup(config, other)
                for other in keys
                if other[0] != key[0] and prompt_class[other[0]] == cls
            ]
            mean_speedup = _mean([x for x in others if x is not None])
            if mean_speedup is not None and mean_speedup > best_mean:
                best_mean = mean_speedup
                best_config = config
        if best_config is None:
            class_policy_rows.append(baseline_rows[key])
        else:
            class_policy_spec += 1
            class_choices[best_config] = class_choices.get(best_config, 0) + 1
            class_policy_rows.append(spec_rows[best_config][key])

    ewma = {cls: {config: 1.0 for config in spec_configs} for cls in classes}
    ewma_rows = []
    ewma_spec = 0
    ewma_choices: dict[str, int] = {}
    ordered = list(keys)
    random.Random(42).shuffle(ordered)
    for key in ordered:
        cls = prompt_class[key[0]]
        best_config = max(spec_configs, key=lambda c: ewma[cls][c])
        use_spec = ewma[cls][best_config] > 1.0
        if use_spec:
            ewma_spec += 1
            ewma_choices[best_config] = ewma_choices.get(best_config, 0) + 1
            ewma_rows.append(spec_rows[best_config][key])
        else:
            ewma_rows.append(baseline_rows[key])
        for config in spec_configs:
            observed = speedup(config, key)
            if observed is not None:
                ewma[cls][config] = 0.7 * ewma[cls][config] + 0.3 * observed

    return {
        "best_fixed_config": best_fixed,
        "always_baseline_ms": always_base,
        "always_spec_ms": always_spec,
        "oracle_ms": oracle,
        "oracle_spec_requests": oracle_spec,
        "class_policy_ms": total(class_policy_rows),
        "class_policy_spec_requests": class_policy_spec,
        "class_policy_choices": class_choices,
        "ewma_policy_ms": total(ewma_rows),
        "ewma_policy_spec_requests": ewma_spec,
        "ewma_policy_choices": ewma_choices,
        "requests": len(keys),
    }


def render_markdown(data: dict[str, Any], source: str) -> str:
    configs = _configs(data)
    summaries = [summarize_config(data, c) for c in configs]
    summary_by_name = {s["config"]: s for s in summaries}
    baseline = summary_by_name.get("baseline") or summaries[0]
    lines: list[str] = []
    lines.append(f"Source: `{source}`")
    lines.append("")
    lines.append(
        f"Target `{data['models']['target']['path'].split('/')[-1]}`, draft "
        f"`{data['models']['draft']['path'].split('/')[-1]}`, "
        f"llama.cpp {data['llama_cpp']['version']} ({data['llama_cpp']['commit']})."
    )
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append(
        "| Config | Decode tok/s | Speedup vs baseline | Acceptance | Mean accepted len | First token ms | VRAM GiB | Exact output |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        speedup = s["speedup_mean"]
        speedup_s = f"{speedup:.3f}x" if speedup is not None else "1.000x"
        acc = s["acceptance_mean"]
        acc_s = f"{acc * 100:.1f}%" if acc is not None else "-"
        alen = s["mean_accepted_length"]
        alen_s = f"{alen:.2f}" if alen is not None else "-"
        ft = s["first_token_ms_median"]
        ft_s = f"{ft:.0f}" if ft is not None else "-"
        vram = s["vram_delta_bytes"]
        vram_s = f"{vram / (1024**3):.2f}" if vram else "-"
        tps = s["decode_tps_mean"]
        tps_s = f"{tps:.1f}" if tps is not None else "-"
        exact_s = (
            f"{s['outputs_exact']}/{s['outputs_compared']}"
            if s["outputs_compared"]
            else "-"
        )
        lines.append(
            f"| {s['config']} | {tps_s} | {speedup_s} | {acc_s} | {alen_s} | {ft_s} | {vram_s} | {exact_s} |"
        )

    spec_configs = _spec_configs(data)
    if spec_configs:
        lines.append("")
        lines.append("### Acceptance by draft position")
        lines.append("")
        header = "| Config | " + " | ".join(f"p{i}" for i in range(8)) + " |"
        lines.append(header)
        lines.append("|---|" + "---:|" * 8)
        for config in spec_configs:
            positions = acceptance_by_position(data, config)
            cells = []
            for i in range(8):
                value = positions.get(i)
                cells.append(f"{value * 100:.0f}%" if value is not None else "-")
            lines.append(f"| {config} | " + " | ".join(cells) + " |")

        lines.append("")
        lines.append("### Per prompt class")
        lines.append("")
        lines.append("| Class | Config | Speedup | Acceptance | Requests |")
        lines.append("|---|---|---:|---:|---:|")
        for config in spec_configs:
            for cls, row in sorted(per_class(data, config).items()):
                speedup = row["speedup_mean"]
                acc = row["acceptance_mean"]
                speedup_s = f"{speedup:.3f}x" if speedup is not None else "-"
                acc_s = f"{acc * 100:.1f}%" if acc is not None else "-"
                lines.append(
                    f"| {cls} | {config} | {speedup_s} | {acc_s} | {row['requests']} |"
                )

        lines.append("")
        lines.append("### Break-even fit")
        lines.append("")
        lines.append(
            "Linear fit of measured speedup against acceptance. A crossing above 1.0x "
            "is the acceptance needed for speculation to pay for itself at that draft depth."
        )
        lines.append("")
        lines.append("| Config | Slope | Intercept | R² | Acceptance at 1.0x | Samples |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for config, fit in break_even(data).items():
            if fit is None:
                lines.append(f"| {config} | - | - | - | - | 0 |")
                continue
            crossing = fit["acceptance_at_break_even"]
            crossing_s = f"{crossing * 100:.1f}%" if crossing is not None else "-"
            lines.append(
                f"| {config} | {fit['slope']:.3f} | {fit['intercept']:.3f} | "
                f"{fit['r2']:.3f} | {crossing_s} | {fit['n']} |"
            )

        lines.append("")
        lines.append("### Router replay")
        lines.append("")
        router = router_replay(data)
        lines.append(
            f"Best fixed speculative config is `{router['best_fixed_config']}`. "
            f"{router['requests']} requests, decode time only."
        )
        lines.append("")
        lines.append("| Policy | Total ms | vs always baseline | Spec requests |")
        lines.append("|---|---:|---:|---:|")
        base = router["always_baseline_ms"]
        for label, key, spec_key in (
            ("Always baseline", "always_baseline_ms", None),
            ("Always spec (best fixed)", "always_spec_ms", None),
            ("Oracle per prompt", "oracle_ms", "oracle_spec_requests"),
            ("Per-class, leave-one-out", "class_policy_ms", "class_policy_spec_requests"),
            ("Online EWMA", "ewma_policy_ms", "ewma_policy_spec_requests"),
        ):
            value = router[key]
            delta = (value - base) / base * 100 if base else 0.0
            spec_count = router[spec_key] if spec_key else "-"
            lines.append(f"| {label} | {value:.0f} | {delta:+.1f}% | {spec_count} |")
        if router.get("class_policy_choices"):
            lines.append("")
            lines.append(
                "Per-class choices: "
                + ", ".join(f"`{k}` x{v}" for k, v in sorted(router["class_policy_choices"].items()))
                + "."
            )
        if router.get("ewma_policy_choices"):
            lines.append(
                "Online EWMA choices: "
                + ", ".join(f"`{k}` x{v}" for k, v in sorted(router["ewma_policy_choices"].items()))
                + "."
            )

    lines.append("")
    lines.append(
        f"Validation: {data['validation']['outputs_checked']} outputs compared, "
        f"{data['validation']['output_mismatches']} mismatches."
    )
    return "\n".join(lines) + "\n"
