"""Run the speculative-decoding sweep and write one artifact."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import completion, generated_text, timing_row
from .config import Config
from .metrics import parse_metrics, request_spec_metrics
from .server import LlamaServer, read_vram


SCHEMA = "specdec/v1"


class RunError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if shutil.which("aria2c"):
        subprocess.run(
            [
                "aria2c", "-x", "8", "-s", "8", "-k", "1M",
                "--file-allocation=none", "-o", tmp.name, "-d", str(tmp.parent), url,
            ],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def fetch(config: Config) -> None:
    for model in (config.target, config.draft):
        if model.path.exists() and model.path.stat().st_size > 0:
            print(f"have {model.path} ({model.path.stat().st_size} bytes)")
            continue
        if not model.url:
            raise RunError(f"{model.path} is missing and has no url")
        print(f"fetch {model.path}")
        _download(model.url, model.path)
        print(f"  wrote {model.path} ({model.path.stat().st_size} bytes)")


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise RunError(f"{path} must contain a non-empty JSON array")
    for prompt in prompts:
        for key in ("id", "class", "prompt"):
            if key not in prompt:
                raise RunError(f"prompt missing {key!r}: {prompt}")
    return prompts


def _hardware() -> dict[str, Any]:
    gpu = None
    proc = subprocess.run(
        ["bash", "-lc", "lspci | grep -Ei 'vga|display'"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        gpu = proc.stdout.strip()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def _version(llama_bin: Path, log_dir: Path) -> dict[str, str]:
    proc = subprocess.run(
        [str(llama_bin / "llama-cli"), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = proc.stdout + proc.stderr
    (log_dir / "llama-version.log").write_text(text)
    import re

    version = re.search(r"version:\s*([^\s]+)", text)
    build = re.search(r"build\s+(\d+)", text)
    commit = re.search(r"commit\s+([0-9a-f]+)", text)
    return {
        "version": version.group(1) if version else "unknown",
        "build": build.group(1) if build else "unknown",
        "commit": commit.group(1) if commit else "unknown",
    }


def run_sweep(
    config: Config,
    *,
    llama_bin: Path,
    out_dir: Path,
    run_id: str | None = None,
    strict: bool = False,
) -> Path:
    prompts = load_prompts(config.prompts_path)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = out_dir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    version = _version(llama_bin, log_dir)

    rows: list[dict[str, Any]] = []
    server_meta: dict[str, Any] = {}
    baseline_text: dict[tuple[str, int], str] = {}
    baseline_rows: dict[tuple[str, int], dict[str, Any]] = {}

    for sweep in config.configs:
        log_path = log_dir / f"{sweep.name}.log"
        print(f"=== {sweep.name}: {'speculative' if sweep.speculation else 'baseline'}")
        vram_peak = None
        with LlamaServer(config, sweep, llama_bin, log_path) as server:
            rows_for_config: list[dict[str, Any]] = []
            for prompt in prompts:
                for run_index in range(config.run.runs):
                    before_text = server.metrics() or ""
                    payload, wall_ms = completion(
                        server.base_url,
                        prompt["prompt"],
                        max_tokens=config.run.max_tokens,
                        seed=config.run.seed + run_index,
                        temperature=config.run.temperature,
                    )
                    after_text = server.metrics() or ""
                    timings = timing_row(payload, wall_ms)
                    spec = request_spec_metrics(
                        parse_metrics(before_text), parse_metrics(after_text)
                    )
                    text = generated_text(payload)
                    row = {
                        "config": sweep.name,
                        "speculative": sweep.speculation,
                        "n_max": sweep.n_max,
                        "p_min": sweep.p_min,
                        "prompt_id": prompt["id"],
                        "prompt_class": prompt["class"],
                        "prompt": prompt["prompt"],
                        "run": run_index,
                        "seed": config.run.seed + run_index,
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "timings": timings,
                        "spec": spec,
                        "wall_ms": wall_ms,
                    }
                    rows.append(row)
                    rows_for_config.append(row)

                    if not sweep.speculation:
                        baseline_text[(prompt["id"], run_index)] = text
                        baseline_rows[(prompt["id"], run_index)] = timings
                    vram = read_vram()
                    if vram is not None and (vram_peak is None or vram > vram_peak):
                        vram_peak = vram

            server_meta[sweep.name] = {
                "command": server.command(),
                "log": str(log_path),
                "vram_idle_bytes": server.vram_idle,
                "vram_peak_bytes": vram_peak,
                "vram_delta_bytes": (
                    max(0, vram_peak - server.vram_idle)
                    if vram_peak is not None and server.vram_idle is not None
                    else None
                ),
            }
            print(
                f"    {len(rows_for_config)} requests, "
                f"vram peak {vram_peak / (1024**3):.2f} GiB"
                if vram_peak
                else f"    {len(rows_for_config)} requests"
            )

    mismatches = 0
    for row in rows:
        key = (row["prompt_id"], row["run"])
        if key not in baseline_text:
            continue
        row["matches_baseline"] = row["text"] == baseline_text[key]
        if not row["matches_baseline"]:
            mismatches += 1

    artifact = {
        "schema": SCHEMA,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name": config.name,
        "hardware": _hardware(),
        "llama_cpp": {**version, "bin_dir": str(llama_bin)},
        "models": {
            "target": {
                "path": str(config.target.path),
                "bytes": config.target.path.stat().st_size,
                "sha256": _sha256(config.target.path),
            },
            "draft": {
                "path": str(config.draft.path),
                "bytes": config.draft.path.stat().st_size,
                "sha256": _sha256(config.draft.path),
            },
        },
        "run": {
            "max_tokens": config.run.max_tokens,
            "runs": config.run.runs,
            "seed": config.run.seed,
            "temperature": config.run.temperature,
            "context": config.server.context,
            "parallel": config.server.parallel,
        },
        "servers": server_meta,
        "rows": rows,
        "validation": {
            "outputs_checked": sum(1 for r in rows if "matches_baseline" in r),
            "output_mismatches": mismatches,
        },
    }
    out_path = out_dir / f"spec_decode_{config.name}_{run_id}.json"
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {out_path}")
    if strict and mismatches:
        raise RunError(
            f"{mismatches} generated outputs differ from the greedy baseline; "
            f"artifact kept at {out_path} for inspection"
        )
    return out_path
