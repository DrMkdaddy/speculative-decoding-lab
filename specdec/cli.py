"""Command line interface for the speculative-decoding lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import load, render_markdown
from .config import ConfigError, load_config
from .router import AcceptanceRouter, classify_prompt
from .run import RunError, fetch, load_prompts, run_sweep


DEFAULT_LLAMA_BIN = Path.home() / "llama.cpp" / "src-b11050" / "build" / "bin"


def _common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default="configs/qwen3-4b-0.6b.json",
        help="model-pair manifest",
    )
    parser.add_argument("--model-dir", default=None, help="override SPECLAB_MODEL_DIR")
    return parser


def _parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="specdec",
        description="Measure speculative decoding against a greedy baseline.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", parents=[common], help="download target and draft")
    fetch_p.add_argument("--force", action="store_true")

    run_p = sub.add_parser("run", parents=[common], help="run the sweep")
    run_p.add_argument("--llama-bin", default=str(DEFAULT_LLAMA_BIN))
    run_p.add_argument("--out-dir", default="results")
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument(
        "--no-strict",
        action="store_true",
        help="keep rows even if speculative output differs from the baseline",
    )

    report_p = sub.add_parser("report", parents=[common], help="render result JSON")
    report_p.add_argument("results", nargs="+")

    route_p = sub.add_parser("route", parents=[common], help="classify and route a prompt")
    route_p.add_argument("prompt", nargs="?", help="prompt text; stdin if omitted")

    check_p = sub.add_parser("check", parents=[common], help="validate local files")

    sub.add_parser("all", parents=[common], help="fetch, run, and report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config, args.model_dir)
    except (ConfigError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "fetch":
            fetch(config)
            return 0

        if args.command == "run":
            path = run_sweep(
                config,
                llama_bin=Path(args.llama_bin),
                out_dir=Path(args.out_dir),
                run_id=args.run_id,
                strict=not args.no_strict,
            )
            data = load(path)
            print(render_markdown(data, str(path)))
            return 0

        if args.command == "report":
            paths = [Path(p) for p in args.results]
            newest = max(paths, key=lambda p: p.stat().st_mtime)
            print(render_markdown(load(newest), str(newest)))
            return 0

        if args.command == "route":
            prompt = args.prompt
            if prompt is None:
                prompt = sys.stdin.read()
            router = AcceptanceRouter()
            cls = classify_prompt(prompt)
            decision = router.choose(prompt)
            print(json.dumps({
                "class": cls,
                "decision": decision,
                "threshold": router.threshold,
                "ewma_speedup": router.ewma.get(cls, router.initial),
            }, indent=2))
            return 0

        if args.command == "check":
            missing = [
                str(m.path) for m in (config.target, config.draft) if not m.path.exists()
            ]
            if not config.prompts_path.exists():
                missing.append(str(config.prompts_path))
            if missing:
                print("missing:", *missing, sep="\n  ")
                return 1
            prompts = load_prompts(config.prompts_path)
            print(
                f"config ok: {config.name}, {len(config.configs)} configurations, "
                f"{len(prompts)} prompts"
            )
            return 0

        if args.command == "all":
            fetch(config)
            path = run_sweep(
                config,
                llama_bin=Path(DEFAULT_LLAMA_BIN),
                out_dir=Path("results"),
            )
            print(render_markdown(load(path), str(path)))
            return 0
    except (RunError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2
