"""Configuration for a target/draft pair and the sweep grid."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path.home() / "models" / "spec-lab"
DEFAULT_PROMPTS = Path("prompts/prompts.json")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    path: Path


@dataclass(frozen=True)
class ServerSpec:
    context: int
    parallel: int
    gpu_layers: int
    draft_gpu_layers: int
    flash_attn: str
    kv_type: str
    host: str
    port: int


@dataclass(frozen=True)
class RunSpec:
    max_tokens: int
    runs: int
    seed: int
    temperature: float


@dataclass(frozen=True)
class SweepConfig:
    name: str
    n_max: int | None = None
    p_min: float | None = None

    @property
    def speculation(self) -> bool:
        return self.n_max is not None


@dataclass(frozen=True)
class Config:
    name: str
    target: ModelSpec
    draft: ModelSpec
    server: ServerSpec
    run: RunSpec
    configs: tuple[SweepConfig, ...]
    prompts_path: Path

    def sweep(self, name: str) -> SweepConfig:
        for cfg in self.configs:
            if cfg.name == name:
                return cfg
        raise ConfigError(f"unknown sweep config {name!r}")


def _required(obj: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in obj:
        raise ConfigError(f"missing required key {key!r}")
    value = obj[key]
    if not isinstance(value, kind):
        raise ConfigError(f"{key!r} must be {kind}, got {type(value).__name__}")
    return value


def _model(obj: dict[str, Any], root: Path) -> ModelSpec:
    filename = _required(obj, "filename", str)
    return ModelSpec(
        filename=filename,
        url=obj.get("url", ""),
        path=root / filename,
    )


def load_config(path: str | Path, model_dir: str | Path | None = None) -> Config:
    manifest = Path(path)
    data = json.loads(manifest.read_text())
    root = Path(
        model_dir or os.environ.get("SPECLAB_MODEL_DIR") or DEFAULT_MODEL_DIR
    ).expanduser()

    server = data.get("server", {})
    run = data.get("run", {})
    raw_configs = _required(data, "configs", list)
    if not raw_configs:
        raise ConfigError("configs must not be empty")
    configs = []
    for raw in raw_configs:
        spec = raw.get("spec")
        configs.append(
            SweepConfig(
                name=_required(raw, "name", str),
                n_max=int(spec["n_max"]) if spec and "n_max" in spec else None,
                p_min=float(spec["p_min"]) if spec and "p_min" in spec else None,
            )
        )
    names = [c.name for c in configs]
    if len(names) != len(set(names)):
        raise ConfigError(f"duplicate config names: {names}")
    if not any(not c.speculation for c in configs):
        raise ConfigError("the sweep needs at least one baseline config (spec: null)")

    prompts = Path(os.environ.get("SPECLAB_PROMPTS", DEFAULT_PROMPTS))
    return Config(
        name=_required(data, "name", str),
        target=_model(_required(data, "target", dict), root),
        draft=_model(_required(data, "draft", dict), root),
        server=ServerSpec(
            context=int(server.get("context", 4096)),
            parallel=int(server.get("parallel", 1)),
            gpu_layers=int(server.get("gpu_layers", 99)),
            draft_gpu_layers=int(server.get("draft_gpu_layers", 99)),
            flash_attn=str(server.get("flash_attn", "on")),
            kv_type=str(server.get("kv_type", "f16")),
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8091)),
        ),
        run=RunSpec(
            max_tokens=int(run.get("max_tokens", 128)),
            runs=int(run.get("runs", 3)),
            seed=int(run.get("seed", 7)),
            temperature=float(run.get("temperature", 0.0)),
        ),
        configs=tuple(configs),
        prompts_path=prompts,
    )
