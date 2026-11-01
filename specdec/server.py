"""Start and stop llama-server for one sweep configuration."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config, SweepConfig


class ServerError(RuntimeError):
    pass


def _http_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _http_text(url: str, timeout: float = 2.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return None
            return response.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


class LlamaServer:
    def __init__(
        self,
        config: Config,
        sweep: SweepConfig,
        llama_bin: Path,
        log_path: Path,
    ) -> None:
        self.config = config
        self.sweep = sweep
        self.llama_bin = llama_bin
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log_handle = None
        self.vram_idle: int | None = None
        self.vram_peak: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.server.host}:{self.config.server.port}"

    def command(self) -> list[str]:
        cfg = self.config
        srv = cfg.server
        args = [
            str(self.llama_bin / "llama-server"),
            "-m", str(cfg.target.path),
            "--host", srv.host,
            "--port", str(srv.port),
            "-c", str(srv.context),
            "-np", str(srv.parallel),
            "-ngl", str(srv.gpu_layers),
            "-fa", srv.flash_attn,
            "-ctk", srv.kv_type,
            "-ctv", srv.kv_type,
            "--metrics",
        ]
        if self.sweep.speculation:
            args += [
                "--spec-draft-model", str(cfg.draft.path),
                "--spec-draft-ngl", str(srv.draft_gpu_layers),
                "--spec-type", "draft-simple",
                "--spec-draft-n-max", str(self.sweep.n_max),
            ]
            if self.sweep.p_min is not None:
                args += ["--spec-draft-p-min", str(self.sweep.p_min)]
        return args

    def __enter__(self) -> "LlamaServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w")
        self.proc = subprocess.Popen(
            self.command(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ServerError(
                    f"llama-server exited with {self.proc.returncode}; see {self.log_path}"
                )
            if _http_json(f"{self.base_url}/health") is not None:
                # Health can flip before the first token is ready; a tiny
                # completion is the real readiness gate.
                if self._warmup():
                    self.vram_idle = read_vram()
                    return self
            time.sleep(0.5)
        raise ServerError(f"llama-server did not become ready; see {self.log_path}")

    def _warmup(self) -> bool:
        body = json.dumps(
            {
                "prompt": "Reply with the single word ready.",
                "n_predict": 4,
                "temperature": 0,
                "top_k": 1,
                "seed": 1,
                "cache_prompt": False,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def metrics(self) -> str | None:
        return _http_text(f"{self.base_url}/metrics")

    def __exit__(self, *_exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()


def read_vram() -> int | None:
    if shutil.which("rocm-smi") is None:
        return None
    proc = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    for card in data.values():
        if not isinstance(card, dict):
            continue
        for key, value in card.items():
            if "Used Memory" in key and "B" in key:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None
