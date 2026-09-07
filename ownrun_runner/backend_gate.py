"""Exclusive Ollama ↔ Stable Diffusion backend gate."""

from __future__ import annotations

import logging
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Callable

import requests

from ownrun_runner.config import Config

log = logging.getLogger(__name__)


class Backend(str, Enum):
    OLLAMA = "ollama"
    SD = "stablediffusion"


class BackendState(str, Enum):
    ACTIVE_OLLAMA = "active_ollama"
    ACTIVE_SD = "active_sd"
    SWITCHING = "switching"
    FAILED = "failed"


ProgressFn = Callable[[dict], None] | None
CancelFn = Callable[[], bool] | None


class BackendGate:
    """Serialize GPU ownership: at most one of Ollama or Forge may be running."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = BackendState.ACTIVE_OLLAMA
        self._last_ollama_models: list[str] = []
        self._last_sd_models: list[str] = []
        self._ollama_models_stale = False

    @property
    def enabled(self) -> bool:
        return self.config.sd_enabled

    @property
    def sd_available(self) -> bool:
        if not self.config.sd_enabled:
            return False
        return all(
            self._script_ok(p)
            for p in (
                self.config.ollama_start_script,
                self.config.ollama_stop_script,
                self.config.sd_start_script,
                self.config.sd_stop_script,
            )
        )

    def _script_ok(self, path: str) -> bool:
        if not path:
            return False
        p = Path(path)
        return p.is_file() and os_access_executable(p)

    def boot(self) -> None:
        """On runner start with gate enabled, bring up the idle backend (Ollama)."""
        if not self.enabled:
            return
        if not self.sd_available:
            log.warning(
                "OWNRUN_SD_ENABLED=1 but start/stop scripts are missing or not executable; "
                "image jobs disabled until configured"
            )
            self.state = BackendState.FAILED
            return
        try:
            idle = self.config.backend_idle
            required = Backend.OLLAMA if idle == "ollama" else Backend.SD
            self.ensure(required)
        except Exception as e:
            log.error("Backend gate boot failed: %s", e)
            self.state = BackendState.FAILED

    def ensure(
        self,
        required: Backend,
        *,
        on_progress: ProgressFn = None,
        should_cancel: CancelFn = None,
    ) -> float:
        """Ensure *required* backend is healthy. Returns switch duration ms (0 if already)."""
        if not self.enabled:
            if required == Backend.SD:
                raise RuntimeError(
                    "Stable Diffusion is not enabled on this runner (set OWNRUN_SD_ENABLED=1)"
                )
            return 0.0

        if not self.sd_available:
            raise RuntimeError(
                "Image backend scripts are not configured. Set OWNRUN_OLLAMA_START_SCRIPT, "
                "OWNRUN_OLLAMA_STOP_SCRIPT, OWNRUN_SD_START_SCRIPT, and OWNRUN_SD_STOP_SCRIPT "
                "to executable paths."
            )

        start = time.monotonic()
        if required == Backend.OLLAMA and self.state == BackendState.ACTIVE_OLLAMA:
            if self._ollama_healthy() and not self._sd_healthy():
                return 0.0
        if required == Backend.SD and self.state == BackendState.ACTIVE_SD:
            if self._sd_healthy() and not self._ollama_healthy():
                return 0.0

        self.state = BackendState.SWITCHING
        try:
            if required == Backend.SD:
                self._switch_to_sd(on_progress=on_progress, should_cancel=should_cancel)
            else:
                self._switch_to_ollama(on_progress=on_progress, should_cancel=should_cancel)
            self.state = (
                BackendState.ACTIVE_SD if required == Backend.SD else BackendState.ACTIVE_OLLAMA
            )
        except Exception:
            self.state = BackendState.FAILED
            raise

        return (time.monotonic() - start) * 1000

    def _switch_to_sd(
        self,
        *,
        on_progress: ProgressFn,
        should_cancel: CancelFn,
    ) -> None:
        self._check_cancel(should_cancel)
        self._emit(on_progress, {"phase": "stopping_ollama"})
        log.info("stop_ollama_begin")
        t0 = time.monotonic()
        if self._ollama_healthy():
            self._run_script(self.config.ollama_stop_script, "stop Ollama")
            self._wait_until(False, self._ollama_healthy, "Ollama", stop=True)
        log.info("stop_ollama_ok duration_ms=%s", int((time.monotonic() - t0) * 1000))
        self._ollama_models_stale = True

        self._check_cancel(should_cancel)
        self._emit(on_progress, {"phase": "starting_stable_diffusion"})
        log.info("start_sd_begin")
        t1 = time.monotonic()
        if not self._sd_healthy():
            self._run_script(self.config.sd_start_script, "start Stable Diffusion")
            self._wait_until(True, self._sd_healthy, "Stable Diffusion", stop=False)
        log.info("start_sd_ok duration_ms=%s", int((time.monotonic() - t1) * 1000))
        self._refresh_sd_models()

    def _switch_to_ollama(
        self,
        *,
        on_progress: ProgressFn,
        should_cancel: CancelFn,
    ) -> None:
        self._check_cancel(should_cancel)
        self._emit(on_progress, {"phase": "stopping_stable_diffusion"})
        log.info("stop_sd_begin")
        t0 = time.monotonic()
        if self._sd_healthy():
            self._run_script(self.config.sd_stop_script, "stop Stable Diffusion")
            self._wait_until(False, self._sd_healthy, "Stable Diffusion", stop=True)
        log.info("stop_sd_ok duration_ms=%s", int((time.monotonic() - t0) * 1000))

        self._check_cancel(should_cancel)
        self._emit(on_progress, {"phase": "starting_ollama"})
        log.info("start_ollama_begin")
        t1 = time.monotonic()
        if not self._ollama_healthy():
            self._run_script(self.config.ollama_start_script, "start Ollama")
            self._wait_until(True, self._ollama_healthy, "Ollama", stop=False)
        log.info("start_ollama_ok duration_ms=%s", int((time.monotonic() - t1) * 1000))
        self._ollama_models_stale = False

    def _run_script(self, path: str, label: str) -> None:
        if not self._script_ok(path):
            raise RuntimeError(f"Cannot {label}: script missing or not executable: {path}")
        log.info("Running %s script: %s", label, path)
        try:
            proc = subprocess.run(
                [path],
                capture_output=True,
                text=True,
                timeout=self.config.backend_script_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Script to {label} timed out after {self.config.backend_script_timeout}s"
            ) from e
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if out:
            log.info("Script %s output (truncated): %s", label, out[:2000])
        if proc.returncode != 0:
            raise RuntimeError(
                f"Script to {label} exited {proc.returncode}"
                + (f": {out[:500]}" if out else "")
            )

    def _wait_until(
        self,
        want_healthy: bool,
        check: Callable[[], bool],
        name: str,
        *,
        stop: bool,
    ) -> None:
        timeout = (
            self.config.backend_stop_timeout
            if stop
            else self.config.backend_start_timeout
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok = check()
            if want_healthy and ok:
                return
            if not want_healthy and not ok:
                return
            time.sleep(1.0)
        if want_healthy:
            raise RuntimeError(f"Could not start {name} within {int(timeout)}s")
        raise RuntimeError(f"Could not stop {name} within {int(timeout)}s")

    def _ollama_healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False

    def _sd_healthy(self) -> bool:
        try:
            resp = requests.get(
                f"{self.config.sd_base_url}/sdapi/v1/sd-models",
                timeout=5,
            )
            return resp.ok
        except requests.RequestException:
            return False

    def _refresh_sd_models(self) -> None:
        try:
            resp = requests.get(
                f"{self.config.sd_base_url}/sdapi/v1/sd-models",
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            names: list[str] = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        title = item.get("title") or item.get("model_name") or item.get("name")
                        if title:
                            names.append(str(title))
            self._last_sd_models = sorted(set(names))
        except requests.RequestException as e:
            log.warning("Could not list SD models: %s", e)

    def note_ollama_models(self, models: list[str]) -> None:
        self._last_ollama_models = list(models)
        self._ollama_models_stale = False

    def note_sd_models(self, models: list[str]) -> None:
        self._last_sd_models = list(models)

    def heartbeat_fields(self) -> dict:
        """Fields for runner heartbeat (always include when gate aware)."""
        state = self.state.value
        if not self.enabled:
            return {
                "backend_state": "active_ollama",
                "sd_available": False,
                "sd_models": [],
                "ollama_models_stale": False,
            }
        return {
            "backend_state": state,
            "sd_available": self.sd_available,
            "sd_models": list(self._last_sd_models),
            "ollama_models_stale": self._ollama_models_stale,
        }

    def should_probe_ollama(self) -> bool:
        if not self.enabled:
            return True
        return self.state == BackendState.ACTIVE_OLLAMA

    def should_probe_sd(self) -> bool:
        return self.enabled and self.sd_available and self.state == BackendState.ACTIVE_SD

    def cached_ollama_models(self) -> list[str]:
        return list(self._last_ollama_models)

    @staticmethod
    def _emit(on_progress: ProgressFn, payload: dict) -> None:
        if on_progress:
            try:
                on_progress(payload)
            except Exception:
                pass

    @staticmethod
    def _check_cancel(should_cancel: CancelFn) -> None:
        if should_cancel and should_cancel():
            raise RuntimeError("cancelled")


def os_access_executable(path: Path) -> bool:
    import os

    return os.access(path, os.X_OK)
