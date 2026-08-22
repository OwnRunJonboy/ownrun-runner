from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ownrun_runner.config import Config

log = logging.getLogger(__name__)


def list_models(config: Config) -> list[str]:
    try:
        resp = requests.get(f"{config.ollama_base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", []) if "name" in m]
        return sorted(models)
    except requests.RequestException as e:
        log.warning("Could not list Ollama models: %s", e)
        return []


def run_ollama_chat(config: Config, request: dict[str, Any]) -> dict[str, Any]:
    model = request.get("model")
    prompt = request.get("prompt")
    if not model or not prompt:
        raise ValueError("Job request missing model or prompt")

    messages: list[dict[str, str]] = []
    system = request.get("system_prompt")
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(prompt)})

    body = {
        "model": str(model),
        "messages": messages,
        "stream": False,
    }

    start = time.monotonic()
    try:
        resp = requests.post(
            f"{config.ollama_base_url}/api/chat",
            json=body,
            timeout=config.job_timeout,
        )
    except requests.Timeout as e:
        raise RuntimeError(f"Ollama request timed out after {config.job_timeout}s") from e
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama unreachable: {e}") from e

    if resp.status_code == 404:
        raise RuntimeError(f"Model not found: {model}")
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Ollama error {resp.status_code}: {detail}")

    data = resp.json()
    message = data.get("message") or {}
    text = message.get("content", "")
    duration_ms = int((time.monotonic() - start) * 1000)

    return {
        "text": text,
        "model": str(model),
        "duration_ms": duration_ms,
    }
