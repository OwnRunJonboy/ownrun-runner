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


_ALLOWED_ROLES = {"system", "user", "assistant"}


def messages_from_request(request: dict[str, Any]) -> list[dict[str, str]]:
    raw = request.get("messages")
    if isinstance(raw, list) and raw:
        messages: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", ""))
            content = item.get("content")
            if role not in _ALLOWED_ROLES or content is None:
                continue
            messages.append({"role": role, "content": str(content)})
        if messages:
            return messages

    messages = []
    system = request.get("system_prompt")
    if system:
        messages.append({"role": "system", "content": str(system)})
    prompt = request.get("prompt")
    if prompt:
        messages.append({"role": "user", "content": str(prompt)})
    return messages


def run_ollama_chat(config: Config, request: dict[str, Any]) -> dict[str, Any]:
    model = request.get("model")
    if not model:
        raise ValueError("Job request missing model")

    messages = messages_from_request(request)
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("Job request missing prompt or messages")

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
