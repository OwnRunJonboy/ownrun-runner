"""Stable Diffusion (Forge) txt2img adapter."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import requests

from ownrun_runner.client import ServerClient
from ownrun_runner.config import Config

log = logging.getLogger(__name__)

ProgressFn = Callable[[dict[str, Any]], None] | None
CancelFn = Callable[[], bool] | None


def list_sd_models(config: Config) -> list[str]:
    try:
        resp = requests.get(f"{config.sd_base_url}/sdapi/v1/sd-models", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        names: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("model_name") or item.get("name")
                    if title:
                        names.append(str(title))
        return sorted(set(names))
    except requests.RequestException as e:
        log.warning("Could not list SD models: %s", e)
        return []


def _select_checkpoint(config: Config, model: str) -> None:
    """Set Forge/A1111 sd_model_checkpoint if a model was requested."""
    if not model:
        return
    try:
        resp = requests.get(f"{config.sd_base_url}/sdapi/v1/options", timeout=30)
        resp.raise_for_status()
        options = resp.json()
        current = str(options.get("sd_model_checkpoint") or "")
        if model in current or current == model:
            return
        resp = requests.post(
            f"{config.sd_base_url}/sdapi/v1/options",
            json={"sd_model_checkpoint": model},
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Could not select checkpoint {model!r}: HTTP {resp.status_code} {resp.text[:300]}"
            )
    except requests.RequestException as e:
        raise RuntimeError(f"Could not select checkpoint {model!r}: {e}") from e


def run_txt2img(
    config: Config,
    request: dict[str, Any],
    *,
    job_id: int,
    client: ServerClient,
    backend_switch_ms: float = 0.0,
    on_progress: ProgressFn = None,
    should_cancel: CancelFn = None,
) -> dict[str, Any]:
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Job request missing prompt")

    negative = str(request.get("negative_prompt") or "")
    model = str(request.get("model") or "").strip()
    width = int(request.get("width") or 1024)
    height = int(request.get("height") or 1024)
    steps = int(request.get("steps") or 30)
    cfg_scale = float(request.get("cfg_scale") or 7)
    seed = int(request.get("seed") if request.get("seed") is not None else -1)
    batch_size = int(request.get("batch_size") or 1)

    if should_cancel and should_cancel():
        raise RuntimeError("cancelled")

    if model:
        _select_checkpoint(config, model)

    body = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "batch_size": batch_size,
        "n_iter": 1,
    }

    if on_progress:
        on_progress({"phase": "generating"})

    start = time.monotonic()
    try:
        resp = requests.post(
            f"{config.sd_base_url}/sdapi/v1/txt2img",
            json=body,
            timeout=config.sd_job_timeout,
        )
    except requests.Timeout as e:
        raise RuntimeError(
            f"Stable Diffusion timed out after {config.sd_job_timeout}s"
        ) from e
    except requests.RequestException as e:
        raise RuntimeError(f"Stable Diffusion unreachable: {e}") from e

    if should_cancel and should_cancel():
        raise RuntimeError("cancelled")

    if not resp.ok:
        raise RuntimeError(f"Stable Diffusion error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    images_b64 = data.get("images") or []
    if not images_b64:
        raise RuntimeError("Stable Diffusion returned no images")

    info = data.get("info")
    used_seed = seed
    used_model = model
    if isinstance(info, str):
        try:
            info_obj = json.loads(info)
            if isinstance(info_obj, dict):
                if "seed" in info_obj:
                    used_seed = int(info_obj["seed"])
                if info_obj.get("sd_model_name"):
                    used_model = str(info_obj["sd_model_name"])
                elif info_obj.get("sd_model_checkpoint"):
                    used_model = str(info_obj["sd_model_checkpoint"])
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    duration_ms = int((time.monotonic() - start) * 1000)

    if on_progress:
        on_progress({"phase": "uploading"})

    uploaded: list[dict[str, Any]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"ownrun-sd-{job_id}-"))
    try:
        for i, b64 in enumerate(images_b64):
            if should_cancel and should_cancel():
                raise RuntimeError("cancelled")
            raw_b64 = str(b64).split(",", 1)[0]
            try:
                raw = base64.b64decode(raw_b64)
            except Exception as e:
                raise RuntimeError(f"Invalid image data from Stable Diffusion: {e}") from e
            filename = f"{i + 1:05d}.png"
            path = tmp_dir / filename
            path.write_bytes(raw)
            artifact = client.upload_artifact(
                job_id,
                path,
                content_type="image/png",
                filename=filename,
            )
            uploaded.append(
                {
                    "artifact_id": int(artifact["id"]),
                    "filename": str(artifact.get("filename") or filename),
                    "content_type": str(artifact.get("content_type") or "image/png"),
                }
            )
    finally:
        for p in tmp_dir.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return {
        "seed": used_seed,
        "model": used_model,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "duration_ms": duration_ms,
        "backend_switch_ms": int(backend_switch_ms),
        "images": uploaded,
    }
