from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import Any

from ownrun_runner.client import ServerClient
from ownrun_runner.config import Config
from ownrun_runner.ollama import list_models, run_ollama_chat

log = logging.getLogger(__name__)

_stop = threading.Event()


def _handle_signal(signum: int, _frame: Any) -> None:
    log.info("Received signal %s, shutting down…", signum)
    _stop.set()


def _heartbeat_loop(client: ServerClient, config: Config) -> None:
    while not _stop.is_set():
        try:
            models = list_models(config)
            client.heartbeat(models)
            log.debug("Heartbeat sent (%d models)", len(models))
        except Exception as e:
            log.warning("Heartbeat failed: %s", e)
        _stop.wait(config.poll_interval)


def _process_job(client: ServerClient, config: Config, job: dict[str, Any]) -> None:
    job_id = job["id"]
    job_type = job.get("type", "")
    request = job.get("request") or {}

    log.info("Processing job #%s type=%s model=%s", job_id, job_type, request.get("model"))

    stop_hb = threading.Event()

    def heartbeat_worker() -> None:
        while not stop_hb.wait(30):
            try:
                client.job_heartbeat(job_id)
            except Exception as e:
                log.warning("Job #%s heartbeat failed: %s", job_id, e)

    hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    hb_thread.start()

    try:
        if job_type == "ollama.chat":
            result = run_ollama_chat(config, request)
        else:
            raise RuntimeError(f"Unsupported job type: {job_type}")

        client.complete(job_id, result)
        log.info("Job #%s completed", job_id)
    except Exception as e:
        log.error("Job #%s failed: %s", job_id, e)
        try:
            client.fail(job_id, str(e))
        except Exception as upload_err:
            log.error("Could not report failure for job #%s: %s", job_id, upload_err)
    finally:
        stop_hb.set()
        hb_thread.join(timeout=2)


def run(config: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client = ServerClient(config)
    log.info("OwnRun runner started — server=%s poll=%ss", config.server_url, config.poll_interval)

    hb_thread = threading.Thread(target=_heartbeat_loop, args=(client, config), daemon=True)
    hb_thread.start()

    while not _stop.is_set():
        try:
            job = client.claim()
            if job:
                _process_job(client, config, job)
            else:
                _stop.wait(config.poll_interval)
        except SystemExit:
            raise
        except Exception as e:
            log.warning("Poll/claim error: %s", e)
            _stop.wait(config.poll_interval)

    log.info("Runner stopped")


def main() -> None:
    config = Config.from_env()
    run(config)


if __name__ == "__main__":
    main()
