from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Any

from pathlib import Path

from ownrun_runner.agent import agent_capable_models, run_agent_task
from ownrun_runner.agent.tools import all_tool_ids, list_projects
from ownrun_runner.backend_gate import Backend, BackendGate
from ownrun_runner.client import ServerClient
from ownrun_runner.config import Config
from ownrun_runner.kb import run_kb_delete, run_kb_ingest
from ownrun_runner.ollama import list_models, run_ollama_chat
from ownrun_runner.stablediffusion import list_sd_models, run_txt2img

log = logging.getLogger(__name__)

_stop = threading.Event()


def _handle_signal(signum: int, _frame: Any) -> None:
    log.info("Received signal %s, shutting down…", signum)
    _stop.set()


def _heartbeat_loop(client: ServerClient, config: Config, gate: BackendGate) -> None:
    while not _stop.is_set():
        try:
            ollama_stale = False
            if gate.should_probe_ollama():
                models = list_models(config)
                if models:
                    gate.note_ollama_models(models)
            else:
                models = gate.cached_ollama_models()
                ollama_stale = True

            if gate.should_probe_sd():
                sd_models = list_sd_models(config)
                if sd_models:
                    gate.note_sd_models(sd_models)

            agent_models = agent_capable_models(models) if models else []
            projects = list_projects(Path(config.agent_workspace))
            search_ready = config.agent_web_search and bool(config.ollama_api_key)
            hb = gate.heartbeat_fields()
            client.heartbeat(
                models,
                agent_profiles=list(config.agent_profiles),
                agent_models=agent_models,
                agent_projects=projects,
                timezone=config.timezone,
                ollama_web_search=search_ready,
                agent_tool_ids=all_tool_ids(),
                backend_state=hb["backend_state"],
                sd_available=hb["sd_available"],
                sd_models=hb["sd_models"],
                ollama_models_stale=ollama_stale or hb["ollama_models_stale"],
            )
            log.debug(
                "Heartbeat sent (%d models, sd_available=%s, backend=%s)",
                len(models),
                hb["sd_available"],
                hb["backend_state"],
            )
        except Exception as e:
            log.warning("Heartbeat failed: %s", e)
        _stop.wait(config.poll_interval)


def _process_job(
    client: ServerClient,
    config: Config,
    gate: BackendGate,
    job: dict[str, Any],
) -> None:
    job_id = job["id"]
    job_type = job.get("type", "")
    request = job.get("request") or {}

    log.info("Processing job #%s type=%s model=%s", job_id, job_type, request.get("model"))

    stop_hb = threading.Event()
    cancel_flag = threading.Event()

    def heartbeat_worker() -> None:
        while not stop_hb.wait(30):
            try:
                if client.job_heartbeat(job_id):
                    log.info("Job #%s cancel requested", job_id)
                    cancel_flag.set()
            except Exception as e:
                log.warning("Job #%s heartbeat failed: %s", job_id, e)

    hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    hb_thread.start()

    def on_progress(partial: dict[str, Any]) -> None:
        if cancel_flag.is_set():
            return
        try:
            client.progress(job_id, partial)
        except Exception as e:
            log.warning("Job #%s progress failed: %s", job_id, e)

    try:
        if job_type == "ollama.chat":
            gate.ensure(
                Backend.OLLAMA,
                on_progress=on_progress,
                should_cancel=cancel_flag.is_set,
            )
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            result = run_ollama_chat(config, request)
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            client.complete(job_id, result)
            log.info("Job #%s completed", job_id)
        elif job_type == "agent.task":
            gate.ensure(
                Backend.OLLAMA,
                on_progress=on_progress,
                should_cancel=cancel_flag.is_set,
            )
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            try:
                result = run_agent_task(
                    config,
                    request,
                    job_id=job_id,
                    calendar_client=client,
                    on_progress=on_progress,
                    should_cancel=cancel_flag.is_set,
                )
            except RuntimeError as e:
                if str(e) == "cancelled" or cancel_flag.is_set():
                    log.info("Job #%s cancelled", job_id)
                    return
                raise

            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            client.complete(job_id, result)
            log.info(
                "Job #%s completed stop_reason=%s iterations=%s",
                job_id,
                result.get("stop_reason"),
                result.get("iterations"),
            )
        elif job_type == "kb.ingest":
            gate.ensure(
                Backend.OLLAMA,
                on_progress=on_progress,
                should_cancel=cancel_flag.is_set,
            )
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            try:
                result = run_kb_ingest(
                    config,
                    request,
                    job_id=job_id,
                    client=client,
                    on_progress=on_progress,
                    should_cancel=cancel_flag.is_set,
                )
            except RuntimeError as e:
                msg = str(e)
                if msg == "cancelled" or cancel_flag.is_set():
                    log.info("Job #%s cancelled", job_id)
                    return
                doc_id = int(request.get("document_id") or 0)
                if doc_id:
                    try:
                        client.kb_document_status(
                            doc_id, {"status": "failed", "error_message": msg[:2000]}
                        )
                    except Exception as status_err:
                        log.warning("Could not mark document failed: %s", status_err)
                raise

            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            client.complete(job_id, result)
            log.info("Job #%s kb.ingest completed chunks=%s", job_id, result.get("chunk_count"))
        elif job_type == "kb.delete":
            # No GPU backend required.
            result = run_kb_delete(
                config,
                request,
                job_id=job_id,
                client=client,
                should_cancel=cancel_flag.is_set,
            )
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            client.complete(job_id, result)
            log.info("Job #%s kb.delete completed", job_id)
        elif job_type == "stablediffusion.txt2img":
            switch_ms = gate.ensure(
                Backend.SD,
                on_progress=on_progress,
                should_cancel=cancel_flag.is_set,
            )
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            try:
                result = run_txt2img(
                    config,
                    request,
                    job_id=job_id,
                    client=client,
                    backend_switch_ms=switch_ms,
                    on_progress=on_progress,
                    should_cancel=cancel_flag.is_set,
                )
            except RuntimeError as e:
                if str(e) == "cancelled" or cancel_flag.is_set():
                    log.info("Job #%s cancelled", job_id)
                    return
                raise
            if cancel_flag.is_set():
                log.info("Job #%s cancelled; skipping complete", job_id)
                return
            client.complete(job_id, result)
            log.info(
                "Job #%s txt2img completed images=%s duration_ms=%s",
                job_id,
                len(result.get("images") or []),
                result.get("duration_ms"),
            )
        else:
            raise RuntimeError(f"Unsupported job type: {job_type}")
    except Exception as e:
        if cancel_flag.is_set():
            log.info("Job #%s cancelled during error path: %s", job_id, e)
            return
        log.error("Job #%s failed: %s", job_id, e)
        if job_type == "kb.ingest":
            doc_id = int(request.get("document_id") or 0)
            if doc_id:
                try:
                    client.kb_document_status(
                        doc_id, {"status": "failed", "error_message": str(e)[:2000]}
                    )
                except Exception as status_err:
                    log.warning("Could not mark document failed: %s", status_err)
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
    gate = BackendGate(config)
    log.info(
        "OwnRun runner started — server=%s poll=%ss workspace=%s profiles=%s timezone=%s sd=%s",
        config.server_url,
        config.poll_interval,
        config.agent_workspace,
        ",".join(config.agent_profiles),
        config.timezone,
        "on" if config.sd_enabled else "off",
    )

    gate.boot()

    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(client, config, gate), daemon=True
    )
    hb_thread.start()

    while not _stop.is_set():
        try:
            # Do not claim while a backend switch is in progress (should be rare:
            # switches run inside _process_job). Also skip claim when gate is mid-switch.
            if gate.enabled and gate.state.value == "switching":
                _stop.wait(config.poll_interval)
                continue
            job = client.claim()
            if job:
                _process_job(client, config, gate, job)
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
